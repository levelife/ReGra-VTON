import os
import gc
from pathlib import Path
from functools import lru_cache

try:
    import gradio_client.utils as _gradio_client_utils

    _original_json_schema_to_python_type = (
        _gradio_client_utils._json_schema_to_python_type
    )

    if not getattr(_gradio_client_utils, "_bool_schema_patch_applied", False):
        def _safe_json_schema_to_python_type(schema, defs):
            if isinstance(schema, bool):
                schema = {}
            return _original_json_schema_to_python_type(schema, defs)

        _gradio_client_utils._json_schema_to_python_type = (
            _safe_json_schema_to_python_type
        )
        _gradio_client_utils._bool_schema_patch_applied = True
except Exception as _schema_patch_error:
    print(f"[WARN] Gradio schema compatibility patch failed: {_schema_patch_error}")

import gradio as gr
import torch
from PIL import Image

import infer as backend


DEFAULT_TRANSFORMER_DIR = "model path"
DEFAULT_DENSEPOSE_DIR = "densepose path"
DEFAULT_SCHP_DIR = "schp path"
DEFAULT_INPAINT_MODEL = "black-forest-labs/FLUX.1-dev"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "bf16"

# Keep loaded FLUX models in memory. Re-loading the model for every click would
# be extremely expensive and can easily exhaust VRAM.
_MODEL_CACHE = {}
_MASKER_CACHE = {}


def _device():
    return DEFAULT_DEVICE if torch.cuda.is_available() else "cpu"


def _dtype(dtype_str):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str]


def _model_cache_key(
    transformer_dir,
    inpaint_model,
    densepose_dir,
    schp_dir,
    dtype_str,
    enable_rgja,
    enable_dual,
    enable_vae_tiling,
    max_sequence_length,
):
    return (
        str(Path(transformer_dir).expanduser().resolve()),
        str(inpaint_model),
        str(Path(densepose_dir).expanduser().resolve()),
        str(Path(schp_dir).expanduser().resolve()),
        str(dtype_str),
        bool(enable_rgja),
        bool(enable_dual),
        bool(enable_vae_tiling),
        int(max_sequence_length),
    )


def _load_models(
    transformer_dir,
    inpaint_model,
    device,
    dtype_str,
    densepose_dir,
    schp_dir,
    enable_rgja,
    enable_dual,
    enable_vae_tiling,
    max_sequence_length,
):
    """Build the exact model stack used by infer_backend.main()."""
    if max_sequence_length > 512:
        raise ValueError("max_sequence_length should be <= 512 for FLUX.")

    torch_dtype = _dtype(dtype_str)
    device_obj = torch.device(device)
    key = _model_cache_key(
        transformer_dir,
        inpaint_model,
        densepose_dir,
        schp_dir,
        dtype_str,
        enable_rgja,
        enable_dual,
        enable_vae_tiling,
        max_sequence_length,
    )

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    transformer_path, root_dir = backend.resolve_transformer_dir_and_root(
        transformer_dir
    )

    print(f"[Model] loading transformer: {transformer_path}")
    transformer = backend.FluxTransformer2DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
    )

    print(f"[Model] loading FLUX Fill: {inpaint_model}")
    pipe = backend.FluxFillPipeline.from_pretrained(
        inpaint_model,
        transformer=transformer,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    if enable_vae_tiling:
        backend.maybe_enable_vae_tiling(pipe)

    if enable_dual and device != "cpu":
        ok_dual = backend.maybe_load_dual_region(
            pipe.transformer,
            root_dir,
            device=device_obj,
            dtype=torch_dtype,
        )
        print(
            "[DualRegion] loaded"
            if ok_dual
            else "[DualRegion] checkpoint not found; continuing without it"
        )

    if enable_rgja and device != "cpu":
        ok_rgja = backend.maybe_load_rgja(
            pipe.transformer,
            root_dir,
            device=device_obj,
            dtype=torch_dtype,
        )
        print(
            "[RGJA] loaded"
            if ok_rgja
            else "[RGJA] checkpoint not found; continuing without it"
        )

    prompt_cache = backend.build_prompt_cache(
        pipe=pipe,
        device=device_obj,
        dtype=torch_dtype,
        max_sequence_length=int(max_sequence_length),
        custom_prompt="",
    )

    # The original inference script deliberately removes the text encoders after
    # prompt caching to reduce VRAM usage.
    backend.drop_text_encoders(pipe)
    print("[Prompt] text encoders deleted after caching.")

    project_root = Path(backend.__file__).resolve().parent
    densepose_path = Path(densepose_dir).expanduser().resolve()
    schp_path = Path(schp_dir).expanduser().resolve()

    masker_key = (str(densepose_path), str(schp_path), str(device))
    if masker_key not in _MASKER_CACHE:
        print("[Mask] loading AutoMasker...")
        _MASKER_CACHE[masker_key] = backend.AutoMaskerFromImage(
            densepose_ckpt_dir=str(densepose_path),
            schp_ckpt_dir=str(schp_path),
            device=device,
        )

    masker = _MASKER_CACHE[masker_key]
    bundle = (pipe, masker, prompt_cache)
    _MODEL_CACHE[key] = bundle
    return bundle


def _get_masker(densepose_dir, schp_dir, device):
    key = (
        str(Path(densepose_dir).expanduser().resolve()),
        str(Path(schp_dir).expanduser().resolve()),
        str(device),
    )
    if key not in _MASKER_CACHE:
        _MASKER_CACHE[key] = backend.AutoMaskerFromImage(
            densepose_ckpt_dir=key[0],
            schp_ckpt_dir=key[1],
            device=device,
        )
    return _MASKER_CACHE[key]


def _make_mask_overlay(person, mask):
    """Create a lightweight visualization without changing the inference mask."""
    base = person.convert("RGB")
    m = mask.convert("L")
    overlay = base.copy()
    # Red mask visualization; inference itself always uses the original L mask.
    tint = Image.new("RGB", base.size, (255, 0, 0))
    overlay = Image.composite(tint, base, m)
    return Image.blend(base, overlay, 0.35)


def _make_mask(person, part, width, height, densepose_dir, schp_dir):
    masker = _get_masker(densepose_dir, schp_dir, _device())
    mask = masker(
        person,
        part=part,
        out_size_hw=None,
        square_cloth_mask=False,
        return_labels=False,
    ).convert("L")
    mask = backend.binarize_mask(mask, width, height)

    if part == "shoe":
        mask = backend.refine_shoe_mask(mask)
        mask = backend.binarize_mask(mask, width, height)

    return mask


def generate_mask_fn(
    person_pil,
    garment_pil,
    part,
    width,
    height,
    densepose_dir,
    schp_dir,
):
    if person_pil is None:
        raise gr.Error("请先上传人物图片。")
    if garment_pil is None:
        raise gr.Error("请先上传服装/鞋子图片。")

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise gr.Error("Width / Height 必须大于 0。")

    try:
        person = person_pil.convert("RGB").resize((width, height))
        mask = _make_mask(
            person,
            part,
            width,
            height,
            densepose_dir,
            schp_dir,
        )
        masked = backend.build_masked_person_preview(person, mask)
        overlay = _make_mask_overlay(person, mask)
        return mask, masked, overlay
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Mask 生成失败：{type(e).__name__}: {e}")


def tryon_fn(
    person_pil,
    garment_pil,
    part,
    steps,
    guidance,
    seed,
    height,
    width,
    transformer_dir,
    inpaint_model,
    densepose_dir,
    schp_dir,
    dtype_str,
    enable_rgja,
    enable_dual,
    enable_vae_tiling,
    max_sequence_length,
    upper_occupy_mode,
    upper_torso_ratio,
    upper_sleeve_ratio,
):
    if person_pil is None:
        raise gr.Error("请先上传人物图片。")
    if garment_pil is None:
        raise gr.Error("请先上传服装/鞋子图片。")

    width, height = int(width), int(height)
    steps, seed = int(steps), int(seed)
    guidance = float(guidance)
    max_sequence_length = int(max_sequence_length)

    if width <= 0 or height <= 0:
        raise gr.Error("Width / Height 必须大于 0。")
    if steps <= 0:
        raise gr.Error("Inference Steps 必须大于 0。")

    device = _device()

    try:
        pipe, masker, prompt_cache = _load_models(
            transformer_dir=transformer_dir,
            inpaint_model=inpaint_model,
            device=device,
            dtype_str=dtype_str,
            densepose_dir=densepose_dir,
            schp_dir=schp_dir,
            enable_rgja=bool(enable_rgja),
            enable_dual=bool(enable_dual),
            enable_vae_tiling=bool(enable_vae_tiling),
            max_sequence_length=max_sequence_length,
        )

        person = person_pil.convert("RGB").resize((width, height))
        garment = garment_pil.convert("RGB").resize((width, height))

        mask = masker(
            person,
            part=part,
            out_size_hw=None,
            square_cloth_mask=False,
            return_labels=False,
        ).convert("L")
        mask = backend.binarize_mask(mask, width, height)

        if part == "shoe":
            mask = backend.refine_shoe_mask(mask)
            mask = backend.binarize_mask(mask, width, height)

        cached = prompt_cache[part]

        result = backend.run_tryon(
            pipe=pipe,
            person_img=person,
            garment_img=garment,
            mask_img=mask,
            part=part,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance,
            seed=seed,
            prompt_embeds=cached["prompt_embeds"],
            pooled_prompt_embeds=cached["pooled_prompt_embeds"],
            max_sequence_length=max_sequence_length,
            upper_occupy_mode=upper_occupy_mode,
            upper_torso_ratio=float(upper_torso_ratio),
            upper_sleeve_ratio=float(upper_sleeve_ratio),
        )

        masked = backend.build_masked_person_preview(person, mask)
        overlay = _make_mask_overlay(person, mask)

        return result, mask, masked, overlay

    except gr.Error:
        raise
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise gr.Error(f"虚拟试衣失败：{type(e).__name__}: {e}")


def prompt_for_part(part):
    return backend.part_prompt(part)


CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #666; }
.result-image img { object-fit: contain !important; }
"""


def build_app():
    with gr.Blocks(title="FLUX Virtual Try-On", css=CSS) as demo:
        gr.Markdown("# FLUX Virtual Try-On", elem_id="title")
        gr.Markdown(
            "FLUX Fill + AutoMask + RGJA + Dual Region",
            elem_id="subtitle",
        )

        with gr.Row():
            with gr.Column():
                person_in = gr.Image(label="人物图片", type="pil", height=520)
            with gr.Column():
                garment_in = gr.Image(label="服装 / 鞋子图片", type="pil", height=520)

        part = gr.Dropdown(
            choices=[
                ("上装 Upper", "upper"),
                ("下装 Lower", "lower"),
                ("连衣裙 / 全身 Overall", "overall"),
                ("鞋子 Shoe", "shoe"),
            ],
            value="upper",
            label="服装类别",
        )

        with gr.Row():
            generate_mask_btn = gr.Button("① 生成 Mask", variant="secondary")
            tryon_btn = gr.Button("② 开始虚拟试衣", variant="primary")

        with gr.Row():
            mask_out = gr.Image(label="Auto Mask", type="pil")
            masked_out = gr.Image(label="Masked Person", type="pil")
            overlay_out = gr.Image(label="Mask Overlay", type="pil")

        result_out = gr.Image(
            label="Virtual Try-On Result",
            type="pil",
            height=650,
            elem_classes=["result-image"],
        )

        with gr.Accordion("⚙ 高级推理参数", open=False):
            with gr.Row():
                steps = gr.Slider(1, 60, value=30, step=1, label="Inference Steps")
                guidance = gr.Slider(1, 50, value=30.0, step=0.5, label="Guidance Scale")
                seed = gr.Number(value=42, precision=0, label="Seed")

            with gr.Row():
                width = gr.Number(value=576, precision=0, label="Width")
                height = gr.Number(value=768, precision=0, label="Height")
                max_sequence_length = gr.Slider(
                    64, 512, value=256, step=64, label="Max Sequence Length"
                )

            with gr.Row():
                upper_occupy_mode = gr.Dropdown(
                    choices=["conservative", "erase"],
                    value="conservative",
                    label="Upper Occupy Mode",
                )
                upper_torso_ratio = gr.Slider(
                    0.3, 1.0, value=0.58, step=0.01, label="Upper Torso Ratio"
                )
                upper_sleeve_ratio = gr.Slider(
                    0.1, 1.0, value=0.46, step=0.01, label="Upper Sleeve Ratio"
                )

            with gr.Row():
                enable_rgja = gr.Checkbox(value=True, label="Enable RGJA")
                enable_dual = gr.Checkbox(value=True, label="Enable Dual Region")
                enable_vae_tiling = gr.Checkbox(value=False, label="Enable VAE Tiling")

            dtype_str = gr.Dropdown(
                choices=["fp16", "bf16", "fp32"],
                value=DEFAULT_DTYPE,
                label="Model Dtype",
            )

        with gr.Accordion("📦 模型路径", open=False):
            transformer_dir = gr.Textbox(
                value=DEFAULT_TRANSFORMER_DIR,
                label="Transformer Checkpoint",
            )
            inpaint_model = gr.Textbox(
                value=DEFAULT_INPAINT_MODEL,
                label="FLUX Fill Base Model",
            )
            densepose_dir = gr.Textbox(
                value=DEFAULT_DENSEPOSE_DIR,
                label="DensePose Checkpoint",
            )
            schp_dir = gr.Textbox(
                value=DEFAULT_SCHP_DIR,
                label="SCHP Checkpoint",
            )

        with gr.Accordion("📖 Category Prompt", open=False):
            prompt_preview = gr.Textbox(
                value=prompt_for_part("upper"),
                lines=6,
                interactive=False,
                label="Prompt",
            )

        part.change(fn=prompt_for_part, inputs=part, outputs=prompt_preview)

        generate_mask_btn.click(
            fn=generate_mask_fn,
            inputs=[person_in, garment_in, part, width, height, densepose_dir, schp_dir],
            outputs=[mask_out, masked_out, overlay_out],
            api_name=False,
        )

        tryon_btn.click(
            fn=tryon_fn,
            inputs=[
                person_in,
                garment_in,
                part,
                steps,
                guidance,
                seed,
                height,
                width,
                transformer_dir,
                inpaint_model,
                densepose_dir,
                schp_dir,
                dtype_str,
                enable_rgja,
                enable_dual,
                enable_vae_tiling,
                max_sequence_length,
                upper_occupy_mode,
                upper_torso_ratio,
                upper_sleeve_ratio,
            ],
            outputs=[result_out, mask_out, masked_out, overlay_out],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    # The server-side health check uses requests. If a proxy is configured for
    # the shell, it may incorrectly intercept localhost and make Gradio believe
    # localhost is inaccessible. Keep loopback addresses out of the proxy.
    no_proxy = os.environ.get("NO_PROXY", "")
    required = ["localhost", "127.0.0.1", "::1"]
    entries = [x.strip() for x in no_proxy.split(",") if x.strip()]
    for host in required:
        if host not in entries:
            entries.append(host)
    os.environ["NO_PROXY"] = ",".join(entries)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "0") == "1",
        strict_cors=False,
        show_error=True,
    )
