import argparse
import gc
import json
import os
import types
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import check_min_version, load_image

from masker import AutoMaskerFromImage


CATEGORY_TO_ID = {"upper": 0, "lower": 1, "overall": 2, "shoe": 3}


def resolve_ckpt_dir(name: str, env_key: str, project_root: Path) -> Path:
    v = os.environ.get(env_key, "").strip()
    if v:
        p = Path(v).expanduser().resolve()
        if p.exists():
            return p
    p = (project_root / "ckpt" / name).resolve()
    if p.exists():
        return p
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        p = (Path(local) / "vton_ckpt" / name).resolve()
        if p.exists():
            return p
    p = Path.home() / ".cache" / "vton_ckpt" / name
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Cannot locate {name} ckpt dir.\n"
        f"Set env var {env_key}, or put it under {project_root / 'ckpt' / name}, "
        f"or put it under ~/.cache/vton_ckpt/{name}."
    )


def make_transforms():
    image_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    mask_t = transforms.Compose([transforms.ToTensor()])
    return image_t, mask_t


def refine_shoe_mask(mask_img, dilate_k: int = 5):
    m = np.array(mask_img.convert("L"), dtype=np.uint8)
    binary = (m > 127).astype(np.uint8)
    if binary.sum() == 0:
        return Image.fromarray(binary * 255, mode="L")
    k = max(1, int(dilate_k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    return Image.fromarray(binary * 255, mode="L")


def binarize_mask(mask: Image.Image, width: int, height: int) -> Image.Image:
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.NEAREST)
    return mask.point(lambda p: 255 if p >= 127 else 0)


def build_masked_person_preview(person_img: Image.Image, mask_img: Image.Image, fill_value: int = 128) -> Image.Image:
    person = person_img.convert("RGB")
    mask = mask_img.convert("L")
    if mask.size != person.size:
        mask = mask.resize(person.size, resample=Image.NEAREST)
    fill = Image.new("RGB", person.size, (fill_value, fill_value, fill_value))
    return Image.composite(fill, person, mask)


def concat_horiz(left: Image.Image, right: Image.Image) -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    out = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def save_overview_inputs(
    save_dir: str,
    person_img: Image.Image,
    garment_img: Image.Image,
    mask_img: Image.Image,
) -> None:
    out_dir = Path(save_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = mask_img.convert("L")
    masked_person = build_masked_person_preview(person_img, mask)
    zero_mask = Image.new("L", mask.size, 0)

    person_img.save(out_dir / "person_image.png")
    garment_img.save(out_dir / "reference_garment.png")
    mask.save(out_dir / "edit_mask.png")
    masked_person.save(out_dir / "masked_person.png")
    concat_horiz(garment_img, masked_person).save(out_dir / "concat_input_image.png")
    concat_horiz(zero_mask.convert("RGB"), mask.convert("RGB")).save(out_dir / "concat_input_mask.png")


def part_prompt(part: str) -> str:
    prefix = (
        "The pair of images highlights a fashion item (garment or shoes) and its styling on a model, "
        "high resolution, 4K, 8K; "
    )
    if part == "upper":
        return prefix + (
            "[IMAGE1] Detailed product shot of the upper garment, clear fabric texture, seams, neckline, sleeves, and silhouette. "
            "[IMAGE2] The same upper garment is worn by a model, natural drape on torso and arms."
        )
    if part == "lower":
        return prefix + (
            "[IMAGE1] Detailed product shot of the lower garment, clear waistband, cut, length, stitching, and fabric texture. "
            "[IMAGE2] The same lower garment is worn by a model, realistic fit on waist/hips."
        )
    if part == "overall":
        return prefix + (
            "[IMAGE1] Detailed product shot of the one-piece outfit, clear structure, seams, closures, design elements, and fabric texture. "
            "[IMAGE2] The same outfit is worn by a model, natural full-body fit, realistic drape."
        )
    if part == "shoe":
        return prefix + (
            "[IMAGE1] Detailed product shot of the shoes, clear materials, stitching, sole, toe box, and silhouette. "
            "[IMAGE2] The same shoes are worn by a model, natural alignment on feet, realistic scale."
        )
    return prefix + (
        "[IMAGE1] Detailed product shot of the fashion item, clear texture and shape. "
        "[IMAGE2] The same item is worn by a model, natural fit and realistic proportion."
    )


class RefTokenEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        if n_layers > 0:
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        else:
            self.encoder = nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.proj(x)))


class RGJACrossAttnAdapter(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, hd = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * hd)

    @torch.no_grad()
    def compute_kv_cache(self, ref_tokens: torch.Tensor):
        ref_tokens = self.norm_kv(ref_tokens)
        return self._to_heads(self.k_proj(ref_tokens)), self._to_heads(self.v_proj(ref_tokens))

    def forward(self, person_tokens: torch.Tensor, ref_tokens: torch.Tensor = None, kv_cache=None) -> torch.Tensor:
        q = self._to_heads(self.q_proj(self.norm_q(person_tokens)))
        if kv_cache is None:
            assert ref_tokens is not None
            ref_tokens = self.norm_kv(ref_tokens)
            k = self._to_heads(self.k_proj(ref_tokens))
            v = self._to_heads(self.v_proj(ref_tokens))
        else:
            k, v = kv_cache
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        return self.out_proj(self._from_heads(attn))


class DualRegionLayoutAdapter(nn.Module):
    def __init__(self, layout_in_dim: int, d_model: int, hidden_ratio: float = 1.0):
        super().__init__()
        hidden = max(d_model, int(d_model * hidden_ratio))
        self.norm = nn.LayerNorm(layout_in_dim)
        self.proj = nn.Sequential(
            nn.Linear(layout_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, packed_hidden_states: torch.Tensor, layout_cond: torch.Tensor, height: int, width: int) -> torch.Tensor:
        layout_cond = F.interpolate(layout_cond.float(), size=(height, width), mode="nearest")
        packed_layout = FluxFillPipeline._pack_latents(
            layout_cond,
            batch_size=layout_cond.shape[0],
            num_channels_latents=layout_cond.shape[1],
            height=height,
            width=width,
        )
        packed_layout = packed_layout.to(device=packed_hidden_states.device, dtype=packed_hidden_states.dtype)
        return packed_hidden_states + self.proj(self.norm(packed_layout))


def load_matching_state_dict(module: nn.Module, state: dict, label: str) -> bool:
    current = module.state_dict()
    matched = {k: v for k, v in state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    if len(matched) == 0:
        print(f"[{label}] no compatible tensors found, skipped")
        return False
    module.load_state_dict(matched, strict=False)
    skipped = len(state) - len(matched)
    if skipped:
        print(f"[{label}] loaded partially, skipped {skipped} incompatible tensors")
    return True


def get_half_split_indices_from_img_ids(img_ids: torch.Tensor, device: torch.device, cache: dict):
    img_ids_ = img_ids[0] if img_ids.dim() == 3 else img_ids
    x = img_ids_[:, -1]
    key = ("imgids_half", int(img_ids_.shape[0]), int(x.min().item()), int(x.max().item()), str(device))
    if key in cache:
        return cache[key]
    x_min = int(x.min().item())
    x_max = int(x.max().item())
    x_half = (x_min + x_max + 1) // 2
    cloth_mask = x < x_half
    person_mask = ~cloth_mask
    cloth_idx = torch.nonzero(cloth_mask, as_tuple=False).squeeze(1).to(device=device, dtype=torch.long)
    person_idx = torch.nonzero(person_mask, as_tuple=False).squeeze(1).to(device=device, dtype=torch.long)
    cache[key] = (cloth_idx, person_idx, x_half)
    return cache[key]


def parse_layer_indices(value, default: str) -> List[int]:
    if value is None or str(value).strip() == "":
        value = default
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]


def rgja_layer_key(kind: str, idx: int) -> str:
    return f"{kind}_{idx}"


def _replace_rgja_hook_tensor(output, tensor_index, new_tensor):
    if tensor_index is None:
        return new_tensor
    if isinstance(output, tuple):
        items = list(output)
        items[tensor_index] = new_tensor
        return tuple(items)
    if isinstance(output, list):
        items = list(output)
        items[tensor_index] = new_tensor
        return items
    return output


def _find_rgja_hook_tensor(output, num_image_tokens: int, d_model: int):
    candidates = list(output) if isinstance(output, (tuple, list)) else [output]
    best = None
    for i, item in enumerate(candidates):
        if not torch.is_tensor(item) or item.dim() != 3 or item.shape[-1] != d_model:
            continue
        if item.shape[1] == num_image_tokens:
            return (i if isinstance(output, (tuple, list)) else None), item, 0
        if item.shape[1] > num_image_tokens:
            best = (i if isinstance(output, (tuple, list)) else None), item, item.shape[1] - num_image_tokens
    if best is not None:
        return best
    return None, None, 0


def _apply_rgja_to_hidden(transformer, hidden_states: torch.Tensor, offset: int, layer_key: str) -> torch.Tensor:
    ctx = getattr(transformer, "_rgja_ctx", None)
    if ctx is None or getattr(transformer, "rgja_adapter", None) is None:
        return hidden_states
    person_idx = ctx["person_idx"].to(device=hidden_states.device)
    if offset:
        person_idx = person_idx + int(offset)
    person_tokens = hidden_states.index_select(1, person_idx)
    ref_tokens = ctx["ref_tokens"].to(device=hidden_states.device, dtype=hidden_states.dtype)
    delta = transformer.rgja_adapter(person_tokens, ref_tokens=ref_tokens)
    gate_person = ctx["gate_person"].to(device=hidden_states.device, dtype=hidden_states.dtype)

    scales = getattr(transformer, "rgja_layer_scales", None)
    scale_param = scales[layer_key] if scales is not None and layer_key in scales else None
    layer_scale = torch.tanh(scale_param.to(device=hidden_states.device, dtype=hidden_states.dtype)) if scale_param is not None else 1.0
    total_scale = hidden_states.new_tensor(float(getattr(transformer, "rgja_scale", 1.0))) * layer_scale
    updated = person_tokens + gate_person * total_scale * delta.to(dtype=hidden_states.dtype)
    hidden_states = hidden_states.clone()
    hidden_states.index_copy_(1, person_idx, updated)
    return hidden_states


def make_rgja_forward_hook(transformer, layer_key: str):
    def _hook(_module, _inputs, output):
        ctx = getattr(transformer, "_rgja_ctx", None)
        if ctx is None or ctx.get("mode") != "multi_layer_hooks":
            return output
        tensor_index, hidden_states, offset = _find_rgja_hook_tensor(
            output,
            num_image_tokens=int(ctx["num_image_tokens"]),
            d_model=int(ctx["d_model"]),
        )
        if hidden_states is None:
            return output
        hidden_states = _apply_rgja_to_hidden(transformer, hidden_states, offset=offset, layer_key=layer_key)
        return _replace_rgja_hook_tensor(output, tensor_index, hidden_states)
    return _hook


def register_multilayer_rgja_hooks(transformer, transformer_layers: Sequence[int], single_layers: Sequence[int]) -> None:
    for handle in getattr(transformer, "_rgja_hook_handles", []):
        try:
            handle.remove()
        except Exception:
            pass
    handles = []
    for idx in transformer_layers:
        if idx < len(transformer.transformer_blocks):
            handles.append(transformer.transformer_blocks[idx].register_forward_hook(make_rgja_forward_hook(transformer, rgja_layer_key("tb", idx))))
        else:
            print(f"[RGJA] transformer_blocks.{idx} out of range, skipped")
    for idx in single_layers:
        if idx < len(transformer.single_transformer_blocks):
            handles.append(transformer.single_transformer_blocks[idx].register_forward_hook(make_rgja_forward_hook(transformer, rgja_layer_key("sb", idx))))
        else:
            print(f"[RGJA] single_transformer_blocks.{idx} out of range, skipped")
    transformer._rgja_hook_handles = handles
    print(f"[RGJA] multi-layer hooks registered: transformer_blocks={list(transformer_layers)}, single_transformer_blocks={list(single_layers)}")


def maybe_load_dual_region(transformer: FluxTransformer2DModel, root_dir: Path, device: torch.device, dtype: torch.dtype):
    cfg_path = root_dir / "dual_region_config.json"
    w_path = root_dir / "dual_region_adapter" / "pytorch_model.bin"
    if not (cfg_path.exists() and w_path.exists()):
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = DualRegionLayoutAdapter(
        layout_in_dim=int(cfg["layout_in_dim"]),
        d_model=int(cfg["d_model"]),
    ).to(device=device, dtype=dtype)
    adapter.load_state_dict(torch.load(str(w_path), map_location="cpu"), strict=True)
    adapter.eval()
    transformer.dual_region_adapter = adapter
    return True


def maybe_load_rgja(transformer: FluxTransformer2DModel, root_dir: Path, device: torch.device, dtype: torch.dtype):
    cfg_path = root_dir / "rgja_config.json"
    ref_w = root_dir / "ref_encoder" / "pytorch_model.bin"
    adp_w = root_dir / "rgja_adapter" / "pytorch_model.bin"
    if not (cfg_path.exists() and ref_w.exists() and adp_w.exists()):
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ref_encoder = RefTokenEncoder(
        in_dim=int(cfg["in_dim"]),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg.get("ref_encoder_layers", 2)),
        n_heads=int(cfg.get("rgja_heads", 8)),
        dropout=float(cfg.get("rgja_dropout", 0.0)),
    ).to(device=device, dtype=dtype)
    rgja_adapter = RGJACrossAttnAdapter(
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg.get("rgja_heads", 8)),
        dropout=float(cfg.get("rgja_dropout", 0.0)),
    ).to(device=device, dtype=dtype)

    load_matching_state_dict(ref_encoder, torch.load(str(ref_w), map_location="cpu"), "RGJA ref_encoder")
    load_matching_state_dict(rgja_adapter, torch.load(str(adp_w), map_location="cpu"), "RGJA adapter")
    ref_encoder.eval()
    rgja_adapter.eval()

    transformer.ref_encoder = ref_encoder
    transformer.rgja_adapter = rgja_adapter
    transformer.rgja_scale = float(cfg.get("rgja_scale", 1.0))
    transformer.rgja_injection = str(cfg.get("rgja_injection", "input_level"))
    transformer.rgja_d_model = int(cfg["d_model"])
    transformer._rgja_token_index_cache = {}

    if transformer.rgja_injection == "multi_layer_hooks":
        transformer_layers = parse_layer_indices(cfg.get("rgja_transformer_layers", "3,7,11,15"), "3,7,11,15")
        single_layers = parse_layer_indices(cfg.get("rgja_single_layers", "8,16,24,32"), "8,16,24,32")
        scales = nn.ParameterDict()
        for idx in transformer_layers:
            scales[rgja_layer_key("tb", idx)] = nn.Parameter(torch.zeros((), device=device, dtype=dtype), requires_grad=False)
        for idx in single_layers:
            scales[rgja_layer_key("sb", idx)] = nn.Parameter(torch.zeros((), device=device, dtype=dtype), requires_grad=False)
        scales_w = root_dir / "rgja_layer_scales" / "pytorch_model.bin"
        if scales_w.exists():
            scales.load_state_dict(torch.load(str(scales_w), map_location="cpu"), strict=False)
        scales.eval()
        transformer.rgja_layer_scales = scales
        register_multilayer_rgja_hooks(transformer, transformer_layers, single_layers)
    return True


def make_upper_conservative_occupy(
    mask_tensor: torch.Tensor,
    *,
    torso_ratio: float = 0.58,
    sleeve_ratio: float = 0.46,
) -> torch.Tensor:
    occupy = torch.zeros_like(mask_tensor)
    for i in range(mask_tensor.shape[0]):
        m = mask_tensor[i, 0] > 0.5
        coords = torch.nonzero(m, as_tuple=False)
        if coords.numel() == 0:
            occupy[i] = mask_tensor[i]
            continue
        y1 = int(coords[:, 0].min().item())
        y2 = int(coords[:, 0].max().item()) + 1
        x1 = int(coords[:, 1].min().item())
        x2 = int(coords[:, 1].max().item()) + 1
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        torso_pad = int(round(w * (1.0 - float(torso_ratio)) * 0.5))
        cx1 = min(x2, max(x1, x1 + torso_pad))
        cx2 = max(cx1, min(x2, x2 - torso_pad))
        sleeve_y2 = min(y2, y1 + int(round(h * float(sleeve_ratio))))
        keep = torch.zeros_like(m)
        keep[y1:y2, cx1:cx2] = True
        keep[y1:sleeve_y2, x1:x2] = True
        occupy[i, 0] = (m & keep).to(dtype=mask_tensor.dtype)
    return occupy


def make_layout_cond(
    mask_tensor: torch.Tensor,
    part: str,
    width: int,
    upper_occupy_mode: str = "conservative",
    upper_torso_ratio: float = 0.58,
    upper_sleeve_ratio: float = 0.46,
) -> torch.Tensor:
    b, _, h, w2 = mask_tensor.shape
    cond = torch.zeros((b, 6, h, w2), device=mask_tensor.device, dtype=mask_tensor.dtype)
    right = slice(width, width * 2)
    occupy = mask_tensor
    if part == "upper" and upper_occupy_mode == "conservative":
        occupy = make_upper_conservative_occupy(
            mask_tensor,
            torso_ratio=upper_torso_ratio,
            sleeve_ratio=upper_sleeve_ratio,
        )
    cond[:, 0:1, :, right] = mask_tensor[:, :, :, right]
    cond[:, 1:2, :, right] = occupy[:, :, :, right]
    cond[:, 2 + CATEGORY_TO_ID.get(part, 0), :, right] = 1.0
    return cond


def patch_forward_for_extra_modules(transformer: FluxTransformer2DModel):
    if getattr(transformer, "_extra_modules_patched", False):
        return
    old_forward = transformer.forward

    def new_forward(self, hidden_states, timestep, guidance, pooled_projections, encoder_hidden_states, txt_ids, img_ids, return_dict=False, **kwargs):
        ctx = getattr(self, "_tryon_ctx", None)
        if ctx is not None and hasattr(self, "dual_region_adapter") and ctx.get("layout_cond") is not None:
            hidden_states = self.dual_region_adapter(
                hidden_states,
                layout_cond=ctx["layout_cond"].to(device=hidden_states.device, dtype=hidden_states.dtype),
                height=int(ctx["latent_h"]),
                width=int(ctx["latent_w"]),
            )

        rgja_ctx = getattr(self, "_rgja_ctx", None)
        if rgja_ctx is not None and hasattr(self, "rgja_adapter") and hasattr(self, "ref_encoder"):
            cache = getattr(self, "_rgja_token_index_cache", {})
            _, person_idx, _ = get_half_split_indices_from_img_ids(img_ids, hidden_states.device, cache)
            self._rgja_token_index_cache = cache

            packed_gate = rgja_ctx["packed_gate"].to(device=hidden_states.device, dtype=hidden_states.dtype)
            gate_person = packed_gate.index_select(1, person_idx)
            if gate_person.shape[-1] != 1:
                gate_person = gate_person.amax(dim=-1, keepdim=True)
            rgja_ctx["person_idx"] = person_idx
            rgja_ctx["gate_person"] = gate_person
            rgja_ctx["num_image_tokens"] = int(hidden_states.shape[1])
            rgja_ctx["d_model"] = int(getattr(self, "rgja_d_model", hidden_states.shape[-1]))

            if getattr(self, "rgja_injection", "input_level") != "multi_layer_hooks":
                kv_cache = rgja_ctx.get("kv_cache")
                if kv_cache is not None:
                    k, v = kv_cache
                    k = k.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    v = v.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    if k.shape[0] == 1 and hidden_states.shape[0] > 1:
                        k = k.repeat(hidden_states.shape[0], 1, 1, 1)
                        v = v.repeat(hidden_states.shape[0], 1, 1, 1)
                    kv_cache = (k, v)
                person_tokens = hidden_states.index_select(1, person_idx)
                delta = self.rgja_adapter(person_tokens, kv_cache=kv_cache)
                scale_t = hidden_states.new_tensor(float(getattr(self, "rgja_scale", 1.0)))
                person_tokens = person_tokens + gate_person * (scale_t * delta)
                hidden_states = hidden_states.clone()
                hidden_states.index_copy_(1, person_idx, person_tokens)

        return old_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=return_dict,
            **kwargs,
        )

    transformer.forward = types.MethodType(new_forward, transformer)
    transformer._extra_modules_patched = True


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length=512, prompt=None, num_images_per_prompt=1, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, prompt_embeds.shape[1], -1)


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None, num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    pooled = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
    pooled = pooled.to(dtype=text_encoder.dtype, device=device)
    return pooled.repeat(1, num_images_per_prompt, 1).view(len(prompt) * num_images_per_prompt, -1)


@torch.no_grad()
def encode_prompt_manual(tokenizers, text_encoders, prompt: str, max_sequence_length: int, device: torch.device, num_images_per_prompt: int = 1):
    tokenizer_one, tokenizer_two = tokenizers
    text_encoder_one, text_encoder_two = text_encoders
    pooled_prompt_embeds = _encode_prompt_with_clip(text_encoder_one, tokenizer_one, prompt, device=device, num_images_per_prompt=num_images_per_prompt)
    prompt_embeds = _encode_prompt_with_t5(text_encoder_two, tokenizer_two, max_sequence_length=max_sequence_length, prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=text_encoder_one.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


@torch.no_grad()
def build_prompt_cache(pipe: FluxFillPipeline, device: torch.device, dtype: torch.dtype, max_sequence_length: int, custom_prompt: str = ""):
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    if pipe.text_encoder is None or pipe.text_encoder_2 is None:
        raise RuntimeError("Pipeline text encoders are missing before prompt cache build.")
    prompt_map = {"__custom__": custom_prompt.strip()} if custom_prompt.strip() else {
        "upper": part_prompt("upper"),
        "lower": part_prompt("lower"),
        "overall": part_prompt("overall"),
        "shoe": part_prompt("shoe"),
    }
    cache = {}
    for key, text in prompt_map.items():
        pe, pp, ti = encode_prompt_manual(tokenizers, text_encoders, text, max_sequence_length, device=device)
        cache[key] = {
            "prompt_embeds": pe.detach().cpu().to(dtype=dtype),
            "pooled_prompt_embeds": pp.detach().cpu().to(dtype=dtype),
            "text_ids": ti.detach().cpu().to(dtype=dtype),
            "prompt_text": text,
        }
    return cache


def drop_text_encoders(pipe: FluxFillPipeline):
    for name in ["text_encoder", "text_encoder_2"]:
        try:
            if getattr(pipe, name, None) is not None:
                delattr(pipe, name)
                setattr(pipe, name, None)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_enable_vae_tiling(pipe: FluxFillPipeline):
    ok = False
    try:
        pipe.enable_vae_tiling()
        ok = True
    except Exception:
        pass
    if not ok:
        try:
            pipe.vae.enable_tiling()
            ok = True
        except Exception:
            pass
    print("[VAE] tiling enabled." if ok else "[VAE] tiling not available, skipped.")


def get_transformer_dtype(transformer: nn.Module):
    return next(transformer.parameters()).dtype


@torch.inference_mode()
def run_tryon(
    pipe: FluxFillPipeline,
    person_img,
    garment_img,
    mask_img,
    part: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    max_sequence_length: int,
    upper_occupy_mode: str = "conservative",
    upper_torso_ratio: float = 0.58,
    upper_sleeve_ratio: float = 0.46,
):
    device = pipe.device
    image_t, mask_t = make_transforms()
    person_tensor = image_t(person_img)
    garment_tensor = image_t(garment_img)
    mask_tensor = (mask_t(mask_img)[:1] >= 0.5).float()

    masked_person_tensor = person_tensor * (1.0 - mask_tensor)
    inpaint_image = torch.cat([garment_tensor, masked_person_tensor], dim=2)
    extended_mask = torch.cat([torch.zeros_like(mask_tensor), mask_tensor], dim=2)
    inpaint_image = inpaint_image.unsqueeze(0).to(device)
    extended_mask = extended_mask.unsqueeze(0).to(device)

    transformer = pipe.transformer
    transformer_dtype = get_transformer_dtype(transformer)
    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype, non_blocking=True)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=transformer_dtype, non_blocking=True)

    needs_extra_context = hasattr(transformer, "dual_region_adapter") or (hasattr(transformer, "ref_encoder") and hasattr(transformer, "rgja_adapter"))
    if needs_extra_context:
        vae = pipe.vae
        vae_dtype = next(vae.parameters()).dtype
        latents = vae.encode(inpaint_image.to(dtype=vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
        h_l, w_l2 = latents.shape[-2], latents.shape[-1]
        transformer._tryon_ctx = {
            "layout_cond": make_layout_cond(
                extended_mask,
                part,
                width,
                upper_occupy_mode=upper_occupy_mode,
                upper_torso_ratio=upper_torso_ratio,
                upper_sleeve_ratio=upper_sleeve_ratio,
            ),
            "latent_h": h_l,
            "latent_w": w_l2,
        }

        if hasattr(transformer, "ref_encoder") and hasattr(transformer, "rgja_adapter"):
            w_half = w_l2 // 2
            cloth_lat = latents[:, :, :, :w_half]
            cloth_packed = FluxFillPipeline._pack_latents(
                cloth_lat,
                batch_size=1,
                num_channels_latents=cloth_lat.shape[1],
                height=h_l,
                width=w_half,
            )
            ref_dtype = next(transformer.ref_encoder.parameters()).dtype
            ref_tokens = transformer.ref_encoder(cloth_packed.to(dtype=ref_dtype))

            gate_lat = F.interpolate(extended_mask.float(), size=(h_l, w_l2), mode="nearest")
            packed_gate = FluxFillPipeline._pack_latents(
                gate_lat,
                batch_size=1,
                num_channels_latents=1,
                height=h_l,
                width=w_l2,
            )
            if packed_gate.shape[-1] != 1:
                packed_gate = packed_gate.amax(dim=-1, keepdim=True)

            rgja_ctx = {
                "mode": getattr(transformer, "rgja_injection", "input_level"),
                "ref_tokens": ref_tokens.detach(),
                "packed_gate": packed_gate.detach(),
            }
            if rgja_ctx["mode"] != "multi_layer_hooks":
                kv_cache = transformer.rgja_adapter.compute_kv_cache(ref_tokens)
                rgja_ctx["kv_cache"] = (kv_cache[0].detach(), kv_cache[1].detach())
            transformer._rgja_ctx = rgja_ctx
        patch_forward_for_extra_modules(transformer)

    generator = torch.Generator(device=str(device)).manual_seed(int(seed))
    out = pipe(
        prompt=None,
        prompt_2=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width * 2,
        image=inpaint_image,
        mask_image=extended_mask,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        max_sequence_length=max_sequence_length,
    ).images[0]

    transformer._tryon_ctx = None
    transformer._rgja_ctx = None
    return out.crop((width, 0, width * 2, height))


def resolve_transformer_dir_and_root(p: str):
    p = Path(p).expanduser().resolve()
    if (p / "config.json").exists() or (p / "model_index.json").exists():
        return str(p), p
    if (p / "transformer" / "config.json").exists() or (p / "transformer" / "model_index.json").exists():
        return str((p / "transformer").resolve()), p
    raise FileNotFoundError(
        f"Cannot find transformer weights under: {p}\n"
        "Expect either <final_transformer>/config.json or <checkpoint_dir>/transformer/config.json."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=str, required=True)
    parser.add_argument("--garment", type=str, required=True)
    parser.add_argument("--part", type=str, default="upper", choices=["upper", "lower", "overall", "shoe"])
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--densepose_ckpt_dir", type=str, default=None)
    parser.add_argument("--schp_ckpt_dir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--pretrained_inpaint_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--base_model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out", type=str, default="tryon.png")
    parser.add_argument("--save_mask", type=str, default=None)
    parser.add_argument(
        "--save_inputs_dir",
        type=str,
        default=None,
        help="Optional directory to save person, reference garment, edit mask, masked person, and concatenated FLUX inputs.",
    )
    parser.add_argument("--no_dual_region", action="store_true")
    parser.add_argument("--no_rgja", action="store_true")
    parser.add_argument("--upper_occupy_mode", type=str, default="conservative", choices=["conservative", "erase"])
    parser.add_argument("--upper_torso_ratio", type=float, default=0.58)
    parser.add_argument("--upper_sleeve_ratio", type=float, default=0.46)
    parser.add_argument("--enable_vae_tiling", action="store_true")
    args = parser.parse_args()
    check_min_version("0.30.2")

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if args.max_sequence_length > 512:
        raise ValueError("--max_sequence_length should be <= 512 for FLUX.")

    transformer_dir, root_dir = resolve_transformer_dir_and_root(args.transformer_path)
    transformer = FluxTransformer2DModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
    pipe = FluxFillPipeline.from_pretrained(
        args.pretrained_inpaint_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    if args.enable_vae_tiling:
        maybe_enable_vae_tiling(pipe)
    if (not args.no_dual_region) and device != "cpu":
        ok_dual = maybe_load_dual_region(pipe.transformer, root_dir, device=torch.device(device), dtype=torch_dtype)
        print(f"[DualRegion] {'loaded from ' + str(root_dir) if ok_dual else 'not found, run without dual-region adapter'}")
    if (not args.no_rgja) and device != "cpu":
        ok_rgja = maybe_load_rgja(pipe.transformer, root_dir, device=torch.device(device), dtype=torch_dtype)
        print(f"[RGJA] {'loaded from ' + str(root_dir) + ' (' + getattr(pipe.transformer, 'rgja_injection', 'input_level') + ')' if ok_rgja else 'not found, run without RGJA'}")

    prompt_cache = build_prompt_cache(
        pipe=pipe,
        device=torch.device(device),
        dtype=torch_dtype,
        max_sequence_length=args.max_sequence_length,
        custom_prompt=args.prompt,
    )
    prompt_key = "__custom__" if args.prompt.strip() else args.part
    print("[Prompt] cached custom prompt once." if args.prompt.strip() else "[Prompt] cached fixed prompts: upper / lower / overall / shoe")
    drop_text_encoders(pipe)
    print("[Prompt] text encoders deleted after caching.")

    project_root = Path(__file__).resolve().parent
    densepose_dir = Path(args.densepose_ckpt_dir).expanduser().resolve() if args.densepose_ckpt_dir else resolve_ckpt_dir("densepose", "DENSEPOSE_CKPT_DIR", project_root)
    schp_dir = Path(args.schp_ckpt_dir).expanduser().resolve() if args.schp_ckpt_dir else resolve_ckpt_dir("schp", "SCHP_CKPT_DIR", project_root)

    person = load_image(args.person).convert("RGB").resize((args.width, args.height))
    garment = load_image(args.garment).convert("RGB").resize((args.width, args.height))
    auto_masker = AutoMaskerFromImage(
        densepose_ckpt_dir=str(densepose_dir),
        schp_ckpt_dir=str(schp_dir),
        device=device,
    )
    mask = auto_masker(
        person,
        part=args.part,
        out_size_hw=None,
        square_cloth_mask=False,
        return_labels=False,
    ).convert("L")
    mask = binarize_mask(mask, args.width, args.height)
    if args.part == "shoe":
        mask = refine_shoe_mask(
            mask,
            dilate_k=int(os.getenv("SHOE_MASK_DILATE_K", "7")),
        )
        mask = binarize_mask(mask, args.width, args.height)

    if args.save_mask:
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.save_mask)

    if args.save_inputs_dir:
        save_overview_inputs(
            save_dir=args.save_inputs_dir,
            person_img=person,
            garment_img=garment,
            mask_img=mask,
        )
        print(f"[OK] saved overview inputs: {Path(args.save_inputs_dir).expanduser().resolve()}")

    cached = prompt_cache[prompt_key]
    tryon = run_tryon(
        pipe=pipe,
        person_img=person,
        garment_img=garment,
        mask_img=mask,
        part=args.part,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        prompt_embeds=cached["prompt_embeds"],
        pooled_prompt_embeds=cached["pooled_prompt_embeds"],
        max_sequence_length=args.max_sequence_length,
        upper_occupy_mode=args.upper_occupy_mode,
        upper_torso_ratio=args.upper_torso_ratio,
        upper_sleeve_ratio=args.upper_sleeve_ratio,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tryon.save(args.out)
    print(f"[OK] saved try-on: {args.out}")
    if args.save_mask:
        print(f"[OK] saved mask:  {args.save_mask}")


if __name__ == "__main__":
    main()
