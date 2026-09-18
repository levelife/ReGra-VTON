import argparse
import gc
import json
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from diffusers import FluxFillPipeline, FluxTransformer2DModel
from diffusers.utils import check_min_version

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
        f"Cannot locate {name} ckpt dir. Set env var {env_key}, or put it under "
        f"{project_root / 'ckpt' / name}, or ~/.cache/vton_ckpt/{name}."
    )


def make_transforms():
    image_t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    mask_t = transforms.Compose([transforms.ToTensor()])
    return image_t, mask_t


def center_crop_max_area_by_aspect_ratio(img: Image.Image, target_ratio: float) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    src_ratio = w / h
    if abs(src_ratio - target_ratio) < 1e-8:
        return img
    if src_ratio > target_ratio:
        new_w, new_h = max(1, int(round(h * target_ratio))), h
    else:
        new_w, new_h = w, max(1, int(round(w / target_ratio)))
    left = max(0, (w - new_w) // 2)
    upper = max(0, (h - new_h) // 2)
    return img.crop((left, upper, left + new_w, upper + new_h))


def preprocess_pil_image(img: Image.Image, width: int, height: int, interpolation: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = center_crop_max_area_by_aspect_ratio(img, width / height)
    return img.resize((width, height), resample=interpolation)


def load_rgb(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = preprocess_pil_image(img, width, height, interpolation=Image.LANCZOS)
    return img


def load_binary_mask(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("L")
        img = preprocess_pil_image(img, width, height, interpolation=Image.NEAREST)
    return img.point(lambda p: 255 if p >= 127 else 0)


def binarize_mask(mask: Image.Image, width: int, height: int) -> Image.Image:
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.NEAREST)
    return mask.point(lambda p: 255 if p >= 127 else 0)


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
            "[IMAGE1] Detailed product shot of the lower garment (pants/skirt/shorts), clear waistband, cut, length, stitching. "
            "[IMAGE2] The same lower garment is worn by a model, realistic fit on waist/hips."
        )
    if part == "overall":
        return prefix + (
            "[IMAGE1] Detailed product shot of the one-piece outfit (dress/jumpsuit/set), clear structure, seams, closures, design elements. "
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
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out_proj(self._from_heads(attn))



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



def maybe_load_rgja(transformer: FluxTransformer2DModel, root_dir: Path, device: torch.device, dtype: torch.dtype) -> bool:
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



def patch_forward_for_extra_modules(transformer: FluxTransformer2DModel):
    """Patch transformer.forward only for RGJA inference. Dual-region logic has been removed."""
    if getattr(transformer, "_extra_modules_patched", False):
        return

    old_forward = transformer.forward

    def new_forward(self, hidden_states, timestep, guidance, pooled_projections, encoder_hidden_states, txt_ids, img_ids, return_dict=False, **kwargs):
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
    if custom_prompt.strip():
        prompt_map = {"__custom__": custom_prompt.strip()}
    else:
        prompt_map = {p: part_prompt(p) for p in ["upper", "lower", "overall", "shoe"]}
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
            delattr(pipe, name)
            setattr(pipe, name, None)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_transformer_dtype(transformer: nn.Module):
    return next(transformer.parameters()).dtype


@torch.inference_mode()
def run_tryon(
    pipe: FluxFillPipeline,
    person_img: Image.Image,
    garment_img: Image.Image,
    mask_img: Image.Image,
    part: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    max_sequence_length: int,
):
    device = pipe.device
    image_t, mask_t = make_transforms()

    person_tensor = image_t(person_img)
    garment_tensor = image_t(garment_img)
    mask_tensor = (mask_t(mask_img)[:1] >= 0.5).float()
    masked_person_tensor = person_tensor * (1.0 - mask_tensor)

    inpaint_image = torch.cat([garment_tensor, masked_person_tensor], dim=2).unsqueeze(0).to(device)
    extended_mask = torch.cat([torch.zeros_like(mask_tensor), mask_tensor], dim=2).unsqueeze(0).to(device)

    transformer = pipe.transformer
    transformer_dtype = get_transformer_dtype(transformer)
    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype, non_blocking=True)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=transformer_dtype, non_blocking=True)

    # RGJA context only. Dual-region layout conditioning has been removed.
    if hasattr(transformer, "ref_encoder") and hasattr(transformer, "rgja_adapter"):
        vae = pipe.vae
        vae_dtype = next(vae.parameters()).dtype
        latents = vae.encode(inpaint_image.to(dtype=vae_dtype)).latent_dist.sample() * vae.config.scaling_factor
        h_l, w_l2 = latents.shape[-2], latents.shape[-1]

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

    tryon = out.crop((width, 0, width * 2, height))
    transformer._rgja_ctx = None
    return tryon


@dataclass
class TryonSample:
    dataset: str
    output_path: Path
    person_path: Path
    garment_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    part: Optional[str] = None
    stages: Optional[List[Tuple[str, Path]]] = None


def resolve_existing_path(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


PAIR_LINE_RE = re.compile(r"\s+")


def parse_pair_lines(pair_file: Path) -> List[List[str]]:
    rows = []
    with pair_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            cols = [c for c in PAIR_LINE_RE.split(line) if c]
            if len(cols) >= 2:
                rows.append(cols)
    return rows


def find_pair_file(base_dir: Path, names: Sequence[str]) -> Path:
    for name in names:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No pair file found in {base_dir}. Tried: {list(names)}")


def save_processed_gt(gt_img: Image.Image, sample: TryonSample, gt_output_dir: Optional[Path]):
    if gt_output_dir is None:
        return
    gt_path = gt_output_dir / sample.output_path.name
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    gt_img.save(gt_path)


def save_used_mask(mask_img: Image.Image, sample: TryonSample, args, suffix: str = ""):
    if not getattr(args, "save_mask", False):
        return
    if getattr(args, "mask_output_dir", ""):
        mask_root = Path(args.mask_output_dir).expanduser().resolve()
    else:
        mask_root = sample.output_path.parent.parent / "mask"
    name = sample.output_path.name
    if suffix:
        stem = Path(name).stem
        ext = Path(name).suffix or ".png"
        name = f"{stem}__{suffix}{ext}"
    mask_path = mask_root / name
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    binarize_mask(mask_img, args.width, args.height).save(mask_path)


class VITONHDBatchDataset:
    def __init__(self, data_dir: str, output_dir: str, paired: bool):
        self.root = Path(data_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.samples = self._build(paired)

    def _build(self, paired: bool) -> List[TryonSample]:
        pair_file = find_pair_file(
            self.root,
            [
                "test_pairs.txt" if paired else "test_unpairs.txt",
                "test_pairs" if paired else "test_unpairs",
                "test_pairs_paired.txt" if paired else "test_pairs_unpaired.txt",
                "test_pairs_paired" if paired else "test_pairs_unpaired",
                "test_pairs.txt",
                "test_pairs",
            ],
        )
        samples = []
        for cols in parse_pair_lines(pair_file):
            person_name, cloth_name = cols[0], cols[1]
            person_path = self.root / "test" / "image" / person_name
            garment_path = self.root / "test" / "cloth" / cloth_name
            mask_path = resolve_existing_path(
                self.root / "test" / "agnostic-mask" / Path(person_name).with_suffix(".png"),
                self.root / "test" / "agnostic-mask" / Path(person_name).name,
                self.root / "test" / "agnostic-mask-catvton" / Path(person_name).with_suffix(".png"),
                self.root / "test" / "agnostic-v3.2" / Path(person_name).with_suffix(".png"),
            )
            output_path = self.output_dir / person_name
            if not output_path.exists():
                samples.append(TryonSample("viton-hd", output_path, person_path, garment_path, mask_path, "upper"))
        return samples


class DressCodeBatchDataset:
    CAT_TO_PART = {"upper_body": "upper", "lower_body": "lower", "dresses": "overall"}
    PART_TO_CAT = {"upper": "upper_body", "lower": "lower_body", "overall": "dresses"}

    def __init__(self, data_dir: str, output_dir: str, paired: bool, dresscode_part: Optional[str] = None):
        self.root = Path(data_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.dresscode_part = dresscode_part
        self.samples = self._build(paired)

    def _build(self, paired: bool) -> List[TryonSample]:
        samples = []
        items = list(self.CAT_TO_PART.items())
        if self.dresscode_part is not None:
            items = [(self.PART_TO_CAT[self.dresscode_part], self.dresscode_part)]
        for cat, part in items:
            cat_dir = self.root / cat
            if not cat_dir.exists():
                continue
            pair_file = find_pair_file(
                cat_dir,
                [
                    "test_pairs_paired.txt" if paired else "test_pairs_unpaired.txt",
                    "test_pairs_paired" if paired else "test_pairs_unpaired",
                    "test_pairs.txt" if paired else "test_unpairs.txt",
                    "test_pairs" if paired else "test_unpairs",
                ],
            )
            for cols in parse_pair_lines(pair_file):
                person_name, cloth_name = cols[0], cols[1]
                person_path = cat_dir / "images" / person_name
                garment_path = cat_dir / "images" / cloth_name
                mask_path = resolve_existing_path(
                    cat_dir / "agnostic_masks" / Path(person_name).with_suffix(".png"),
                    cat_dir / "agnostic_masks" / person_name,
                )
                output_path = self.output_dir / cat / person_name
                if not output_path.exists():
                    samples.append(TryonSample("dresscode", output_path, person_path, garment_path, mask_path, part))
        return samples


class DressCodeMRBatchDataset:
    PRIORITY = ["overall", "upper", "lower", "shoe", "bag"]

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        paired: bool,
        mr_mode: str = "sequential",
        mr_part: Optional[str] = None,
        data_list: str = "",
    ):
        self.root = Path(data_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.mr_mode = mr_mode
        self.mr_part = mr_part
        self.data_list = data_list
        self.samples = self._build(paired)

    def _append_single_part_samples(self, samples: List[TryonSample], person_rel: str, person_path: Path, refs: Dict[str, Path]):
        for key in ["upper", "lower", "overall", "shoe"]:
            if key not in refs:
                continue
            output_name = Path(person_rel).name
            output_path = self.output_dir / key / output_name
            if not output_path.exists():
                samples.append(TryonSample("dresscode-mr-single", output_path, person_path, refs[key], None, key, None))

    def _build(self, paired: bool) -> List[TryonSample]:
        if self.data_list:
            jsonl_file = self.root / self.data_list
            if not jsonl_file.exists():
                raise FileNotFoundError(f"Cannot find DressCode-MR data_list: {jsonl_file}")
        else:
            jsonl_file = resolve_existing_path(
                self.root / ("test.jsonl" if paired else "test_unpair.jsonl"),
                self.root / ("test_paired.jsonl" if paired else "test_unpaired.jsonl"),
            )
        if jsonl_file is None:
            raise FileNotFoundError(f"Cannot find DressCode-MR jsonl under {self.root}.")
        samples = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                record = json.loads(line)
                person_rel = record["person"]
                person_path = self.root / person_rel
                if not person_path.exists():
                    raise FileNotFoundError(f"Missing person image: {person_path}")
                refs = {}
                for key in self.PRIORITY:
                    value = record.get(key)
                    if value:
                        ref_path = self.root / value
                        if ref_path.exists():
                            refs[key] = ref_path
                if not refs:
                    continue
                if self.mr_part is not None:
                    if self.mr_part in refs and self.mr_part != "bag":
                        output_path = self.output_dir / self.mr_part / Path(person_rel).name
                        if not output_path.exists():
                            samples.append(TryonSample("dresscode-mr-single", output_path, person_path, refs[self.mr_part], None, self.mr_part, None))
                    continue
                if self.mr_mode == "expand":
                    self._append_single_part_samples(samples, person_rel, person_path, refs)
                    continue
                stages = []
                if self.mr_mode == "first":
                    for key in ["overall", "upper", "lower", "shoe"]:
                        if key in refs:
                            stages.append((key, refs[key]))
                            break
                else:
                    if "overall" in refs:
                        stages.append(("overall", refs["overall"]))
                    else:
                        if "upper" in refs:
                            stages.append(("upper", refs["upper"]))
                        if "lower" in refs:
                            stages.append(("lower", refs["lower"]))
                    if "shoe" in refs:
                        stages.append(("shoe", refs["shoe"]))
                if stages:
                    output_path = self.output_dir / Path(person_rel).name
                    if not output_path.exists():
                        samples.append(TryonSample("dresscode-mr", output_path, person_path, stages=stages))
        return samples


def build_dataset(
    dataset_name: str,
    data_dir: str,
    output_dir: str,
    paired: bool,
    mr_mode: str,
    mr_part: Optional[str] = None,
    dresscode_part: Optional[str] = None,
    data_list: str = "",
):
    if dataset_name == "viton-hd":
        return VITONHDBatchDataset(data_dir, output_dir, paired)
    if dataset_name == "dresscode":
        return DressCodeBatchDataset(data_dir, output_dir, paired, dresscode_part=dresscode_part)
    if dataset_name == "dresscode-mr":
        return DressCodeMRBatchDataset(data_dir, output_dir, paired, mr_mode=mr_mode, mr_part=mr_part, data_list=data_list)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


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


def get_auto_masker(args, project_root: Path, device: str):
    densepose_dir = Path(args.densepose_ckpt_dir).expanduser().resolve() if args.densepose_ckpt_dir else resolve_ckpt_dir("densepose", "DENSEPOSE_CKPT_DIR", project_root)
    schp_dir = Path(args.schp_ckpt_dir).expanduser().resolve() if args.schp_ckpt_dir else resolve_ckpt_dir("schp", "SCHP_CKPT_DIR", project_root)
    return AutoMaskerFromImage(str(densepose_dir), str(schp_dir), device=device)


def uses_auto_masker_even_with_dataset_mask(sample: TryonSample) -> bool:
    return sample.dataset == "dresscode" and sample.part == "lower"


def maybe_get_mask(sample: TryonSample, person_img: Image.Image, args, auto_masker) -> Image.Image:
    force_auto_for_sample = uses_auto_masker_even_with_dataset_mask(sample)
    if (
        args.use_dataset_mask
        and not force_auto_for_sample
        and sample.mask_path is not None
        and sample.mask_path.exists()
        and not args.force_auto_mask
    ):
        return load_binary_mask(sample.mask_path, args.width, args.height)
    if auto_masker is None:
        raise RuntimeError(f"AutoMasker is required but unavailable for {sample.person_path}.")
    if sample.part is None:
        raise RuntimeError(f"AutoMasker needs sample.part, but got None for {sample.person_path}.")
    mask = auto_masker(
        person_img,
        part=sample.part,
        out_size_hw=None,
        square_cloth_mask=args.square_cloth_mask,
        return_labels=False,
    ).convert("L")
    return binarize_mask(mask, args.width, args.height)


def get_cached_prompt(prompt_cache, args_prompt: str, part: str):
    return prompt_cache["__custom__" if args_prompt.strip() else part]


def process_single_sample(sample: TryonSample, pipe: FluxFillPipeline, args, auto_masker, prompt_cache, gt_output_dir: Optional[Path] = None):
    person_img = load_rgb(sample.person_path, args.width, args.height)
    garment_img = load_rgb(sample.garment_path, args.width, args.height)
    mask_img = maybe_get_mask(sample, person_img, args, auto_masker)
    save_used_mask(mask_img, sample, args)
    cached = get_cached_prompt(prompt_cache, args.prompt, sample.part)
    out = run_tryon(
        pipe=pipe,
        person_img=person_img,
        garment_img=garment_img,
        mask_img=mask_img,
        part=sample.part,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        prompt_embeds=cached["prompt_embeds"],
        pooled_prompt_embeds=cached["pooled_prompt_embeds"],
        max_sequence_length=args.max_sequence_length,
    )
    sample.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(sample.output_path)
    save_processed_gt(person_img, sample, gt_output_dir)


def process_mr_sample(sample: TryonSample, pipe: FluxFillPipeline, args, auto_masker, prompt_cache, gt_output_dir: Optional[Path] = None):
    if auto_masker is None:
        raise RuntimeError("DressCode-MR requires AutoMasker for sequential masking.")
    cur_img = load_rgb(sample.person_path, args.width, args.height)
    gt_img = cur_img.copy()
    for part, garment_path in sample.stages or []:
        if part == "bag":
            continue
        garment_img = load_rgb(garment_path, args.width, args.height)
        mask_img = auto_masker(
            cur_img,
            part=part,
            out_size_hw=None,
            square_cloth_mask=args.square_cloth_mask,
            return_labels=False,
        ).convert("L")
        mask_img = binarize_mask(mask_img, args.width, args.height)
        save_used_mask(mask_img, sample, args, suffix=part)
        cached = get_cached_prompt(prompt_cache, args.prompt, part)
        cur_img = run_tryon(
            pipe=pipe,
            person_img=cur_img,
            garment_img=garment_img,
            mask_img=mask_img,
            part=part,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            prompt_embeds=cached["prompt_embeds"],
            pooled_prompt_embeds=cached["pooled_prompt_embeds"],
            max_sequence_length=args.max_sequence_length,
        )
    sample.output_path.parent.mkdir(parents=True, exist_ok=True)
    cur_img.save(sample.output_path)
    save_processed_gt(gt_img, sample, gt_output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode-mr", "dresscode", "viton-hd"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_list", type=str, default="", help="DressCode-MR json/jsonl file under --data_dir, for example test.jsonl or val_5.jsonl.")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--steps", "--num_inference_steps", dest="steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--pretrained_inpaint_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--densepose_ckpt_dir", type=str, default=None)
    parser.add_argument("--schp_ckpt_dir", type=str, default=None)
    parser.add_argument("--use_dataset_mask", action="store_true")
    parser.add_argument("--force_auto_mask", action="store_true")
    parser.add_argument("--square_cloth_mask", action="store_true")
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--mask_output_dir", type=str, default="")
    parser.add_argument("--no_rgja", action="store_true")
    parser.add_argument("--mr_mode", type=str, default="sequential", choices=["sequential", "first", "expand"])
    parser.add_argument("--mr_part", type=str, default=None, choices=["upper", "lower", "overall", "shoe", "bag"])
    parser.add_argument("--dresscode_part", type=str, default=None, choices=["upper", "lower", "overall"])
    parser.add_argument("--save_gt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    check_min_version("0.30.2")

    if args.max_sequence_length > 512:
        raise ValueError("--max_sequence_length should be <= 512 for FLUX.")
    if args.mr_part is not None and args.dataset != "dresscode-mr":
        raise ValueError("--mr_part is only valid for --dataset dresscode-mr")
    if args.mr_part == "bag":
        raise ValueError("Current pipeline does not support DressCode-MR bag-only inference.")
    if args.dresscode_part is not None and args.dataset != "dresscode":
        raise ValueError("--dresscode_part is only valid for --dataset dresscode")

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    pred_output_dir = Path(args.output_dir).expanduser().resolve()
    gt_output_dir = pred_output_dir / "gt" if args.save_gt else None
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    if gt_output_dir is not None:
        gt_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        args.dataset,
        args.data_dir,
        str(pred_output_dir),
        args.paired,
        args.mr_mode,
        args.mr_part,
        args.dresscode_part,
        args.data_list,
    )
    print(f"[Dataset] {args.dataset}: {len(dataset.samples)} samples pending")
    print(f"[Pred]    {pred_output_dir}")
    print("[Mask]    AutoMasker by default" if not args.use_dataset_mask else "[Mask]    Dataset mask preferred; use --force_auto_mask to override")
    if args.dataset == "dresscode" and args.dresscode_part is not None:
        print(f"[DressCode Part] {args.dresscode_part}")
    if args.dataset == "dresscode-mr":
        print(f"[MR Mode] {args.mr_mode}")
        if args.mr_part is not None:
            print(f"[MR Part] {args.mr_part}")
    if len(dataset.samples) == 0:
        print("All target outputs already exist, nothing to do.")
        return

    transformer_dir, root_dir = resolve_transformer_dir_and_root(args.transformer_path)
    transformer = FluxTransformer2DModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
    pipe = FluxFillPipeline.from_pretrained(
        args.pretrained_inpaint_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if (not args.no_rgja) and device != "cpu":
        ok_rgja = maybe_load_rgja(pipe.transformer, root_dir, device=torch.device(device), dtype=torch_dtype)
        print(f"[RGJA] {'loaded from ' + str(root_dir) + ' (' + getattr(pipe.transformer, 'rgja_injection', 'input_level') + ')' if ok_rgja else 'not found, run without RGJA'}")

    prompt_cache = build_prompt_cache(pipe, device=torch.device(device), dtype=torch_dtype, max_sequence_length=args.max_sequence_length, custom_prompt=args.prompt)
    print("[Prompt] cached custom prompt once." if args.prompt.strip() else "[Prompt] cached upper/lower/overall/shoe prompts.")
    drop_text_encoders(pipe)
    print("[Prompt] text encoders deleted after caching.")

    project_root = Path(__file__).resolve().parent
    has_forced_auto_samples = any(uses_auto_masker_even_with_dataset_mask(s) for s in dataset.samples)
    need_auto_mask = (not args.use_dataset_mask) or args.force_auto_mask or args.dataset == "dresscode-mr" or has_forced_auto_samples
    if args.use_dataset_mask and not args.force_auto_mask:
        need_auto_mask = need_auto_mask or any((s.mask_path is None or not s.mask_path.exists()) for s in dataset.samples)
    auto_masker = get_auto_masker(args, project_root, device) if need_auto_mask else None
    print("[Mask] AutoMasker enabled" if need_auto_mask else "[Mask] AutoMasker disabled")
    if args.use_dataset_mask and has_forced_auto_samples:
        print("[Mask] DressCode lower uses AutoMasker; dataset masks remain preferred for other available masks.")

    ok_count, err_count = 0, 0
    pbar = tqdm(dataset.samples, desc=f"Infer {args.dataset}")
    for sample in pbar:
        try:
            if sample.dataset == "dresscode-mr":
                process_mr_sample(sample, pipe, args, auto_masker, prompt_cache, gt_output_dir)
            else:
                process_single_sample(sample, pipe, args, auto_masker, prompt_cache, gt_output_dir)
            ok_count += 1
        except Exception as e:
            err_count += 1
            print(f"[Error] {sample.person_path}: {e}")
        pbar.set_postfix(ok=ok_count, err=err_count)

    print(f"[Done] ok={ok_count}, err={err_count}")
    if args.dataset == "dresscode-mr":
        if args.mr_part is not None:
            print(f"[Note] DressCode-MR single-part mode: only {args.mr_part} is applied.")
        elif args.mr_mode == "expand":
            print("[Note] DressCode-MR expand mode: each record is expanded into independent upper/lower/overall/shoe samples.")
        else:
            print("[Note] sequential/first are compatibility modes for a single-reference FLUX pipeline.")


if __name__ == "__main__":
    main()
