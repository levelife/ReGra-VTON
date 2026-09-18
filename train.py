#!/usr/bin/env python
# coding=utf-8

import json
import argparse
import inspect
import copy
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
    InitProcessGroupKwargs,
    DistributedType,
)
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

try:
    from dataset import DressCodeMRFluxDataset
except Exception:
    from image_datasets.dataset import DressCodeMRFluxDataset

from paser_helper import parse_args
from src.flux.train_utils import prepare_fill_with_mask, prepare_latents, encode_images_to_latents
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

check_min_version("0.30.2")
logger = get_logger(__name__)


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
        x = self.proj(x)
        x = self.encoder(x)
        return self.norm(x)


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
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Hd)

    @torch.no_grad()
    def compute_kv_cache(self, ref_tokens: torch.Tensor):
        ref_tokens = self.norm_kv(ref_tokens)
        k = self._to_heads(self.k_proj(ref_tokens))
        v = self._to_heads(self.v_proj(ref_tokens))
        return (k, v)

    def forward(self, person_tokens: torch.Tensor, ref_tokens: torch.Tensor = None, kv_cache=None) -> torch.Tensor:
        q = self.norm_q(person_tokens)
        q = self._to_heads(self.q_proj(q))

        if kv_cache is None:
            assert ref_tokens is not None
            ref_tokens = self.norm_kv(ref_tokens)
            k = self._to_heads(self.k_proj(ref_tokens))
            v = self._to_heads(self.v_proj(ref_tokens))
        else:
            k, v = kv_cache

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        out = self._from_heads(attn)
        return self.out_proj(out)


def build_prompts_from_batch(batch, args, batch_size: int):
    mode = getattr(args, "prompt_mode", "fixed")
    drop_p = float(getattr(args, "prompt_dropout_prob", 0.0))

    fixed_prompt = (
        "The pair of images highlights a fashion item (garment or shoes) and its styling on a model, high resolution, 4K, 8K; "
        "[IMAGE1] Detailed product shot of the fashion item, clear texture and shape"
        "[IMAGE2] The same fashion item is worn by a model in a lifestyle setting, natural fit and realistic body proportion."
    )

    def fallback(ref_type: str) -> str:
        if ref_type == "shoe":
            return (
                "The pair of images highlights a pair of shoes and its styling on a model, high resolution; "
                "[IMAGE1] Detailed product shot of the shoes, clear material and silhouette. "
                "[IMAGE2] The same shoes are worn by a model in a lifestyle setting, realistic fit and proportion."
            )
        return fixed_prompt

    if mode == "none":
        prompts = [""] * batch_size
    elif mode == "fixed":
        prompts = [fixed_prompt] * batch_size
    else:
        prompts = list(batch.get("prompt", [""] * batch_size))
        ref_types = list(batch.get("ref_type", [""] * batch_size))

        if mode == "template":
            prompts = [fallback(t) for t in ref_types]
        elif mode == "mix":
            out = []
            for p, t in zip(prompts, ref_types):
                if isinstance(p, str) and len(p.strip()) > 0:
                    out.append(p.strip())
                else:
                    out.append(fallback(t))
            prompts = out
        else:
            prompts = [p if isinstance(p, str) else "" for p in prompts]

    if drop_p > 0:
        for i in range(batch_size):
            if random.random() < drop_p:
                prompts[i] = ""
    return prompts


def get_person_token_indices(H_l: int, W_l2: int, device: torch.device, cache: dict):
    key = (H_l, W_l2, str(device))
    if key in cache:
        return cache[key]
    W_half = W_l2 // 2
    idx = []
    for y in range(H_l):
        base = y * W_l2
        idx.extend(range(base + W_half, base + W_l2))
    cache[key] = torch.tensor(idx, device=device, dtype=torch.long)
    return cache[key]


def get_half_split_indices_from_img_ids(img_ids: torch.Tensor, device: torch.device, cache: dict):
    if img_ids.dim() == 3:
        img_ids_ = img_ids[0]
    else:
        img_ids_ = img_ids

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


def normalize_dataset_name(name: str) -> str:
    name = str(name).strip().lower().replace("-", "_")
    aliases = {
        "viton": "vitonhd",
        "viton_hd": "vitonhd",
        "vitonhd": "vitonhd",
        "dresscode": "dresscode",
        "dresscode_mr": "dresscode_mr",
        "dresscodemr": "dresscode_mr",
        "mr": "dresscode_mr",
        "joint": "joint_dresscode_vitonhd",
        "joint_dc_viton": "joint_dresscode_vitonhd",
        "joint_dresscode_viton": "joint_dresscode_vitonhd",
        "joint_dresscode_vitonhd": "joint_dresscode_vitonhd",
        "dresscode_vitonhd": "joint_dresscode_vitonhd",
        "dresscode_viton": "joint_dresscode_vitonhd",
    }
    if name not in aliases:
        raise ValueError(f"Unknown dataset name: {name}")
    return aliases[name]

    if name not in aliases:
        raise ValueError(f"Unknown dataset name: {name}")
    return aliases[name]


def resolve_dataset_name(args) -> str:
    explicit = getattr(args, "dataset_name", None)
    if explicit is None or str(explicit).strip() == "":
        explicit = os.environ.get("DATASET_NAME", "").strip()
    if explicit:
        return normalize_dataset_name(explicit)

    dataroot = str(getattr(args, "dataroot", "")).lower()
    train_list = str(getattr(args, "train_data_list", "")).lower()
    val_list = str(getattr(args, "validation_data_list", "")).lower()

    if any(s in dataroot for s in ["viton-hd", "viton_hd", "vitonhd"]):
        return "vitonhd"
    if "dresscode-mr" in dataroot or "dresscode_mr" in dataroot:
        return "dresscode_mr"
    if "dresscode" in dataroot:
        return "dresscode"
    if train_list.endswith(".json") or train_list.endswith(".jsonl") or val_list.endswith(".json") or val_list.endswith(".jsonl"):
        return "dresscode_mr"
    if "viton" in train_list or "viton" in val_list:
        return "vitonhd"
    return "dresscode"


def dataset_accepts_dataset_name(dataset_cls) -> bool:
    try:
        sig = inspect.signature(dataset_cls.__init__)
        return "dataset_name" in sig.parameters
    except Exception:
        return False


def build_dataset_instance(dataset_cls, dataroot_path: str, phase: str, size, data_list: str, dataset_name: str):
    kwargs = dict(dataroot_path=dataroot_path, phase=phase, size=size, data_list=data_list)
    if dataset_accepts_dataset_name(dataset_cls):
        kwargs["dataset_name"] = dataset_name
    return dataset_cls(**kwargs)


def detect_rgja_bundle(root_dir: str):
    if not root_dir:
        return None
    root = Path(root_dir)
    cfg_path = root / "rgja_config.json"
    ref_path = root / "ref_encoder" / "pytorch_model.bin"
    adp_path = root / "rgja_adapter" / "pytorch_model.bin"
    if not (cfg_path.exists() and ref_path.exists() and adp_path.exists()):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_root_dir"] = str(root)
    cfg["_ref_path"] = str(ref_path)
    cfg["_adp_path"] = str(adp_path)
    return cfg


def restore_rgja_side_modules_from_bundle(base_transformer, bundle_cfg: dict):
    if bundle_cfg is None:
        return False
    ref_path = bundle_cfg.get("_ref_path")
    adp_path = bundle_cfg.get("_adp_path")
    loaded_any = False
    if hasattr(base_transformer, "ref_encoder") and ref_path and os.path.exists(ref_path):
        state = torch.load(ref_path, map_location="cpu")
        base_transformer.ref_encoder.load_state_dict(state, strict=True)
        loaded_any = True
    if hasattr(base_transformer, "rgja_adapter") and adp_path and os.path.exists(adp_path):
        state = torch.load(adp_path, map_location="cpu")
        base_transformer.rgja_adapter.load_state_dict(state, strict=True)
        loaded_any = True
    if loaded_any and "rgja_scale" in bundle_cfg:
        try:
            base_transformer.rgja_scale = float(bundle_cfg.get("rgja_scale", 1.0))
        except Exception:
            pass
    return loaded_any


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"):
    text_encoder_config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, revision=revision)
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length=512, prompt=None, num_images_per_prompt=1, device=None, text_input_ids=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt: str, device=None, text_input_ids=None, num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
    return prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt: str, max_sequence_length, device=None, num_images_per_prompt: int = 1, text_input_ids_list=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )
    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


class PromptEmbeddingCache:
    def __init__(self, max_items: int = 256, store_on_cpu: bool = True, pin_memory: bool = True):
        self.max_items = int(max_items)
        self.store_on_cpu = bool(store_on_cpu)
        self.pin_memory = bool(pin_memory)
        self._cache = OrderedDict()
        self._text_ids = None

    def _maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        if self.pin_memory and t.device.type == "cpu":
            try:
                return t.pin_memory()
            except Exception:
                return t
        return t

    def has(self, prompt: str) -> bool:
        return prompt in self._cache

    def put_many(self, prompts, prompt_embeds, pooled_prompt_embeds, text_ids):
        if self._text_ids is None:
            self._text_ids = text_ids[0].detach() if text_ids.dim() == 3 else text_ids.detach()
            if self.store_on_cpu:
                self._text_ids = self._maybe_pin(self._text_ids.to("cpu"))

        for i, p in enumerate(prompts):
            pe = prompt_embeds[i : i + 1].detach()
            pp = pooled_prompt_embeds[i : i + 1].detach()
            if self.store_on_cpu:
                pe = self._maybe_pin(pe.to("cpu"))
                pp = self._maybe_pin(pp.to("cpu"))
            if p in self._cache:
                self._cache.pop(p, None)
            self._cache[p] = (pe, pp)
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)

    def fetch_batch(self, prompts, device: torch.device, dtype: torch.dtype):
        pes, pps = [], []
        for p in prompts:
            pe_cached, pp_cached = self._cache[p]
            pes.append(pe_cached.to(device=device, dtype=dtype, non_blocking=True))
            pps.append(pp_cached.to(device=device, dtype=dtype, non_blocking=True))
            try:
                self._cache.move_to_end(p)
            except Exception:
                v = self._cache.pop(p)
                self._cache[p] = v

        prompt_embeds = torch.cat(pes, dim=0)
        pooled_prompt_embeds = torch.cat(pps, dim=0)
        if self._text_ids is None:
            raise RuntimeError("PromptEmbeddingCache: text_ids not initialized")
        text_ids = self._text_ids.to(device=device, dtype=dtype, non_blocking=True)
        return prompt_embeds, pooled_prompt_embeds, text_ids


def masked_mean(per_pixel: torch.Tensor, mask: torch.Tensor, eps: float = 1.0):
    mask = mask.to(device=per_pixel.device, dtype=per_pixel.dtype)
    if mask.shape[1] != per_pixel.shape[1]:
        mask = mask.expand(-1, per_pixel.shape[1], -1, -1)
    weighted = per_pixel * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=eps)
    return weighted.sum(dim=(1, 2, 3)) / denom


def build_person_mask_from_edit_mask(edit_mask: torch.Tensor) -> torch.Tensor:
    person_mask = torch.zeros_like(edit_mask)
    w = edit_mask.shape[-1]
    person_mask[..., w // 2 :] = 1.0
    return person_mask


def ref_type_to_region_index(ref_types):
    idx_map = {"upper": 0, "lower": 1, "overall": 2, "shoe": 3}
    return torch.tensor([idx_map.get(str(t), 0) for t in ref_types], dtype=torch.long)


def compute_rgja_injection(
    packed_hidden_states: torch.Tensor,
    clean_latents: torch.Tensor,
    control_mask: torch.Tensor,
    latent_image_ids: torch.Tensor,
    ref_encoder,
    rgja_adapter,
    token_index_cache: dict,
    rgja_scale: float,
    use_kv_cache: bool,
):
    H_l, W_l2 = clean_latents.shape[2], clean_latents.shape[3]
    batch_size = clean_latents.shape[0]
    N = packed_hidden_states.shape[1]
    device = packed_hidden_states.device

    img_ids = latent_image_ids
    use_img_ids = (
        (img_ids is not None)
        and ((img_ids.dim() == 2 and img_ids.shape[0] == N) or (img_ids.dim() == 3 and img_ids.shape[1] == N))
    )
    if use_img_ids:
        cloth_idx, person_idx, _ = get_half_split_indices_from_img_ids(img_ids, device, token_index_cache)
    else:
        person_idx = get_person_token_indices(H_l, W_l2, device, token_index_cache)
        cloth_idx = None

    gate_lat = F.interpolate(control_mask.float(), size=(H_l, W_l2), mode="nearest")
    packed_gate = FluxFillPipeline._pack_latents(
        gate_lat,
        batch_size=batch_size,
        num_channels_latents=1,
        height=H_l,
        width=W_l2,
    )
    if packed_gate.shape[-1] != 1:
        packed_gate = packed_gate.amax(dim=-1, keepdim=True)
    gate_person = packed_gate.index_select(1, person_idx)
    if gate_person.shape[-1] != 1:
        gate_person = gate_person.amax(dim=-1, keepdim=True)

    if cloth_idx is not None:
        clean_packed = FluxFillPipeline._pack_latents(
            clean_latents,
            batch_size=batch_size,
            num_channels_latents=clean_latents.shape[1],
            height=H_l,
            width=W_l2,
        )
        cloth_packed = clean_packed.index_select(1, cloth_idx)
    else:
        W_half = W_l2 // 2
        cloth_lat = clean_latents[:, :, :, :W_half]
        cloth_packed = FluxFillPipeline._pack_latents(
            cloth_lat,
            batch_size=batch_size,
            num_channels_latents=cloth_lat.shape[1],
            height=H_l,
            width=W_half,
        )

    ref_tokens = ref_encoder(cloth_packed.to(dtype=packed_hidden_states.dtype))
    person_tokens = packed_hidden_states.index_select(1, person_idx)

    if use_kv_cache:
        kv_cache = rgja_adapter.compute_kv_cache(ref_tokens)
        delta = rgja_adapter(person_tokens, kv_cache=kv_cache)
    else:
        delta = rgja_adapter(person_tokens, ref_tokens=ref_tokens)

    dst_dtype = packed_hidden_states.dtype
    gate_person = gate_person.to(dtype=dst_dtype)
    delta = delta.to(dtype=dst_dtype)
    rgja_scale_t = packed_hidden_states.new_tensor(rgja_scale)
    person_tokens = person_tokens.to(dtype=dst_dtype)
    person_tokens = person_tokens + gate_person * (rgja_scale_t * delta)

    packed_hidden_states = packed_hidden_states.clone()
    packed_hidden_states.index_copy_(1, person_idx, person_tokens)
    return packed_hidden_states


def install_rgja_forward_hook_for_pipeline(transformer, ref_tokens, gate_person, person_idx, rgja_scale, use_kv_cache):
    if not hasattr(transformer, "rgja_adapter"):
        return None

    original_forward = transformer.forward
    kv_cache = None
    if use_kv_cache:
        with torch.no_grad():
            kv_cache = transformer.rgja_adapter.compute_kv_cache(ref_tokens)

    def wrapped_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states", None)
        args_is_tuple = False
        if hidden_states is None and len(args) > 0:
            args = list(args)
            hidden_states = args[0]
            args_is_tuple = True
        if hidden_states is None:
            return original_forward(*args, **kwargs)

        person_tokens = hidden_states.index_select(1, person_idx)
        if kv_cache is not None:
            delta = transformer.rgja_adapter(person_tokens, kv_cache=kv_cache)
        else:
            delta = transformer.rgja_adapter(person_tokens, ref_tokens=ref_tokens)

        updated = person_tokens.to(dtype=hidden_states.dtype) + gate_person.to(dtype=hidden_states.dtype) * (
            hidden_states.new_tensor(rgja_scale) * delta.to(dtype=hidden_states.dtype)
        )
        hidden_states = hidden_states.clone()
        hidden_states.index_copy_(1, person_idx, updated)

        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden_states
            return original_forward(*args, **kwargs)
        if args_is_tuple:
            args[0] = hidden_states
            return original_forward(*args, **kwargs)
        return original_forward(hidden_states, *args, **kwargs)

    transformer.forward = wrapped_forward
    return original_forward


def prepare_rgja_validation_context(
    batch,
    args,
    accelerator,
    vae,
    runtime_dtype,
    token_index_cache,
    ref_encoder,
):
    control_image = batch["im_mask"].to(device=accelerator.device, dtype=runtime_dtype)
    control_mask = batch["inpaint_mask"].to(device=accelerator.device, dtype=runtime_dtype)

    clean_latents = encode_images_to_latents(vae, control_image, runtime_dtype, args.height, args.width * 2)
    latent_image_ids = prepare_latents(
        2 ** (len(vae.config.block_out_channels) - 1),
        control_image.shape[0],
        args.height,
        args.width * 2,
        runtime_dtype,
        accelerator.device,
    )

    H_l, W_l2 = clean_latents.shape[2], clean_latents.shape[3]
    clean_packed = FluxFillPipeline._pack_latents(
        clean_latents,
        batch_size=clean_latents.shape[0],
        num_channels_latents=clean_latents.shape[1],
        height=H_l,
        width=W_l2,
    )

    N = clean_packed.shape[1]
    img_ids = latent_image_ids
    use_img_ids = (
        (img_ids is not None)
        and ((img_ids.dim() == 2 and img_ids.shape[0] == N) or (img_ids.dim() == 3 and img_ids.shape[1] == N))
    )
    if use_img_ids:
        cloth_idx, person_idx, _ = get_half_split_indices_from_img_ids(img_ids, accelerator.device, token_index_cache)
    else:
        person_idx = get_person_token_indices(H_l, W_l2, accelerator.device, token_index_cache)
        cloth_idx = None

    gate_lat = F.interpolate(control_mask.float(), size=(H_l, W_l2), mode="nearest")
    packed_gate = FluxFillPipeline._pack_latents(
        gate_lat,
        batch_size=clean_latents.shape[0],
        num_channels_latents=1,
        height=H_l,
        width=W_l2,
    )
    if packed_gate.shape[-1] != 1:
        packed_gate = packed_gate.amax(dim=-1, keepdim=True)
    gate_person = packed_gate.index_select(1, person_idx)
    if gate_person.shape[-1] != 1:
        gate_person = gate_person.amax(dim=-1, keepdim=True)

    if cloth_idx is not None:
        cloth_packed = clean_packed.index_select(1, cloth_idx)
    else:
        W_half = W_l2 // 2
        cloth_lat = clean_latents[:, :, :, :W_half]
        cloth_packed = FluxFillPipeline._pack_latents(
            cloth_lat,
            batch_size=cloth_lat.shape[0],
            num_channels_latents=cloth_lat.shape[1],
            height=H_l,
            width=W_half,
        )

    ref_tokens = ref_encoder(cloth_packed.to(dtype=clean_packed.dtype))
    return ref_tokens, gate_person, person_idx


def save_deploy_checkpoint(
    save_dir: str,
    transformer,
    args,
    accelerator,
    enable_rgja_ref: bool,
    rgja_scale: float,
    rgja_heads: int,
    rgja_dropout: float,
    ref_encoder_layers: int,
    d_model: int = None,
    in_dim: int = None,
):
    os.makedirs(save_dir, exist_ok=True)
    transformer_to_save = accelerator.unwrap_model(transformer)
    if is_compiled_module(transformer_to_save):
        transformer_to_save = transformer_to_save._orig_mod

    _tmp_ref = getattr(transformer_to_save, "ref_encoder", None)
    _tmp_adp = getattr(transformer_to_save, "rgja_adapter", None)
    if _tmp_ref is not None:
        try:
            delattr(transformer_to_save, "ref_encoder")
        except Exception:
            pass
    if _tmp_adp is not None:
        try:
            delattr(transformer_to_save, "rgja_adapter")
        except Exception:
            pass

    transformer_to_save.save_pretrained(save_dir, safe_serialization=True)

    if enable_rgja_ref:
        if _tmp_ref is not None:
            transformer_to_save.ref_encoder = _tmp_ref
            ref_dir = os.path.join(save_dir, "ref_encoder")
            os.makedirs(ref_dir, exist_ok=True)
            torch.save(_tmp_ref.state_dict(), os.path.join(ref_dir, "pytorch_model.bin"))
        if _tmp_adp is not None:
            transformer_to_save.rgja_adapter = _tmp_adp
            adp_dir = os.path.join(save_dir, "rgja_adapter")
            os.makedirs(adp_dir, exist_ok=True)
            torch.save(_tmp_adp.state_dict(), os.path.join(adp_dir, "pytorch_model.bin"))
        rgja_cfg = {
            "enable_rgja_ref": True,
            "rgja_scale": float(rgja_scale),
            "rgja_heads": int(rgja_heads),
            "rgja_dropout": float(rgja_dropout),
            "ref_encoder_layers": int(ref_encoder_layers),
            "d_model": int(d_model) if d_model is not None else None,
            "in_dim": int(in_dim) if in_dim is not None else None,
        }
        with open(os.path.join(save_dir, "rgja_config.json"), "w", encoding="utf-8") as f:
            json.dump(rgja_cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def load_deploy_checkpoint_weights_only(load_dir: str, transformer, accelerator, enable_rgja_ref: bool):
    base = accelerator.unwrap_model(transformer)
    if is_compiled_module(base):
        base = base._orig_mod
    load_model = FluxTransformer2DModel.from_pretrained(load_dir, subfolder=None)
    base.register_to_config(**load_model.config)
    base.load_state_dict(load_model.state_dict(), strict=False)
    del load_model

    if enable_rgja_ref:
        p_ref = os.path.join(load_dir, "ref_encoder", "pytorch_model.bin")
        if hasattr(base, "ref_encoder") and os.path.exists(p_ref):
            state = torch.load(p_ref, map_location="cpu")
            base.ref_encoder.load_state_dict(state, strict=True)
        p_adp = os.path.join(load_dir, "rgja_adapter", "pytorch_model.bin")
        if hasattr(base, "rgja_adapter") and os.path.exists(p_adp):
            state = torch.load(p_adp, map_location="cpu")
            base.rgja_adapter.load_state_dict(state, strict=True)


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    dataloader,
    tag,
    vae,
    runtime_dtype,
    token_index_cache,
    enable_rgja_ref=False,
    use_kv_cache=False,
    rgja_scale=1.0,
    is_final_validation=False,
):
    logger.info(f"Running {tag}...")
    pipeline = pipeline.to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    autocast_ctx = accelerator.autocast()

    images, prompts = [], []
    control_images, control_masks = [], []

    base_transformer = accelerator.unwrap_model(pipeline.transformer)
    if is_compiled_module(base_transformer):
        base_transformer = base_transformer._orig_mod

    with autocast_ctx:
        for batch in dataloader:
            bs = batch["image"].shape[0]
            prompt = build_prompts_from_batch(batch, args, batch_size=bs)

            control_image = batch["im_mask"]
            control_mask = batch["inpaint_mask"]

            restore_forward = None
            if enable_rgja_ref and hasattr(base_transformer, "ref_encoder") and hasattr(base_transformer, "rgja_adapter"):
                ref_tokens, gate_person, person_idx = prepare_rgja_validation_context(
                    batch=batch,
                    args=args,
                    accelerator=accelerator,
                    vae=vae,
                    runtime_dtype=runtime_dtype,
                    token_index_cache=token_index_cache,
                    ref_encoder=base_transformer.ref_encoder,
                )
                restore_forward = install_rgja_forward_hook_for_pipeline(
                    transformer=base_transformer,
                    ref_tokens=ref_tokens,
                    gate_person=gate_person,
                    person_idx=person_idx,
                    rgja_scale=rgja_scale,
                    use_kv_cache=use_kv_cache,
                )

            try:
                result = pipeline(
                    prompt=prompt,
                    height=args.height,
                    width=args.width * 2,
                    image=control_image,
                    mask_image=control_mask,
                    num_inference_steps=28,
                    generator=generator,
                    guidance_scale=30,
                ).images
            finally:
                if restore_forward is not None:
                    base_transformer.forward = restore_forward

            images.extend(result)
            prompts.extend(prompt)
            control_images.extend(control_image)
            control_masks.extend(control_mask)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    tag: [wandb.Image(img, caption=f"{i} {pr}") for i, (img, pr) in enumerate(zip(images, prompts))],
                    f"{tag}_control_images": [wandb.Image(ci, caption=f"{i} Control Image") for i, ci in enumerate(control_images)],
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return images


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Do not use --report_to=wandb together with --hub_token.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    timeout_sec = int(os.getenv("DDP_TIMEOUT", os.getenv("DEEPSPEED_TIMEOUT", "7200")))
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_sec))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = resolve_dataset_name(args)
    setattr(args, "dataset_name", dataset_name)
    logger.info(f"Resolved dataset_name = {dataset_name}")

    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision)
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder")
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=True, do_convert_rgb=True, do_normalize=True)
    mask_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_resize=True, do_convert_grayscale=True, do_normalize=False, do_binarize=True)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        runtime_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        runtime_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
        runtime_dtype = torch.float32

    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_inpaint_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )

    pretrained_rgja_bundle = detect_rgja_bundle(args.pretrained_inpaint_model_name_or_path)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_one.eval()
    text_encoder_two.eval()

    grad_params = [
        "transformer_blocks.0.","transformer_blocks.1.","transformer_blocks.2.","transformer_blocks.3.","transformer_blocks.4.",
        "transformer_blocks.5.","transformer_blocks.6.","transformer_blocks.7.","transformer_blocks.8.","transformer_blocks.9.",
        "transformer_blocks.10.","transformer_blocks.11.","transformer_blocks.12.","transformer_blocks.13.","transformer_blocks.14.",
        "transformer_blocks.15.","transformer_blocks.16.","transformer_blocks.17.","transformer_blocks.18.",
        "single_transformer_blocks.0.","single_transformer_blocks.1.","single_transformer_blocks.2.","single_transformer_blocks.3.",
        "single_transformer_blocks.4.","single_transformer_blocks.5.","single_transformer_blocks.6.","single_transformer_blocks.7.",
        "single_transformer_blocks.8.","single_transformer_blocks.9.","single_transformer_blocks.10.",
        "single_transformer_blocks.13.","single_transformer_blocks.14.","single_transformer_blocks.15.","single_transformer_blocks.16.",
        "single_transformer_blocks.17.","single_transformer_blocks.18.","single_transformer_blocks.19.","single_transformer_blocks.20.",
        "single_transformer_blocks.21.","single_transformer_blocks.22.","single_transformer_blocks.23.","single_transformer_blocks.24.",
        "single_transformer_blocks.25.","single_transformer_blocks.26.","single_transformer_blocks.27.","single_transformer_blocks.28.",
        "single_transformer_blocks.29.","single_transformer_blocks.30.","single_transformer_blocks.31.","single_transformer_blocks.32.",
        "single_transformer_blocks.33.","single_transformer_blocks.34.","single_transformer_blocks.35.","single_transformer_blocks.36.",
        "single_transformer_blocks.37.",
    ]
    if args.train_base_model:
        transformer.requires_grad_(False)
        for name, param in transformer.named_parameters():
            if any(gp in name for gp in grad_params) and ("attn" in name):
                param.requires_grad = True

    vae.to(accelerator.device, dtype=runtime_dtype)
    text_encoder_one.to(accelerator.device, dtype=runtime_dtype)
    text_encoder_two.to(accelerator.device, dtype=runtime_dtype)
    transformer.to(accelerator.device, dtype=runtime_dtype)

    if args.gradient_checkpointing and args.train_base_model:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        m = accelerator.unwrap_model(model)
        m = m._orig_mod if is_compiled_module(m) else m
        return m

    train_dataset = build_dataset_instance(
        DressCodeMRFluxDataset, args.dataroot, "train", (args.height, args.width), args.train_data_list, dataset_name
    )
    train_verification_dataset = build_dataset_instance(
        DressCodeMRFluxDataset, args.dataroot, "train", (args.height, args.width), args.train_verification_list, dataset_name
    )
    validation_dataset = build_dataset_instance(
        DressCodeMRFluxDataset, args.dataroot, "test", (args.height, args.width), args.validation_data_list, dataset_name
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    train_verification_dataloader = torch.utils.data.DataLoader(train_verification_dataset, shuffle=True, batch_size=args.train_batch_size)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, shuffle=True, batch_size=args.train_batch_size)

    d_model = None
    in_dim = None
    enable_rgja_ref = bool(getattr(args, "enable_rgja_ref", False))
    if pretrained_rgja_bundle is not None and not enable_rgja_ref:
        enable_rgja_ref = True
        setattr(args, "enable_rgja_ref", True)

    use_kv_cache = bool(getattr(args, "use_kv_cache", False))
    rgja_scale = float(getattr(args, "rgja_scale", 1.0))
    rgja_heads = int(getattr(args, "rgja_heads", 8))
    rgja_dropout = float(getattr(args, "rgja_dropout", 0.0))
    ref_encoder_layers = int(getattr(args, "ref_encoder_layers", 2))
    if pretrained_rgja_bundle is not None:
        rgja_scale = float(pretrained_rgja_bundle.get("rgja_scale", rgja_scale))
        rgja_heads = int(pretrained_rgja_bundle.get("rgja_heads", rgja_heads))
        rgja_dropout = float(pretrained_rgja_bundle.get("rgja_dropout", rgja_dropout))
        ref_encoder_layers = int(pretrained_rgja_bundle.get("ref_encoder_layers", ref_encoder_layers))

    rgja_lr = float(getattr(args, "rgja_lr", args.learning_rate) or args.learning_rate)
    token_index_cache = {}
    ref_encoder = None
    rgja_adapter = None

    # Optimizer will be built after optional RGJA modules are created.
    # Do NOT pass transformer.parameters() directly here: for DeepSpeed ZeRO this can
    # make frozen FLUX parameters participate in optimizer partition/flatten logic and
    # cause OOM during the first optimizer step.

    if enable_rgja_ref:
        probe_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, batch_size=1)
        probe_batch = next(iter(probe_loader))
        with torch.no_grad():
            probe_control_mask = probe_batch["inpaint_mask"].to(device=accelerator.device, dtype=vae.dtype)
            probe_control_image = probe_batch["im_mask"].to(device=accelerator.device, dtype=vae.dtype)
            probe_pixel_values = probe_batch["image"].to(device=accelerator.device, dtype=vae.dtype)
            probe_inpaint_cond, _, _ = prepare_fill_with_mask(
                image_processor=image_processor,
                mask_processor=mask_processor,
                vae=vae,
                vae_scale_factor=vae_scale_factor,
                image=probe_control_image,
                mask=probe_control_mask,
                width=args.width * 2,
                height=args.height,
                batch_size=1,
                num_images_per_prompt=1,
                device=accelerator.device,
                dtype=runtime_dtype,
            )
            probe_latents = encode_images_to_latents(vae, probe_pixel_values, runtime_dtype, args.height, args.width * 2)
            probe_clean_packed = FluxFillPipeline._pack_latents(
                probe_latents,
                batch_size=probe_latents.shape[0],
                num_channels_latents=probe_latents.shape[1],
                height=probe_latents.shape[2],
                width=probe_latents.shape[3],
            )
            in_dim = probe_clean_packed.shape[-1]
            probe_packed = probe_clean_packed
            if probe_inpaint_cond is not None:
                probe_packed = torch.cat([probe_packed, probe_inpaint_cond], dim=-1)
            d_model = probe_packed.shape[-1]

        ref_encoder = RefTokenEncoder(in_dim=in_dim, d_model=d_model, n_layers=ref_encoder_layers, n_heads=rgja_heads, dropout=rgja_dropout).to(
            device=accelerator.device, dtype=runtime_dtype
        )
        rgja_adapter = RGJACrossAttnAdapter(d_model=d_model, n_heads=rgja_heads, dropout=rgja_dropout).to(
            device=accelerator.device, dtype=runtime_dtype
        )
        transformer.ref_encoder = ref_encoder
        transformer.rgja_adapter = rgja_adapter
        if pretrained_rgja_bundle is not None:
            restore_rgja_side_modules_from_bundle(transformer, pretrained_rgja_bundle)

    # Build optimizer from trainable parameters only.
    # Important: after transformer.ref_encoder / transformer.rgja_adapter are attached,
    # transformer.named_parameters() also includes those RGJA modules. Exclude them
    # from the base-transformer group, then add them as separate groups with rgja_lr.
    params_to_optimize = []

    trainable_transformer_params = [
        p for n, p in transformer.named_parameters()
        if p.requires_grad and not n.startswith(("ref_encoder.", "rgja_adapter."))
    ]
    if len(trainable_transformer_params) > 0:
        params_to_optimize.append({"params": trainable_transformer_params, "lr": args.learning_rate})

    if enable_rgja_ref:
        if ref_encoder is None or rgja_adapter is None:
            raise RuntimeError("enable_rgja_ref=True, but RGJA modules were not created.")
        ref_params = [p for p in ref_encoder.parameters() if p.requires_grad]
        adapter_params = [p for p in rgja_adapter.parameters() if p.requires_grad]
        if len(ref_params) > 0:
            params_to_optimize.append({"params": ref_params, "lr": rgja_lr})
        if len(adapter_params) > 0:
            params_to_optimize.append({"params": adapter_params, "lr": rgja_lr})

    # Extra guard: remove duplicate parameter objects across groups.
    seen_param_ids = set()
    for group in params_to_optimize:
        unique_params = []
        for p in group["params"]:
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            unique_params.append(p)
        group["params"] = unique_params
    params_to_optimize = [g for g in params_to_optimize if len(g["params"]) > 0]

    if len(params_to_optimize) == 0:
        raise RuntimeError(
            "No trainable parameters found. Enable --train_base_model and/or --enable_rgja_ref."
        )

    total_trainable_params = sum(p.numel() for group in params_to_optimize for p in group["params"])
    logger.info(f"Trainable parameter count passed to optimizer: {total_trainable_params:,}")

    if getattr(args, "optimizer", "adamw").lower() != "adamw":
        args.optimizer = "adamw"

    if getattr(args, "use_8bit_adam", False):
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # Optional text-encoder offload. This saves VRAM after prompt embeddings are
    # cached, at the cost of slower cache misses. Enable with:
    #   export OFFLOAD_TEXT_ENCODERS=1
    offload_text_encoders = bool(getattr(args, "offload_text_encoders", False)) or (
        os.getenv("OFFLOAD_TEXT_ENCODERS", "0") == "1"
    )
    prompt_cache_max = int(getattr(args, "prompt_cache_max", int(os.getenv("PROMPT_CACHE_MAX", "256"))))
    prompt_cache_on_cpu = bool(getattr(args, "prompt_cache_on_cpu", True))
    prompt_cache = PromptEmbeddingCache(
        max_items=prompt_cache_max,
        store_on_cpu=prompt_cache_on_cpu,
        pin_memory=True,
    )
    _text_encoders_on_gpu = True

    def _offload_text_encoders_to_cpu():
        nonlocal _text_encoders_on_gpu
        if not offload_text_encoders or not _text_encoders_on_gpu:
            return
        text_encoder_one.to("cpu")
        text_encoder_two.to("cpu")
        _text_encoders_on_gpu = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_text_encoders_on_gpu():
        nonlocal _text_encoders_on_gpu
        if _text_encoders_on_gpu:
            return
        text_encoder_one.to(accelerator.device, dtype=runtime_dtype)
        text_encoder_two.to(accelerator.device, dtype=runtime_dtype)
        _text_encoders_on_gpu = True

    def compute_text_embeddings(prompts):
        prompts_ = [prompts] if isinstance(prompts, str) else list(prompts)
        missing = []
        seen = set()
        for p in prompts_:
            p = "" if p is None else str(p)
            if p not in seen:
                seen.add(p)
            if not prompt_cache.has(p):
                missing.append(p)
        if missing:
            _ensure_text_encoders_on_gpu()
            with torch.no_grad():
                pe, pp, ti = encode_prompt(text_encoders, tokenizers, missing, args.max_sequence_length, device=accelerator.device)
                pe = pe.to(device=accelerator.device, dtype=runtime_dtype)
                pp = pp.to(device=accelerator.device, dtype=runtime_dtype)
                ti = ti.to(device=accelerator.device, dtype=runtime_dtype)
            prompt_cache.put_many(missing, pe, pp, ti)
            _offload_text_encoders_to_cpu()
        return prompt_cache.fetch_batch(prompts_, device=accelerator.device, dtype=runtime_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Clip only parameters actually managed by the optimizer, not the whole
    # transformer. This avoids touching frozen FLUX weights.
    params_to_clip = [
        p
        for group in optimizer.param_groups
        for p in group["params"]
        if getattr(p, "requires_grad", False)
    ]

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        if getattr(accelerator, "deepspeed_engine_wrapped", None) is None:
            accelerator.deepspeed_engine_wrapped = transformer

    if enable_rgja_ref:
        base = accelerator.unwrap_model(transformer)
        if is_compiled_module(base):
            base = base._orig_mod
        ref_encoder = getattr(base, "ref_encoder", ref_encoder)
        rgja_adapter = getattr(base, "rgja_adapter", rgja_adapter)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "dreambooth-flux-inpaint",
            config=vars(args),
            init_kwargs={"wandb": {"settings": wandb.Settings(code_dir=".")}} if args.report_to == "wandb" else None,
        )

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            ckpt_dir = os.path.join(args.output_dir, path)
            load_deploy_checkpoint_weights_only(ckpt_dir, transformer, accelerator, enable_rgja_ref=enable_rgja_ref)
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.train_base_model:
            transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                batch_size = batch["image"].shape[0]
                pixel_values = batch["image"].to(device=accelerator.device, dtype=vae.dtype, non_blocking=True)
                prompts = build_prompts_from_batch(batch, args, batch_size=batch_size)
                control_mask = batch["inpaint_mask"].to(device=accelerator.device, dtype=vae.dtype, non_blocking=True)
                control_image = batch["im_mask"].to(device=accelerator.device, dtype=vae.dtype, non_blocking=True)

                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(prompts)
                inpaint_cond, _, _ = prepare_fill_with_mask(
                    image_processor=image_processor,
                    mask_processor=mask_processor,
                    vae=vae,
                    vae_scale_factor=vae_scale_factor,
                    image=control_image,
                    mask=control_mask,
                    width=args.width * 2,
                    height=args.height,
                    batch_size=batch_size,
                    num_images_per_prompt=1,
                    device=accelerator.device,
                    dtype=runtime_dtype,
                )
                if args.dropout_prob > 0:
                    inpaint_cond = torch.nn.Dropout(p=args.dropout_prob)(inpaint_cond)

                model_input = encode_images_to_latents(vae, pixel_values, runtime_dtype, args.height, args.width * 2)
                latent_image_ids = prepare_latents(vae_scale_factor, batch_size, args.height, args.width * 2, runtime_dtype, accelerator.device)
                noise = torch.randn_like(model_input)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=model_input.shape[0],
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxFillPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )
                guidance = torch.full([1], args.guidance_scale, device=accelerator.device).expand(model_input.shape[0])
                if inpaint_cond is not None:
                    packed_noisy_model_input = torch.cat([packed_noisy_model_input, inpaint_cond], dim=-1)

                if enable_rgja_ref and (ref_encoder is not None) and (rgja_adapter is not None):
                    packed_noisy_model_input = compute_rgja_injection(
                        packed_hidden_states=packed_noisy_model_input,
                        clean_latents=model_input,
                        control_mask=control_mask,
                        latent_image_ids=latent_image_ids,
                        ref_encoder=ref_encoder,
                        rgja_adapter=rgja_adapter,
                        token_index_cache=token_index_cache,
                        rgja_scale=rgja_scale,
                        use_kv_cache=use_kv_cache,
                    )

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=args.height,
                    width=args.width * 2,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                per_pixel = weighting.float().view(-1, 1, 1, 1) * (model_pred.float() - target.float()) ** 2

                B, _, H_l, W_l = per_pixel.shape
                edit_mask_lat = F.interpolate(control_mask.float(), size=(H_l, W_l), mode="nearest").to(device=model_pred.device, dtype=per_pixel.dtype)
                person_mask_lat = build_person_mask_from_edit_mask(edit_mask_lat).to(device=model_pred.device, dtype=per_pixel.dtype)
                base_person_loss = masked_mean(per_pixel, person_mask_lat)
                edit_region_loss = masked_mean(per_pixel, edit_mask_lat)
                lambda_edit = float(getattr(args, "edit_region_loss_lambda", 1.0))

                region_masks = batch.get("region_masks", None)
                if region_masks is not None:
                    region_masks = region_masks.to(device=model_pred.device, dtype=per_pixel.dtype)
                    region_masks = F.interpolate(region_masks, size=(H_l, W_l), mode="nearest")
                    target_region_idx = ref_type_to_region_index(batch["ref_type"]).to(model_pred.device)
                    gather_index = target_region_idx.view(B, 1, 1, 1).expand(B, 1, H_l, W_l)
                    target_region_mask = torch.gather(region_masks, dim=1, index=gather_index)
                    target_region_loss = masked_mean(per_pixel, target_region_mask)
                    region_alpha_map = {"upper": 1.2, "lower": 1.2, "overall": 0.8, "shoe": 2.0}
                    alpha = torch.tensor(
                        [region_alpha_map.get(str(t), 1.0) for t in batch["ref_type"]],
                        device=model_pred.device,
                        dtype=per_pixel.dtype,
                    )
                    lambda_reg = float(getattr(args, "region_loss_lambda", 1.0))
                    loss = (
                        base_person_loss
                        + lambda_edit * edit_region_loss
                        + lambda_reg * alpha * target_region_loss
                    ).mean()
                else:
                    loss = (base_person_loss + lambda_edit * edit_region_loss).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients and len(params_to_clip) > 0:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for rm in checkpoints[0:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, rm), ignore_errors=True)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if accelerator.is_main_process:
                        save_deploy_checkpoint(
                            save_dir=save_path,
                            transformer=transformer,
                            args=args,
                            accelerator=accelerator,
                            enable_rgja_ref=enable_rgja_ref,
                            rgja_scale=rgja_scale,
                            rgja_heads=rgja_heads,
                            rgja_dropout=rgja_dropout,
                            ref_encoder_layers=ref_encoder_layers,
                            d_model=d_model,
                            in_dim=in_dim,
                        )
                    accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.sync_gradients and (global_step > 10) and (global_step % args.validation_steps == 0):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if accelerator.is_main_process:
                    try:
                        _ensure_text_encoders_on_gpu()
                    except Exception:
                        pass

                    pipeline = FluxFillPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        transformer=accelerator.unwrap_model(transformer),
                        torch_dtype=runtime_dtype,
                        vae=vae,
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        text_encoder=text_encoder_one,
                        text_encoder_2=text_encoder_two,
                    )

                    log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        dataloader=train_verification_dataloader,
                        tag="train verification",
                        epoch=epoch,
                        vae=vae,
                        runtime_dtype=runtime_dtype,
                        token_index_cache=token_index_cache,
                        enable_rgja_ref=enable_rgja_ref,
                        use_kv_cache=use_kv_cache,
                        rgja_scale=rgja_scale,
                    )
                    log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        dataloader=validation_dataloader,
                        tag="validation",
                        epoch=epoch,
                        vae=vae,
                        runtime_dtype=runtime_dtype,
                        token_index_cache=token_index_cache,
                        enable_rgja_ref=enable_rgja_ref,
                        use_kv_cache=use_kv_cache,
                        rgja_scale=rgja_scale,
                    )
                    del pipeline
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        _offload_text_encoders_to_cpu()
                    except Exception:
                        pass
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_transformer")
        os.makedirs(final_dir, exist_ok=True)
        transformer_to_save = unwrap_model(transformer)
        _tmp_ref = getattr(transformer_to_save, "ref_encoder", None)
        _tmp_adp = getattr(transformer_to_save, "rgja_adapter", None)
        if _tmp_ref is not None:
            try:
                delattr(transformer_to_save, "ref_encoder")
            except Exception:
                pass
        if _tmp_adp is not None:
            try:
                delattr(transformer_to_save, "rgja_adapter")
            except Exception:
                pass
        transformer_to_save.save_pretrained(final_dir, safe_serialization=True)
        if enable_rgja_ref:
            if _tmp_ref is not None:
                transformer_to_save.ref_encoder = _tmp_ref
                ref_dir = os.path.join(final_dir, "ref_encoder")
                os.makedirs(ref_dir, exist_ok=True)
                torch.save(_tmp_ref.state_dict(), os.path.join(ref_dir, "pytorch_model.bin"))
            if _tmp_adp is not None:
                transformer_to_save.rgja_adapter = _tmp_adp
                adp_dir = os.path.join(final_dir, "rgja_adapter")
                os.makedirs(adp_dir, exist_ok=True)
                torch.save(_tmp_adp.state_dict(), os.path.join(adp_dir, "pytorch_model.bin"))
            rgja_cfg = {
                "enable_rgja_ref": True,
                "rgja_scale": float(rgja_scale),
                "rgja_heads": int(rgja_heads),
                "rgja_dropout": float(rgja_dropout),
                "ref_encoder_layers": int(ref_encoder_layers),
                "d_model": int(d_model) if d_model is not None else None,
                "in_dim": int(in_dim) if in_dim is not None else None,
            }
            with open(os.path.join(final_dir, "rgja_config.json"), "w", encoding="utf-8") as fcfg:
                json.dump(rgja_cfg, fcfg, ensure_ascii=False, indent=2)
        with open(os.path.join(final_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)