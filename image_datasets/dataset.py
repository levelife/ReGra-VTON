# coding=utf-8
from typing import Any, Dict, List, Optional, Tuple, Literal

import json
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise ImportError("dataset.py requires opencv-python for mask morphology.") from exc

from parse_utils.automasker import (
    ATR_MAPPING,
    LIP_MAPPING,
    cloth_agnostic_mask,
)


debug_mode = False

CATEGORY_TO_ID = {"upper": 0, "lower": 1, "overall": 2, "shoe": 3}


def tensor_to_image(tensor: torch.Tensor, image_path: str) -> None:
    if not debug_mode:
        return
    if tensor.dim() == 4:
        tensor = tensor[0]
    t_min, t_max = tensor.min(), tensor.max()
    if t_min < 0 or t_max > 1:
        tensor = (tensor - t_min) / (t_max - t_min + 1e-8)
    img = ToPILImage()(tensor.cpu())
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    img.save(image_path)


def _part_mask(part, parse: np.ndarray, mapping: dict) -> np.ndarray:
    if isinstance(part, str):
        part = [part]
    out = np.zeros_like(parse, dtype=np.uint8)
    for name in part:
        if name not in mapping:
            continue
        ids = mapping[name] if isinstance(mapping[name], list) else [mapping[name]]
        out |= np.isin(parse, ids).astype(np.uint8)
    return out


def _dilate(mask: np.ndarray, k: int = 11, it: int = 1) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.float32)
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return (cv2.dilate(mask.astype(np.uint8), kernel, iterations=it) > 0).astype(np.float32)


def _close(mask: np.ndarray, k: int = 9) -> np.ndarray:
    if mask.sum() == 0:
        return mask.astype(np.float32)
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0).astype(np.float32)


def _mask_or(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        out |= np.asarray(mask) > 0.5
    return out.astype(np.float32)


class UnifiedFluxTryOnDataset(data.Dataset):
    DRESSCODE_SUBFOLDERS = ["upper_body", "lower_body", "dresses"]

    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
        data_list: Optional[str] = None,
        dataset_name: str = "dresscode_mr",
    ) -> None:
        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.order = order
        self.height, self.width = size
        self.size = size
        self.dataset_name = self._normalize_dataset_name(dataset_name)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        if data_list is None:
            raise ValueError("data_list must be specified.")

        if self.dataset_name == "dresscode_mr":
            self.samples = self._build_dresscode_mr_samples(data_list)
        elif self.dataset_name == "dresscode":
            self.samples = self._build_dresscode_samples(data_list)
        elif self.dataset_name == "vitonhd":
            self.samples = self._build_vitonhd_samples(data_list)
        elif self.dataset_name == "joint_dresscode_vitonhd":
            self.samples = self._build_joint_dresscode_vitonhd_samples(data_list)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found for dataset_name={self.dataset_name}, "
                f"dataroot={self.dataroot}, phase={self.phase}, data_list={data_list}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize_dataset_name(name: str) -> str:
        name = str(name).strip().lower().replace("-", "_")
        aliases = {
            "viton": "vitonhd", "viton_hd": "vitonhd", "vitonhd": "vitonhd",
            "dresscode": "dresscode", "dresscode_mr": "dresscode_mr",
            "dresscodemr": "dresscode_mr", "mr": "dresscode_mr",
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

    @staticmethod
    def _stem_to_png(filename: str) -> str:
        return os.path.splitext(filename)[0] + ".png"

    @staticmethod
    def _dresscode_suffix_name(filename: str, suffix: str, ext: str = ".png") -> str:
        stem = os.path.splitext(os.path.basename(filename))[0]
        base = stem.rsplit("_", 1)[0] if "_" in stem else stem
        return base + suffix + ext

    @staticmethod
    def _stem_to_mask_png(filename: str) -> str:
        return os.path.splitext(filename)[0] + "_mask.png"

    @staticmethod
    def _candidate_names(name: str) -> List[str]:
        if os.path.isabs(name):
            return [name]
        root, ext = os.path.splitext(name)
        candidates = [name]
        if ext == "":
            candidates.append(name + ".txt")
        elif ext == ".txt":
            candidates.append(root)
        return list(dict.fromkeys(candidates))

    def _resolve_existing_file(self, candidates: List[str], search_roots: List[str]) -> str:
        for cand in candidates:
            if os.path.isabs(cand) and os.path.exists(cand):
                return cand
            for root in search_roots:
                path = os.path.join(root, cand)
                if os.path.exists(path):
                    return path
        raise FileNotFoundError(f"Could not resolve {candidates} under {search_roots}")

    def _load_resize_rgb_abs(self, abs_path: str) -> Image.Image:
        return Image.open(abs_path).convert("RGB").resize((self.width, self.height), Image.BILINEAR)

    def _load_resize_gray_abs(self, abs_path: str) -> Image.Image:
        return Image.open(abs_path).convert("L").resize((self.width, self.height), Image.NEAREST)

    def _load_optional_resize_parse(self, candidates: List[str]) -> Optional[np.ndarray]:
        for path in candidates:
            if path and os.path.exists(path):
                img = Image.open(path).resize((self.width, self.height), Image.NEAREST)
                arr = np.array(img, dtype=np.uint8)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                return arr
        return None

    def _build_agnostic_with_cached_automasker(
        self,
        sample: Dict[str, Any],
        ref_type: str,
        fallback_mask: np.ndarray,
    ) -> np.ndarray:
        atr_map = self._load_optional_resize_parse(sample.get("atr_candidates", []))
        if atr_map is None:
            atr_map = self._load_optional_resize_parse(sample.get("parse_candidates", []))
        if atr_map is None:
            return fallback_mask

        lip_map = self._load_optional_resize_parse(sample.get("lip_candidates", []))
        if lip_map is None:
            lip_map = np.zeros((self.height, self.width), dtype=np.uint8)

        densepose_map = self._load_optional_resize_parse(sample.get("densepose_candidates", []))
        if densepose_map is None:
            return fallback_mask

        try:
            agn_pil = cloth_agnostic_mask(
                densepose_map,
                lip_map,
                atr_map,
                part=ref_type,
                square_cloth_mask=False,
            )
            agnostic_np = np.array(agn_pil)
            if agnostic_np.ndim == 3:
                agnostic_np = agnostic_np[:, :, 0]
            agnostic_np = (agnostic_np > 127).astype(np.float32)
            if agnostic_np.sum() == 0:
                return fallback_mask
            return agnostic_np
        except Exception:
            return fallback_mask

    def _read_pair_lines(self, pair_txt: str) -> List[List[str]]:
        pairs: List[List[str]] = []
        with open(pair_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append(parts)
        return pairs

    def _mask_image_to_agnostic_generic(self, mask_img: Image.Image) -> np.ndarray:
        mask = np.array(mask_img, dtype=np.uint8)
        agnostic = (mask > 127).astype(np.float32)
        if agnostic.mean() > 0.65:
            agnostic = 1.0 - agnostic
        return agnostic

    def _mask_image_to_agnostic_viton(self, mask_img: Image.Image) -> np.ndarray:
        mask = np.array(mask_img, dtype=np.uint8)
        agnostic = (mask > 127).astype(np.float32)
        if agnostic.mean() > 0.5:
            agnostic = 1.0 - agnostic
        return agnostic

    def _build_prompt(self, ref_type: str) -> str:
        prefix = "The pair of images highlights a fashion item and its styling on a model. "
        if ref_type == "upper":
            body = (
                "[IMAGE1] Detailed product shot of the upper garment, clear fabric texture, neckline, sleeves, and silhouette. "
                "[IMAGE2] The same upper garment is worn by a model, natural drape on torso and arms, realistic fit."
            )
        elif ref_type == "lower":
            body = (
                "[IMAGE1] Detailed product shot of the lower garment, clear waistband, hemline, stitching, and fabric texture. "
                "[IMAGE2] The same lower garment is worn by a model, realistic fit on waist, hips, and legs."
            )
        elif ref_type == "overall":
            body = (
                "[IMAGE1] Detailed product shot of the dress or one-piece garment, clear full-body structure and key design elements. "
                "[IMAGE2] The same outfit is worn by a model, natural fit across the full body."
            )
        elif ref_type == "shoe":
            body = (
                "[IMAGE1] Detailed product shot of the shoes, clear material, sole, toe box, and silhouette. "
                "[IMAGE2] The same shoes are worn by a model, complete pair, natural alignment on both feet."
            )
        else:
            body = (
                "[IMAGE1] Detailed product shot of the fashion item, clear texture and shape. "
                "[IMAGE2] The same fashion item is worn by a model, natural fit and realistic body proportion."
            )
        return prefix + body + " No distortion."

    def _build_region_masks_from_ref(
        self,
        ref_type: str,
        agnostic_np: np.ndarray,
        atr_map: Optional[np.ndarray] = None,
        lip_map: Optional[np.ndarray] = None,
        parse_map: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Build category-aware semantic region masks.

        Unlike the previous implementation, this function does not simply copy
        the edit/inpaint mask into the target category channel. It tries to
        construct semantic garment regions from human parsing maps:

            upper   -> Upper-clothes / Coat
            lower   -> Pants / Skirt
            overall -> Dress / Jumpsuits, fallback to upper+lower clothing
            shoe    -> Left-shoe / Right-shoe / Socks

        If parsing is unavailable or a target category mask is empty, it falls
        back to agnostic_np only for the current target category, so training
        will not crash on missing annotations.
        """
        agnostic_np = np.clip(agnostic_np, 0, 1).astype(np.float32)
        region = np.zeros((4, self.height, self.width), dtype=np.float32)

        maps: List[np.ndarray] = []
        for m in (atr_map, lip_map, parse_map):
            if m is None:
                continue
            if m.ndim == 3:
                m = m[:, :, 0]
            maps.append(m.astype(np.uint8))

        def parse_part(names: List[str]) -> np.ndarray:
            if not maps:
                return np.zeros((self.height, self.width), dtype=np.float32)
            out = np.zeros((self.height, self.width), dtype=np.float32)
            for m in maps:
                out = _mask_or(
                    out,
                    _part_mask(names, m, ATR_MAPPING).astype(np.float32),
                    _part_mask(names, m, LIP_MAPPING).astype(np.float32),
                )
            return np.clip(out, 0, 1).astype(np.float32)

        upper = parse_part(["Upper-clothes", "Coat"])
        lower = parse_part(["Pants", "Skirt"])
        overall = parse_part(["Dress", "Jumpsuits"])
        shoe = parse_part(["Left-shoe", "Right-shoe", "Socks"])

        # Morphological refinement. These masks are used for additional region
        # supervision, so they should be more semantic and more compact than the
        # broad inpainting mask.
        upper = _close(_dilate(upper, 7, 1), 7)
        lower = _close(_dilate(lower, 9, 1), 9)
        if overall.sum() == 0:
            overall = _mask_or(upper, lower)
        overall = _close(_dilate(overall, 11, 1), 11)
        shoe = _close(_dilate(shoe, 13, 1), 9)

        semantic_masks = {
            "upper": upper,
            "lower": lower,
            "overall": overall,
            "shoe": shoe,
        }

        # Keep the semantic region inside a loose support of the edit mask.
        # This avoids supervising unrelated parsing regions while still allowing
        # a slightly larger context around the target garment.
        edit_support = _dilate(agnostic_np, 15, 1)
        for name, idx in CATEGORY_TO_ID.items():
            m = semantic_masks.get(name, np.zeros_like(agnostic_np))
            if m.sum() > 0:
                m = m * edit_support
            region[idx] = np.clip(m, 0, 1).astype(np.float32)

        # Robust fallback only for the current target category.
        target_idx = CATEGORY_TO_ID.get(ref_type, 0)
        if region[target_idx].sum() == 0:
            region[target_idx] = agnostic_np

        return torch.from_numpy(region)

    def _build_single_edit_mask_from_parse(
        self,
        ref_type: str,
        parse_map: Optional[np.ndarray],
        fallback_mask: np.ndarray,
    ) -> np.ndarray:
        fallback_mask = np.clip(fallback_mask, 0, 1).astype(np.float32)
        if parse_map is None:
            return fallback_mask
        if parse_map.ndim == 3:
            parse_map = parse_map[:, :, 0]

        def parse_part(names: List[str]) -> np.ndarray:
            return (
                _part_mask(names, parse_map, ATR_MAPPING)
                | _part_mask(names, parse_map, LIP_MAPPING)
            ).astype(np.float32)

        if ref_type == "upper":
            mask = parse_part(["Upper-clothes", "Coat"])
            mask = _close(_dilate(mask, 9, 1), 9)
        elif ref_type == "lower":
            mask = parse_part(["Pants", "Skirt"])
            mask = _close(_dilate(mask, 11, 1), 11)
        elif ref_type == "overall":
            mask = parse_part(["Dress", "Jumpsuits"])
            if mask.sum() == 0:
                mask = _mask_or(
                    parse_part(["Upper-clothes", "Coat"]),
                    parse_part(["Pants", "Skirt"]),
                ).astype(np.float32)
            mask = _close(_dilate(mask, 11, 1), 11)
        elif ref_type == "shoe":
            mask = parse_part(["Left-shoe", "Right-shoe", "Socks"])
            mask = _close(_dilate(mask, 11, 1), 9)
        else:
            mask = fallback_mask

        background = parse_part(["Background"])
        mask = mask * (1.0 - background)
        if mask.sum() == 0:
            return fallback_mask
        return np.clip(mask, 0, 1).astype(np.float32)

    def _parse_union(self, names: List[str], atr_map: np.ndarray, lip_map: np.ndarray) -> np.ndarray:
        return (
            _part_mask(names, atr_map, ATR_MAPPING)
            | _part_mask(names, lip_map, LIP_MAPPING)
        ).astype(np.float32)

    def _build_dresscode_mr_samples(self, data_list: str) -> List[Dict[str, Any]]:
        json_path = os.path.join(self.dataroot, data_list)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{json_path} not found.")

        records: List[Dict[str, Any]] = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        samples: List[Dict[str, Any]] = []
        for rec in records:
            person_rel = rec.get("person")
            if not person_rel:
                continue
            for ref_type in ["overall", "upper", "lower", "shoe"]:
                ref_path = rec.get(ref_type)
                if not ref_path:
                    continue
                samples.append(
                    {
                        "dataset_name": "dresscode_mr",
                        "person_rel": person_rel,
                        "person": os.path.join(self.dataroot, person_rel),
                        "ref_type": ref_type,
                        "cloth_rel": ref_path,
                        "cloth": os.path.join(self.dataroot, ref_path),
                        "prompt": rec.get("prompt", "") or rec.get("caption", "") or rec.get("garment_caption", ""),
                    }
                )
        return samples

    def _load_resize_parsing_mr(self, subdir: str, person_rel: str) -> np.ndarray:
        filename = os.path.basename(person_rel).rsplit(".", 1)[0] + ".png"
        mask_path = os.path.join(self.dataroot, "annotations", subdir, filename)
        if not os.path.exists(mask_path):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        img = Image.open(mask_path).resize((self.width, self.height), Image.NEAREST)
        return np.array(img)

    def _build_dresscode_samples(self, data_list: str) -> List[Dict[str, Any]]:
        candidates = self._candidate_names(data_list)
        samples: List[Dict[str, Any]] = []
        ref_type_map = {"upper_body": "upper", "lower_body": "lower", "dresses": "overall"}
        for sub_folder in self.DRESSCODE_SUBFOLDERS:
            pair_txt = self._resolve_existing_file(candidates, [os.path.join(self.dataroot, sub_folder)])
            for parts in self._read_pair_lines(pair_txt):
                person_img, cloth_img = parts[0], parts[1]
                samples.append(
                    {
                        "dataset_name": "dresscode",
                        "sub_folder": sub_folder,
                        "ref_type": ref_type_map[sub_folder],
                        "person": os.path.join(self.dataroot, sub_folder, "images", person_img),
                        "cloth": os.path.join(self.dataroot, sub_folder, "images", cloth_img),
                        "mask": os.path.join(self.dataroot, sub_folder, "agnostic_masks", self._stem_to_png(person_img)),
                        "parse_candidates": [
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._dresscode_suffix_name(person_img, "_4", ".png")),
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "label_maps", person_img),
                            os.path.join(self.dataroot, sub_folder, "parse_maps", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "image-parse", self._stem_to_png(person_img)),
                        ],
                        "atr_candidates": [
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._dresscode_suffix_name(person_img, "_4", ".png")),
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "label_maps", person_img),
                        ],
                        "lip_candidates": [
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._dresscode_suffix_name(person_img, "_4", ".png")),
                            os.path.join(self.dataroot, sub_folder, "label_maps", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "label_maps", person_img),
                            os.path.join(self.dataroot, sub_folder, "image-parse-v3", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "image-parse-v3", person_img),
                            os.path.join(self.dataroot, sub_folder, "image-parse", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "image-parse", person_img),
                            os.path.join(self.dataroot, sub_folder, "parse_maps", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "parse_maps", person_img),
                        ],
                        "densepose_candidates": [
                            os.path.join(self.dataroot, sub_folder, "densepose_gray", self._dresscode_suffix_name(person_img, "_0", ".png")),
                            os.path.join(self.dataroot, sub_folder, "densepose_gray", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "densepose_gray", person_img),
                            os.path.join(self.dataroot, sub_folder, "image-densepose", self._dresscode_suffix_name(person_img, "_0", ".jpg")),
                            os.path.join(self.dataroot, sub_folder, "image-densepose", self._dresscode_suffix_name(person_img, "_0", ".png")),
                            os.path.join(self.dataroot, sub_folder, "image-densepose", self._stem_to_png(person_img)),
                            os.path.join(self.dataroot, sub_folder, "image-densepose", person_img),
                        ],
                        "im_name": person_img,
                        "c_name": cloth_img,
                    }
                )
        return samples

    def _build_vitonhd_samples(self, data_list: str) -> List[Dict[str, Any]]:
        candidates = self._candidate_names(data_list)
        pair_txt = self._resolve_existing_file(candidates, [self.dataroot, os.path.join(self.dataroot, self.phase)])
        phase_root = os.path.join(self.dataroot, self.phase)
        if not os.path.exists(phase_root):
            raise FileNotFoundError(f"VITON-HD phase folder not found: {phase_root}")

        samples: List[Dict[str, Any]] = []
        for parts in self._read_pair_lines(pair_txt):
            person_img, cloth_img = parts[0], parts[1]
            mask_path = None
            for mp in [
                os.path.join(phase_root, "agnostic-mask", self._stem_to_mask_png(person_img)),
                os.path.join(phase_root, "agnostic-mask", self._stem_to_png(person_img)),
            ]:
                if os.path.exists(mp):
                    mask_path = mp
                    break
            if mask_path is None:
                raise FileNotFoundError(f"No agnostic-mask found for {person_img} in {phase_root}/agnostic-mask")
            samples.append(
                {
                    "dataset_name": "vitonhd",
                    "ref_type": "upper",
                    "person": os.path.join(phase_root, "image", person_img),
                    "cloth": os.path.join(phase_root, "cloth", cloth_img),
                    "mask": mask_path,
                    "parse_candidates": [
                        os.path.join(phase_root, "image-parse-v3", self._stem_to_png(person_img)),
                        os.path.join(phase_root, "image-parse-v3", person_img),
                        os.path.join(phase_root, "image-parse", self._stem_to_png(person_img)),
                        os.path.join(phase_root, "label_maps", self._stem_to_png(person_img)),
                    ],
                    "im_name": person_img,
                    "c_name": cloth_img,
                }
            )
        return samples

    def _split_joint_data_list(self, data_list: str) -> Tuple[str, str]:
        parts = [p.strip() for p in str(data_list).split(",") if p.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        dresscode_list = os.environ.get("DRESSCODE_DATA_LIST", "").strip()
        vitonhd_list = os.environ.get("VITONHD_DATA_LIST", "").strip()
        if dresscode_list and vitonhd_list:
            return dresscode_list, vitonhd_list
        if len(parts) == 1:
            return parts[0], parts[0]
        raise ValueError("Joint dataset requires data_list, for example 'train_pairs.txt,train_pairs.txt'.")

    def _resolve_joint_root(self, env_name: str, candidates: List[str]) -> str:
        env_root = os.environ.get(env_name, "").strip()
        if env_root:
            if os.path.exists(env_root):
                return env_root
            raise FileNotFoundError(f"{env_name} points to a missing path: {env_root}")
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"Could not resolve {env_name}. Tried: {candidates}.")

    @staticmethod
    def _repeat_to_length(samples: List[Dict[str, Any]], target_len: int) -> List[Dict[str, Any]]:
        if len(samples) == 0:
            return []
        return [samples[i % len(samples)] for i in range(target_len)]

    def _build_samples_with_root(self, root: str, build_fn, data_list: str) -> List[Dict[str, Any]]:
        old_root = self.dataroot
        self.dataroot = root
        try:
            return build_fn(data_list)
        finally:
            self.dataroot = old_root

    def _build_joint_dresscode_vitonhd_samples(self, data_list: str) -> List[Dict[str, Any]]:
        dresscode_list, vitonhd_list = self._split_joint_data_list(data_list)
        dresscode_root = self._resolve_joint_root(
            "DRESSCODE_ROOT",
            [os.path.join(self.dataroot, "DressCode"), os.path.join(self.dataroot, "dresscode")],
        )
        vitonhd_root = self._resolve_joint_root(
            "VITONHD_ROOT",
            [os.path.join(self.dataroot, "VITON-HD"), os.path.join(self.dataroot, "VITONHD"), os.path.join(self.dataroot, "vitonhd")],
        )
        dresscode_samples = self._build_samples_with_root(dresscode_root, self._build_dresscode_samples, dresscode_list)
        vitonhd_samples = self._build_samples_with_root(vitonhd_root, self._build_vitonhd_samples, vitonhd_list)
        dc_ratio = int(os.environ.get("JOINT_DRESSCODE_RATIO", "1"))
        viton_ratio = int(os.environ.get("JOINT_VITONHD_RATIO", "1"))
        if dc_ratio <= 0 or viton_ratio <= 0:
            raise ValueError("JOINT_DRESSCODE_RATIO and JOINT_VITONHD_RATIO must be positive.")
        max_len = max(len(dresscode_samples), len(vitonhd_samples))
        dresscode_balanced = self._repeat_to_length(dresscode_samples, max_len)
        vitonhd_balanced = self._repeat_to_length(vitonhd_samples, max_len)
        samples: List[Dict[str, Any]] = []
        for i in range(max_len):
            for _ in range(dc_ratio):
                samples.append(dresscode_balanced[i])
            for _ in range(viton_ratio):
                samples.append(vitonhd_balanced[i])
        return samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        dataset_name = sample["dataset_name"]
        ref_type = sample["ref_type"]

        # These maps are used only for semantic region loss masks.
        # If they are unavailable, _build_region_masks_from_ref falls back safely.
        atr_map_for_region: Optional[np.ndarray] = None
        lip_map_for_region: Optional[np.ndarray] = None
        parse_map_for_region: Optional[np.ndarray] = None

        im_pil_big = self._load_resize_rgb_abs(sample["person"])
        image = self.transform(im_pil_big)
        cloth = self._load_resize_rgb_abs(sample["cloth"])
        cloth_pure = self.transform(cloth)

        if dataset_name == "dresscode_mr":
            person_rel = sample["person_rel"]
            cloth_rel = sample["cloth_rel"]
            atr_map = self._load_resize_parsing_mr("atr", person_rel)
            lip_map = self._load_resize_parsing_mr("lip", person_rel)
            densepose_map = self._load_resize_parsing_mr("densepose", person_rel)
            atr_map_for_region = atr_map
            lip_map_for_region = lip_map
            if ref_type in ("upper", "lower", "overall"):
                agn_pil = cloth_agnostic_mask(
                    densepose_map,
                    lip_map,
                    atr_map,
                    part=ref_type,
                    square_cloth_mask=False,
                )
                agnostic_np = np.array(agn_pil)
                if agnostic_np.ndim == 3:
                    agnostic_np = agnostic_np[:, :, 0]
                agnostic_np = (agnostic_np > 127).astype(np.float32)
            elif ref_type == "shoe":
                agnostic_np = self._parse_union(["Left-shoe", "Right-shoe", "Socks"], atr_map, lip_map)
                agnostic_np = _close(_dilate(agnostic_np, 11, 1), 9)
            else:
                agnostic_np = np.zeros((self.height, self.width), dtype=np.float32)
            prompt = sample.get("prompt", "") or self._build_prompt(ref_type)
            im_name = os.path.basename(person_rel)
            c_name = f"{ref_type}_{os.path.basename(cloth_rel)}"
        else:
            mask_img = self._load_resize_gray_abs(sample["mask"])
            atr_map_for_region = self._load_optional_resize_parse(sample.get("atr_candidates", []))
            lip_map_for_region = self._load_optional_resize_parse(sample.get("lip_candidates", []))
            parse_map_for_region = self._load_optional_resize_parse(sample.get("parse_candidates", []))
            if dataset_name == "dresscode" and ref_type == "lower":
                fallback_mask = self._mask_image_to_agnostic_generic(mask_img)
                parse_map = self._load_optional_resize_parse(sample.get("atr_candidates", sample.get("parse_candidates", [])))
                parse_fallback = self._build_single_edit_mask_from_parse(
                    ref_type,
                    parse_map,
                    fallback_mask.astype(np.float32),
                )
                agnostic_np = self._build_agnostic_with_cached_automasker(
                    sample,
                    ref_type,
                    parse_fallback.astype(np.float32),
                )
            elif dataset_name == "dresscode":
                agnostic_np = self._mask_image_to_agnostic_generic(mask_img)
            elif dataset_name == "vitonhd":
                agnostic_np = self._mask_image_to_agnostic_viton(mask_img)
            else:
                agnostic_np = self._mask_image_to_agnostic_generic(mask_img)
            prompt = self._build_prompt(ref_type)
            im_name = sample["im_name"]
            c_name = f"{ref_type}_{sample['c_name']}"

        agnostic_np = np.clip(agnostic_np, 0, 1).astype(np.float32)

        edit_mask = torch.from_numpy(agnostic_np).unsqueeze(0)
        keep_mask = 1.0 - edit_mask
        im_mask = image * keep_mask

        inpaint_image = torch.cat([cloth_pure, im_mask], dim=2)
        gt_image = torch.cat([cloth_pure, image], dim=2)

        zeros_one = torch.zeros_like(edit_mask)
        inpaint_mask = torch.cat([zeros_one, edit_mask], dim=2)

        region = self._build_region_masks_from_ref(
            ref_type,
            agnostic_np,
            atr_map=atr_map_for_region,
            lip_map=lip_map_for_region,
            parse_map=parse_map_for_region,
        )
        zeros_region = torch.zeros_like(region)
        region_full = torch.cat([zeros_region, region], dim=2)

        result: Dict[str, Any] = {
            "c_name": c_name,
            "im_name": im_name,
            "ref_type": ref_type,
            "category_id": torch.tensor(CATEGORY_TO_ID.get(ref_type, 0), dtype=torch.long),
            "prompt": prompt,
            "cloth_pure": cloth_pure,
            "image": gt_image,
            "im_mask": inpaint_image,
            "inpaint_mask": inpaint_mask,
            "region_masks": region_full,
        }
        return result


DressCodeMRFluxDataset = UnifiedFluxTryOnDataset
VitonHDTestDataset = UnifiedFluxTryOnDataset
