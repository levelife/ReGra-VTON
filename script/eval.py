import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from cleanfid import fid as FID
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm
from prettytable import PrettyTable


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_files_in_dir(directory: str, postfix=None) -> List[Path]:
    directory = Path(directory)
    files = []
    for p in directory.rglob("*"):
        if p.is_file():
            if postfix is None or p.suffix.lower() in postfix:
                files.append(p)
    return sorted(files)


def extract_id_from_filename(filename: str) -> str:
    # 与你参考的原版逻辑一致：找到第一个数字，从那里起截 8 个字符
    start_i = None
    for i, c in enumerate(filename):
        if c.isdigit():
            start_i = i
            break
    if start_i is None:
        raise ValueError(f"Cannot find number in filename: {filename}")
    return filename[start_i:start_i + 8]


def resize_to_height(img: Image.Image, height: int) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    new_w = int(round(w * height / h))
    return img.resize((new_w, height), Image.LANCZOS)


def copy_resize_folder_to_height(src_folder: str, height: int) -> str:
    """
    类似原版 copy_resize_gt，但这里做成递归版本，更稳。
    只按高度缩放，宽度等比例变化。
    """
    src_folder = Path(src_folder)
    out_folder = src_folder.parent / f"{src_folder.name}_{height}"

    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)

    src_files = scan_files_in_dir(str(src_folder), postfix=VALID_EXTS)
    for src_path in tqdm(src_files, desc=f"Resizing {src_folder.name} -> h={height}"):
        rel = src_path.relative_to(src_folder)
        dst_path = out_folder / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            continue

        img = Image.open(src_path).convert("RGB")
        img = resize_to_height(img, height)
        img.save(dst_path)

    return str(out_folder)


def get_first_image_height(folder: str) -> int:
    files = scan_files_in_dir(folder, postfix=VALID_EXTS)
    if not files:
        raise FileNotFoundError(f"No image found in folder: {folder}")
    img = Image.open(files[0])
    return img.height


def build_pairs(gt_folder: str, pred_folder: str) -> Tuple[List[Tuple[str, str]], str]:
    gt_files = scan_files_in_dir(gt_folder, postfix=VALID_EXTS)
    pred_files = scan_files_in_dir(pred_folder, postfix=VALID_EXTS)

    if not gt_files:
        raise FileNotFoundError(f"No GT image found in: {gt_folder}")
    if not pred_files:
        raise FileNotFoundError(f"No prediction image found in: {pred_folder}")

    # 先尝试同名匹配
    gt_name_map = {p.name: p for p in gt_files}
    pred_name_map = {p.name: p for p in pred_files}
    common_names = sorted(set(gt_name_map.keys()) & set(pred_name_map.keys()))
    if common_names:
        pairs = [(str(gt_name_map[name]), str(pred_name_map[name])) for name in common_names]
        return pairs, "exact_filename"

    # 如果同名匹配不到，再退回到原版那种 id 匹配
    gt_id_map = {extract_id_from_filename(p.name): p for p in gt_files}
    pairs = []
    for pred_path in pred_files:
        pred_id = extract_id_from_filename(pred_path.name)
        if pred_id in gt_id_map:
            pairs.append((str(gt_id_map[pred_id]), str(pred_path)))

    if not pairs:
        raise ValueError(
            f"No matched pairs found.\n"
            f"GT: {gt_folder}\n"
            f"Pred: {pred_folder}\n"
            f"Tried both exact filename matching and 8-char id matching."
        )

    return pairs, "first_8char_id"


class EvalDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], target_height: int):
        self.data = pairs
        self.target_height = target_height
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]

        gt = Image.open(gt_path).convert("RGB")
        pred = Image.open(pred_path).convert("RGB")

        # 类似原版：都按相同目标高度等比例缩放
        gt = resize_to_height(gt, self.target_height)
        pred = resize_to_height(pred, self.target_height)

        # 为了更稳，若高度统一后宽度仍不同，则把 pred 拉到 gt 的尺寸
        # 这一步比原版更保险，避免 paired 指标时 tensor shape 不一致
        if gt.size != pred.size:
            pred = pred.resize(gt.size, Image.LANCZOS)

        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)
        return gt, pred


@torch.no_grad()
def calc_ssim(dataloader, device):
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    score = 0.0
    total = 0

    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        bs = gt.size(0)
        gt = gt.to(device, non_blocking=True)
        pred = pred.to(device, non_blocking=True)
        score += float(metric(pred, gt)) * bs
        total += bs

    return score / max(total, 1)


@torch.no_grad()
def calc_lpips(dataloader, device):
    metric = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
    score = 0.0
    total = 0

    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        bs = gt.size(0)
        gt = gt.to(device, non_blocking=True)
        pred = pred.to(device, non_blocking=True)

        gt = gt * 2 - 1
        pred = pred * 2 - 1

        score += float(metric(gt, pred)) * bs
        total += bs

    return score / max(total, 1)


def eval_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 参考原版：用 pred 的高度作为目标高度
    pred_target_height = get_first_image_height(args.pred_folder)

    # 比原版更稳：全量检查 GT 高度，只要有不一致就整体复制+resize
    gt_files = scan_files_in_dir(args.gt_folder, postfix=VALID_EXTS)
    need_resize_gt = False
    for p in gt_files:
        h = Image.open(p).height
        if h != pred_target_height:
            need_resize_gt = True
            break

    fid_gt_folder = args.gt_folder
    if need_resize_gt:
        print(f"[Eval] Resizing GT images to target height = {pred_target_height}")
        fid_gt_folder = copy_resize_folder_to_height(args.gt_folder, pred_target_height)

    # 建 pair：paired 指标用；同时 FID/KID 也尽量在“对应样本集”上测
    pairs, match_mode = build_pairs(fid_gt_folder, args.pred_folder)

    results = {}
    results["FID"] = float(FID.compute_fid(fid_gt_folder, args.pred_folder, num_workers=0))
    results["KID"] = float(FID.compute_kid(fid_gt_folder, args.pred_folder, num_workers=0) * 1000.0)

    if args.paired:
        dataset = EvalDataset(pairs, target_height=pred_target_height)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )
        results["SSIM"] = float(calc_ssim(dataloader, device))
        results["LPIPS"] = float(calc_lpips(dataloader, device))

    print("GT Folder   :", fid_gt_folder)
    print("Pred Folder :", args.pred_folder)
    print("Target H    :", pred_target_height)
    print("Match Mode  :", match_mode)
    print("Mode        :", "paired" if args.paired else "unpaired")

    table = PrettyTable()
    table.field_names = list(results.keys())
    table.add_row([results[k] for k in results.keys()])
    print(table)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    eval_main(args)