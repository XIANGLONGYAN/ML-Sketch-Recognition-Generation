#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-class + overall LPIPS evaluation (image-image).

功能：
- 对每个类别单独计算 LPIPS 平均值
- 同时给出 overall（加权平均，按图片对数量加权）
- 支持：优先按“同文件名”配对；若找不到同名交集，则退化为“排序后按索引配对”
- 可选输出 CSV

用法示例：
  python lpips_per_class.py --path1 ../GroundTruth/ --path2 ../results/ --device cuda --out_csv lpips.csv
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import lpips


def try_load_categories_from_hp() -> List[str]:
    """
    尝试从 ../hyper_params.py 里的 hp.category 读取类别列表。
    如果导入失败，返回空列表。
    """
    try:
        sys.path.append("../")
        from hyper_params import hp  # type: ignore
        cats = list(hp.category)
        return cats
    except Exception:
        return []


def resolve_category_dir(root: str, cat: str) -> str:
    """
    兼容你之前的目录命名：既可能叫 "alarm clock.npz"，也可能叫 "alarm clock"
    """
    cand1 = os.path.join(root, cat)
    if os.path.exists(cand1):
        return cand1
    cat_tmp = cat.split(".")[0]
    cand2 = os.path.join(root, cat_tmp)
    return cand2


def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()
    return files


def load_image_as_tensor(path: str, device: torch.device, normalize: str = "-1_1") -> torch.Tensor:
    """
    读取图片 -> Tensor [1,3,H,W]
    normalize:
      - "-1_1": 映射到 [-1, 1]（LPIPS 官方推荐的输入范围）
      - "0_1" : 保持 [0, 1]
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1], HWC
    arr = np.transpose(arr, (2, 0, 1))               # CHW
    t = torch.from_numpy(arr).unsqueeze(0)           # 1CHW
    if normalize == "-1_1":
        t = t * 2.0 - 1.0
    t = t.to(device=device, dtype=torch.float32)
    return t


def pair_files(path1: str, path2: str) -> Tuple[List[str], List[str], str]:
    """
    返回两组“配对后的文件名列表”（仅文件名，不含路径）以及配对策略说明：
    - 如果两边有同名交集：用同名交集配对
    - 否则：按排序索引配对（取 min(len1, len2)）
    """
    files1 = list_images(path1)
    files2 = list_images(path2)

    set1 = set(files1)
    set2 = set(files2)
    inter = sorted(list(set1 & set2))
    if len(inter) > 0:
        return inter, inter, "match=by_filename_intersection"

    n = min(len(files1), len(files2))
    return files1[:n], files2[:n], "match=by_sorted_index_fallback"


@torch.no_grad()
def compute_lpips_for_category(
    folder1: str,
    folder2: str,
    loss_fn: lpips.LPIPS,
    device: torch.device,
    normalize: str = "-1_1",
    verbose: bool = True,
) -> Tuple[float, int, str]:
    """
    计算单类别 LPIPS 平均值
    返回：(mean_lpips, n_pairs, match_strategy)
    """
    f1_list, f2_list, strategy = pair_files(folder1, folder2)
    n = len(f1_list)
    if n == 0:
        return float("nan"), 0, "no_images_found"

    total = 0.0
    it = range(n)
    if verbose:
        it = tqdm(it, desc=os.path.basename(folder1), leave=False)

    for i in it:
        p1 = os.path.join(folder1, f1_list[i])
        p2 = os.path.join(folder2, f2_list[i])

        img1 = load_image_as_tensor(p1, device, normalize=normalize)
        img2 = load_image_as_tensor(p2, device, normalize=normalize)

        d = loss_fn(img1, img2)  # shape usually [1,1,1,1]
        total += float(d.item())

    return total / n, n, strategy


def main():
    parser = argparse.ArgumentParser(description="Per-class + overall LPIPS (img-img)")
    parser.add_argument("--path1", required=True, help="root dir of set A (e.g., GroundTruth)")
    parser.add_argument("--path2", required=True, help="root dir of set B (e.g., results)")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ... (default: auto)")
    parser.add_argument("--net", default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    parser.add_argument("--normalize", default="-1_1", choices=["-1_1", "0_1"],
                        help="image tensor range: '-1_1' recommended for LPIPS")
    parser.add_argument("--out_csv", default=None, help="optional csv path to save results")
    parser.add_argument("--quiet", action="store_true", help="less tqdm output")
    args = parser.parse_args()

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # categories
    categories = try_load_categories_from_hp()
    if len(categories) == 0:
        # fallback: 自动用两边目录的子文件夹交集作为类别
        sub1 = set([d for d in os.listdir(args.path1) if os.path.isdir(os.path.join(args.path1, d))])
        sub2 = set([d for d in os.listdir(args.path2) if os.path.isdir(os.path.join(args.path2, d))])
        categories = sorted(list(sub1 & sub2))

    if len(categories) == 0:
        raise RuntimeError("找不到类别列表：hp.category 导入失败，且 path1/path2 也没有可用的子目录交集。")

    print(f"[LPIPS] device={device}, net={args.net}, normalize={args.normalize}")
    print(f"[LPIPS] categories={len(categories)}")

    # LPIPS model
    loss_fn = lpips.LPIPS(net=args.net)
    loss_fn = loss_fn.to(device)
    loss_fn.eval()

    rows = []
    overall_sum = 0.0
    overall_n = 0

    for cat in categories:
        folder1 = resolve_category_dir(args.path1, cat)
        folder2 = resolve_category_dir(args.path2, cat)

        mean_lp, n_pairs, strategy = compute_lpips_for_category(
            folder1, folder2, loss_fn, device,
            normalize=args.normalize,
            verbose=(not args.quiet),
        )

        # 记录
        rows.append((cat, n_pairs, mean_lp, strategy))
        if n_pairs > 0 and not np.isnan(mean_lp):
            overall_sum += mean_lp * n_pairs
            overall_n += n_pairs

        # 打印每类
        if n_pairs == 0:
            print(f"{cat}: n_pairs=0 (skip)  [{strategy}]")
        else:
            print(f"{cat}: n_pairs={n_pairs}, lpips={mean_lp:.6f}  [{strategy}]")

    overall = overall_sum / overall_n if overall_n > 0 else float("nan")
    print(f"\nOVERALL: n_pairs={overall_n}, lpips={overall:.6f}")

    # optional csv
    if args.out_csv is not None:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "n_pairs", "lpips", "match_strategy"])
            for cat, n_pairs, mean_lp, strategy in rows:
                w.writerow([cat, n_pairs, mean_lp, strategy])
            w.writerow(["__OVERALL__", overall_n, overall, "weighted_by_n_pairs"])
        print(f"[LPIPS] saved csv -> {args.out_csv}")


if __name__ == "__main__":
    main()
