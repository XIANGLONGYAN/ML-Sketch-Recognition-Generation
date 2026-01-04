#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)


def save_per_image(in_root: str, out_root: str, rgb: bool, overwrite: bool) -> None:
    """
    把每张图片保存为一个 .npy，目录结构保持一致：
      in_root/a/b/xxx.jpg  -> out_root/a/b/xxx.npy
    """
    num_total = 0
    num_saved = 0
    num_skipped = 0
    num_failed = 0

    for dirpath, _dirnames, filenames in os.walk(in_root):
        for fn in sorted(filenames):
            if not is_image(fn):
                continue
            num_total += 1

            in_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(in_path, in_root)
            out_path = os.path.join(out_root, os.path.splitext(rel)[0] + ".npy")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if (not overwrite) and os.path.exists(out_path):
                num_skipped += 1
                continue

            try:
                img = Image.open(in_path)
                img = img.convert("RGB") if rgb else img.convert("L")
                arr = np.array(img)  # RGB: (H,W,3) uint8 / Gray: (H,W) uint8
                np.save(out_path, arr)
                num_saved += 1
            except Exception as e:
                num_failed += 1
                print(f"[Fail] {in_path} -> {out_path} | {repr(e)}")

            if num_total % 500 == 0:
                print(f"[Progress] seen={num_total} saved={num_saved} skipped={num_skipped} failed={num_failed}")

    print(f"\n[Done] total={num_total} saved={num_saved} skipped={num_skipped} failed={num_failed}")
    print(f"[Out]  {out_root}")


def save_per_folder_npz(in_root: str, out_root: str, rgb: bool, overwrite: bool) -> None:
    """
    每个子文件夹打包成一个 npz：
      in_root/class1/*.jpg -> out_root/class1.npz
    里面 key 是图片文件名，value 是 ndarray。
    """
    os.makedirs(out_root, exist_ok=True)

    subdirs = sorted([d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d))])
    if len(subdirs) == 0:
        raise RuntimeError(f"No subfolders found under: {in_root}")

    total_folders = len(subdirs)
    for i, sd in enumerate(subdirs, 1):
        in_dir = os.path.join(in_root, sd)
        out_npz = os.path.join(out_root, f"{sd}.npz")

        if (not overwrite) and os.path.exists(out_npz):
            print(f"[Skip] {out_npz} exists")
            continue

        data = {}
        for fn in sorted(os.listdir(in_dir)):
            if not is_image(fn):
                continue
            in_path = os.path.join(in_dir, fn)
            try:
                img = Image.open(in_path)
                img = img.convert("RGB") if rgb else img.convert("L")
                data[fn] = np.array(img)
            except Exception as e:
                print(f"[Fail] {in_path} | {repr(e)}")

        np.savez_compressed(out_npz, **data)
        print(f"[{i}/{total_folders}] saved {out_npz}  (n_images={len(data)})")

    print(f"\n[Done] packed npz saved under: {out_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", required=True, help="输入根目录：下面是很多子文件夹，每个子文件夹里很多jpg")
    parser.add_argument("--out_root", required=True, help="输出根目录：会自动创建并保持目录结构")
    parser.add_argument("--mode", choices=["per_image_npy", "per_folder_npz"], default="per_image_npy",
                        help="per_image_npy: 每张图一个npy；per_folder_npz: 每个子文件夹一个npz")
    parser.add_argument("--gray", action="store_true", help="转为灰度(L)存储；默认RGB三通道")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    args = parser.parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    rgb = (not args.gray)

    if not os.path.isdir(in_root):
        raise RuntimeError(f"in_root not found: {in_root}")

    if args.mode == "per_image_npy":
        save_per_image(in_root=in_root, out_root=out_root, rgb=rgb, overwrite=args.overwrite)
    else:
        save_per_folder_npz(in_root=in_root, out_root=out_root, rgb=rgb, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
