#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import uuid
import argparse
from typing import List, Tuple


def target_indices() -> List[int]:
    idx = [2, 20]
    idx += list(range(199, 208))      # 199-207
    idx += list(range(1989, 2078))    # 1989-2077
    assert len(idx) == 100
    return idx


_num_re = re.compile(r"(\d+)")


def sort_key(filename: str) -> Tuple[int, str]:
    """
    尽量按文件名中的数字排序；没有数字则按字典序。
    返回 (num_or_big, filename) 确保稳定可复现。
    """
    m = _num_re.search(filename)
    if m:
        return (int(m.group(1)), filename)
    return (10**18, filename)


def list_npy_files(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".npy")]
    files.sort(key=sort_key)
    return files


def safe_rename_plan(
    folder_path: str,
    folder_name: str,
    files: List[str],
    indices: List[int],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    返回：
    - renames: [(src_path, dst_path), ...] 仅针对要保留的那些
    - deletes: [path_to_delete, ...] 多余文件
    """
    keep_n = min(len(files), len(indices))
    keep_files = files[:keep_n]
    extra_files = files[keep_n:]

    renames = []
    for i, fn in enumerate(keep_files):
        idx = indices[i]
        src = os.path.join(folder_path, fn)
        dst = os.path.join(folder_path, f"{folder_name}_{idx}.npy")
        renames.append((src, dst))

    deletes = [os.path.join(folder_path, fn) for fn in extra_files]
    return renames, deletes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="test-ours 根目录（下面一层是类别文件夹）")
    ap.add_argument("--apply", action="store_true", help="真正执行（默认只打印 dry-run）")
    ap.add_argument("--force", action="store_true", help="若目标文件已存在则覆盖（会先删除目标文件）")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise RuntimeError(f"Not a directory: {root}")

    indices = target_indices()

    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    subdirs.sort()

    total_renamed = 0
    total_deleted = 0

    for d in subdirs:
        folder_path = os.path.join(root, d)
        files = list_npy_files(folder_path)

        if len(files) == 0:
            print(f"[Skip] {d}: no .npy files")
            continue

        renames, deletes = safe_rename_plan(folder_path, d, files, indices)

        if len(files) < len(indices):
            print(f"[Warn] {d}: only {len(files)} files (<100). Will rename {len(renames)} and delete 0.")

        print(f"\n=== {d} ===")
        print(f"Found {len(files)} npy; will rename {len(renames)} keep, delete {len(deletes)} extra")

        # ---- dry-run 打印 ----
        for src, dst in renames[:5]:
            print(f"  RENAME: {os.path.basename(src)} -> {os.path.basename(dst)}")
        if len(renames) > 5:
            print(f"  ... ({len(renames)-5} more renames)")
        if len(deletes) > 0:
            print(f"  DELETE: {len(deletes)} files (extras)")

        if not args.apply:
            continue

        # ---- 真正执行：两阶段重命名避免冲突 ----
        # 1) 先把要保留的文件全部改成临时名
        tmp_tag = uuid.uuid4().hex[:8]
        tmp_paths = []
        for k, (src, _dst) in enumerate(renames):
            tmp = os.path.join(folder_path, f"__tmp__{tmp_tag}_{k}.npy")
            os.rename(src, tmp)
            tmp_paths.append(tmp)

        # 2) 再从临时名改成最终名；如目标已存在，按 --force 处理
        for tmp, (_src, dst) in zip(tmp_paths, renames):
            if os.path.exists(dst):
                if args.force:
                    os.remove(dst)
                else:
                    raise RuntimeError(
                        f"Target exists: {dst}\n"
                        f"Use --force to overwrite."
                    )
            os.rename(tmp, dst)
            total_renamed += 1

        # 3) 删除多余文件
        for p in deletes:
            if os.path.exists(p):
                os.remove(p)
                total_deleted += 1

    print("\n==== Summary ====")
    if args.apply:
        print(f"Renamed: {total_renamed}")
        print(f"Deleted: {total_deleted}")
    else:
        print("Dry-run only. Use --apply to actually rename/delete.")


if __name__ == "__main__":
    main()
