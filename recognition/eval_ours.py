#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate generated images using the recognition model (category-level).

核心：
- 你的 hp 没有 hp.category（类别名列表），只有 hp.categories=345
- 因此本脚本从 gen_root 下的子目录自动收集类别名
- 并自动从 QuickDraw414k 的 list 文件（tiny_train/test/val_set.txt 等）推断 "类别名 -> label id" 映射
- 若推断失败，可用 --categories_txt 指定 345 类的类别名列表（每行一个类别名，顺序=label id）

用法（推荐先这样跑）：
  python eval_ours.py \
    --gen_root /data/user/jackyan/CS3308/sketch_code/generation/results \
    --ckpt ./pretrain/QD414k.pkl \
    --device cuda:0 \
    --data_root ./Data \
    --use_branch img \
    --batch_size 64 \
    --out_cm confusion_matrix_gen.npy \
    --out_csv per_class_acc_gen.csv
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from Networks5 import net
from Hyper_params import hp

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ----------------------------
# name normalization utilities
# ----------------------------
def norm_name(x: str) -> str:
    """
    把各种目录名/类别名规范化，尽量对齐：
    - 去掉末尾 .npz
    - '_' -> ' '
    - 小写
    - 多空格压成单空格
    """
    x = x.strip()
    if x.lower().endswith(".npz"):
        x = x[:-4]
    x = x.replace("_", " ")
    x = x.lower().strip()
    x = " ".join(x.split())
    return x


def list_subdirs(root: str) -> List[str]:
    ds = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            ds.append(name)
    ds.sort()
    return ds


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files


# --------------------------------------------------------
# build mapping: category_name(normalized) -> label_id (int)
# --------------------------------------------------------
def find_qd414k_list_files(data_root: str) -> List[str]:
    """
    在 data_root 下搜索可能的 QuickDraw414k list 文件：
    - tiny_train_set.txt / tiny_test_set.txt / tiny_val_set.txt
    - 或者其他 *_set.txt
    """
    hits = []
    for dirpath, _dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".txt") and ("set" in low) and ("train" in low or "test" in low or "val" in low):
                hits.append(os.path.join(dirpath, fn))
    hits.sort()
    return hits


def build_map_from_list_files(list_files: List[str]) -> Dict[str, int]:
    """
    解析 list 文件行：通常形式为
      <relative_path> <label>
    其中 relative_path 往往包含类别名作为第一个路径段： e.g. "airplane/xxx.png 0"
    返回：norm(category_name) -> label_id
    """
    mapping: Dict[str, int] = {}
    conflicts = 0

    for lf in list_files:
        try:
            with open(lf, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            # 有些可能是 latin1
            with open(lf, "r", encoding="latin1") as f:
                lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            try:
                label = int(parts[-1])
            except Exception:
                continue

            # rel_path 可能是 "airplane/xxx.png" 或 "airplane_xxx.png" 等
            # 优先用路径第一个段当类别名
            cat = rel_path.split("/")[0]
            cat_n = norm_name(cat)

            if cat_n in mapping and mapping[cat_n] != label:
                conflicts += 1
                # 保留第一次出现的
            else:
                mapping[cat_n] = label

    # 如果冲突很多，说明这个猜法可能不对，但一般 QuickDraw414k 是对的
    if conflicts > 0:
        print(f"[Warn] conflicts while building map from list files: {conflicts} (kept first occurrence)")
    return mapping


def build_map_from_categories_txt(categories_txt: str) -> Dict[str, int]:
    """
    如果你有 345 类名字列表（每行一个类别名，顺序=label id），可用这个构建映射
    """
    mapping: Dict[str, int] = {}
    with open(categories_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for i, name in enumerate(lines):
        mapping[norm_name(name)] = i
    return mapping


def build_category_to_label(data_root: str, categories_txt: Optional[str]) -> Dict[str, int]:
    """
    优先：
    1) 如果给了 categories_txt，用它（最稳）
    2) 否则从 data_root 下自动找 list files 推断
    """
    if categories_txt is not None:
        print(f"[Info] Build mapping from categories_txt: {categories_txt}")
        return build_map_from_categories_txt(categories_txt)

    list_files = find_qd414k_list_files(data_root)
    if len(list_files) == 0:
        print(f"[Warn] No list txt found under {data_root}.")
        return {}

    print("[Info] Found list files for mapping:")
    for lf in list_files[:10]:
        print("  -", lf)
    if len(list_files) > 10:
        print(f"  ... (+{len(list_files)-10} more)")

    mapping = build_map_from_list_files(list_files)
    print(f"[Info] Built mapping size={len(mapping)} from list files.")
    return mapping


# ----------------------------
# dataset for generated images
# ----------------------------
class GenImageDataset(Dataset):
    def __init__(self, gen_root: str, folder_names: List[str], folder_to_label: Dict[str, int], tfm):
        """
        folder_names: gen_root 下的子目录名列表（原始名字）
        folder_to_label: norm(folder_name) -> label_id(0..344)
        """
        self.samples: List[Tuple[str, int, str]] = []
        self.tfm = tfm

        missed = 0
        for folder in folder_names:
            folder_path = os.path.join(gen_root, folder)
            nfolder = norm_name(folder)

            if nfolder not in folder_to_label:
                missed += 1
                continue

            label_id = folder_to_label[nfolder]
            imgs = list_images(folder_path)
            for fn in imgs:
                self.samples.append((os.path.join(folder_path, fn), label_id, folder))

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid samples. Usually means: your folder names can't be mapped to label ids.\n"
                "Try providing --categories_txt (345 class names list) or ensure Data list files exist under --data_root."
            )

        if missed > 0:
            print(f"[Warn] {missed} folders could not be mapped to label id and were skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_id, folder = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return img, label_id, path, folder


# ----------------------------
# evaluation
# ----------------------------
@torch.no_grad()
def evaluate(
    model,
    loader,
    device: torch.device,
    num_classes: int,
    seq_len: int,
    sf_num: int,
    use_branch: str = "img",
):
    correct1 = 0
    correct5 = 0
    total = 0

    all_preds = []
    all_labels = []

    for imgs, labels, _paths, _folders in tqdm(loader, desc="Eval", leave=True):
        imgs = imgs.to(device)
        labels = labels.to(device)
        B = imgs.size(0)

        dummy_seq = torch.zeros((B, seq_len, sf_num), dtype=torch.float32, device=device)

        predicted, img_ls, seq_ls, _ = model(imgs, dummy_seq)

        if use_branch == "img":
            logits = img_ls
        elif use_branch == "fusion":
            logits = predicted
        elif use_branch == "seq":
            logits = seq_ls
        else:
            raise ValueError("use_branch must be one of: img/fusion/seq")

        pred1 = logits.argmax(dim=1)
        correct1 += (pred1 == labels).sum().item()

        k = 5 if num_classes >= 5 else num_classes
        topk = logits.topk(k, dim=1).indices
        correct5 += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += B
        all_preds.append(pred1.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc1 = correct1 / max(total, 1)
    acc5 = correct5 / max(total, 1)

    # confusion matrix: full 345x345（可能大，但最标准）
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[int(t), int(p)] += 1

    # per-class acc
    per_class_acc = np.zeros((num_classes,), dtype=np.float64)
    for c in range(num_classes):
        denom = cm[c].sum()
        per_class_acc[c] = (cm[c, c] / denom) if denom > 0 else 0.0

    return acc1, acc5, cm, per_class_acc, total

def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 1) 取出真正的 state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "net" in ckpt:
            state = ckpt["net"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            # 可能本身就是一个 state_dict
            state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint format not supported, got type={type(state)}")

    # 2) 去掉 DataParallel 前缀 module.
    if len(state) > 0:
        first_k = next(iter(state.keys()))
        if first_k.startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}

    # 3) 只加载“模型里确实存在且 shape 一致”的权重（避免多余 key / shape 不匹配）
    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    shape_mismatch = 0
    for k, v in state.items():
        if k not in model_sd:
            dropped += 1
            continue
        if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape != model_sd[k].shape:
            shape_mismatch += 1
            continue
        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)
    # incompatible 是 IncompatibleKeys(missing_keys, unexpected_keys)
    print(f"[CKPT] loaded={len(filtered)} dropped={dropped} shape_mismatch={shape_mismatch}")
    if len(incompatible.missing_keys) > 0:
        print(f"[CKPT] missing_keys({len(incompatible.missing_keys)}): {incompatible.missing_keys[:20]} ...")
    if len(incompatible.unexpected_keys) > 0:
        print(f"[CKPT] unexpected_keys({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:20]} ...")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", required=True, help="生成图像根目录（子目录=类别）")
    parser.add_argument("--ckpt", required=True, help="recognition 权重 .pkl")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu（默认自动）")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_branch", default="img", choices=["img", "fusion", "seq"])
    parser.add_argument("--data_root", default="./Data", help="用于自动推断类别->label映射的 Data 根目录")
    parser.add_argument("--categories_txt", default=None,
                        help="可选：345类类别名列表txt（每行一个类别名，行号=label id），提供这个最稳")
    parser.add_argument("--out_cm", default="confusion_matrix_gen.npy")
    parser.add_argument("--out_csv", default="per_class_acc_gen.csv")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 自动收集 gen_root 下的类别文件夹
    folders = list_subdirs(args.gen_root)
    if len(folders) == 0:
        raise RuntimeError(f"No subdirectories found under gen_root={args.gen_root}")
    print(f"[Info] Found {len(folders)} folders under gen_root")

    # 2) 建立 folder_name -> label_id 映射
    cat_to_label = build_category_to_label(args.data_root, args.categories_txt)
    if len(cat_to_label) == 0:
        raise RuntimeError(
            "Failed to build category->label mapping.\n"
            "Fix options:\n"
            "  (1) ensure QuickDraw414k list files exist under --data_root (e.g., Data/QuickDraw414k/...tiny_*_set.txt)\n"
            "  (2) provide --categories_txt with 345 class names list.\n"
        )

    # 把 gen_root 的 folder 名规范化后查 label
    folder_to_label = {}
    for fd in folders:
        nfd = norm_name(fd)
        if nfd in cat_to_label:
            folder_to_label[nfd] = cat_to_label[nfd]

    print(f"[Info] Mapped folders = {len(folder_to_label)}/{len(folders)}")
    if len(folder_to_label) == 0:
        # 打印几个例子帮助定位
        print("[Debug] Example normalized folder names:")
        for fd in folders[:10]:
            print("  -", fd, "->", norm_name(fd))
        print("[Debug] Example keys in mapping:")
        for k in list(cat_to_label.keys())[:10]:
            print("  -", k, "->", cat_to_label[k])
        raise RuntimeError("None of the folder names can be mapped to label ids.")

    # 3) transform（与你 recognition dataset 一致：ToTensor + Normalize(0.5)）
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 4) dataset/loader
    ds = GenImageDataset(args.gen_root, folders, folder_to_label, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # 5) model
    model = net().to(device)
    model.eval()
    print(f"[Info] Loading ckpt: {args.ckpt}")
    load_ckpt(model, args.ckpt)

    print(f"[Info] device={device}, use_branch={args.use_branch}")
    print(f"[Info] total_samples={len(ds)}, num_classes={hp.categories}, seq_len={hp.seq_len}, sf_num={hp.sf_num}")

    acc1, acc5, cm, per_class_acc, total = evaluate(
        model, dl, device,
        num_classes=hp.categories,
        seq_len=hp.seq_len,
        sf_num=hp.sf_num,
        use_branch=args.use_branch
    )

    print(f"\n[Result] TOTAL={total}")
    print(f"[Result] Top1 Acc = {acc1*100:.2f}%")
    print(f"[Result] Top5 Acc = {acc5*100:.2f}%")

    np.save(args.out_cm, cm)
    print(f"[Save] confusion matrix (345x345) -> {args.out_cm}")

    # per-class acc 只对“你生成里出现的那几个 label”更有意义，所以这里额外输出“出现过的类”
    # 统计哪些 label 在真值里出现
    present = np.where(cm.sum(axis=1) > 0)[0].tolist()

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "n_samples", "acc"])
        for lid in present:
            n = int(cm[lid].sum())
            w.writerow([lid, n, float(per_class_acc[lid])])
        w.writerow(["__OVERALL__", total, float(acc1)])
    print(f"[Save] per-class csv (only present labels) -> {args.out_csv}")

    print("\n[Per-class (present labels)]")
    for lid in present[:50]:
        print(f"label {lid:3d}: n={int(cm[lid].sum())}, acc={per_class_acc[lid]*100:.2f}%")
    if len(present) > 50:
        print(f"... (+{len(present)-50} more)")

    print("\nDone.")


if __name__ == "__main__":
    main()
