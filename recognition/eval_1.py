#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from Networks5 import net
from Hyper_params import hp  # 你的 hp（注意：这里没有 hp.category 也没关系）


def _list_classes(gen_root: str, class_list: str = None) -> List[str]:
    """
    返回类别名列表（顺序就是 label id）。
    优先用 --class_list（每行一个类名），否则用 gen_root 下的子目录名排序。
    """
    if class_list is not None and os.path.isfile(class_list):
        with open(class_list, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes

    classes = [d for d in os.listdir(gen_root) if os.path.isdir(os.path.join(gen_root, d))]
    classes.sort()
    return classes


def _is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


class GenFolderDataset(torch.utils.data.Dataset):
    """
    目录结构假设：
      gen_root/
        classA/ xxx.png ...
        classB/ yyy.png ...
    label id = classes 列表中的 index
    """
    def __init__(self, gen_root: str, classes: List[str], img_size: int = 224):
        self.gen_root = gen_root
        self.classes = classes
        self.img_size = img_size

        self.samples: List[Tuple[str, int]] = []
        for label, cname in enumerate(classes):
            cdir = os.path.join(gen_root, cname)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                if _is_image_file(fn):
                    self.samples.append((os.path.join(cdir, fn), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {gen_root}")

    def __len__(self):
        return len(self.samples)

    def _transform(self, img: Image.Image) -> torch.Tensor:
        # 跟很多 sketch 训练管线一致：RGB + resize + [0,1] tensor
        img = img.convert("RGB")
        if img.size[0] != self.img_size or img.size[1] != self.img_size:
            img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        arr = np.transpose(arr, (2, 0, 1))               # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = self._transform(img)
        return {"img": img, "label": label, "path": path}


def _load_pretrained(model: nn.Module, ckpt_path: str):
    """
    兼容：
    - ckpt 是 dict，包含 'net_state_dict'
    - 或 ckpt 直接就是 state_dict
    自动处理 DataParallel 的 module. 前缀
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "net_state_dict" in ckpt:
        state = ckpt["net_state_dict"]
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format type={type(state)}")

    model_is_dp = isinstance(model, nn.DataParallel)
    state_has_module = any(k.startswith("module.") for k in state.keys())

    # 让 key 前缀和当前 model 对齐
    if model_is_dp and not state_has_module:
        state = {("module." + k): v for k, v in state.items()}
    if (not model_is_dp) and state_has_module:
        state = {k[len("module."):]: v for k, v in state.items()}

    incompatible = model.load_state_dict(state, strict=False)
    print(f"[CKPT] loaded from {ckpt_path}")
    if len(incompatible.missing_keys) > 0:
        print(f"[CKPT] missing_keys({len(incompatible.missing_keys)}): {incompatible.missing_keys[:20]} ...")
    if len(incompatible.unexpected_keys) > 0:
        print(f"[CKPT] unexpected_keys({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:20]} ...")


@torch.no_grad()
def eval_generated_images(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    use_logits: str = "img",
    dummy_seq_len: int = 100,
    dummy_seq_dim: int = 5,
):
    """
    use_logits:
      - "img"  : 用 img_logsoftmax 做预测（推荐，代表 image branch）
      - "mix"  : 用 predicted 做预测（融合输出，可能受 dummy seq 影响较大）
      - "seq"  : 用 seq_logsoftmax 做预测（一般没意义）
    """
    model.eval()

    all_preds = []
    all_labels = []

    top1_correct = 0
    top5_correct = 0
    total = 0

    for batch in tqdm(loader, desc="Eval"):
        imgs = batch["img"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        B = imgs.shape[0]
        # dummy seq：只为满足 forward 的接口
        seqs = torch.zeros((B, dummy_seq_len, dummy_seq_dim), dtype=torch.float32, device=device)

        predicted, img_logsoftmax, seq_logsoftmax, _ = model(imgs, seqs)

        if use_logits == "img":
            logits = img_logsoftmax
        elif use_logits == "seq":
            logits = seq_logsoftmax
        else:
            logits = predicted  # mix

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        # top1 / top5
        total += B
        top1_correct += (preds == labels).sum().item()

        k = 5 if num_classes >= 5 else num_classes
        topk = torch.topk(logits, k=k, dim=1).indices
        top5_correct += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # confusion matrix (numpy 实现，避免 sklearn 依赖不一致)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[int(t), int(p)] += 1

    per_class_total = cm.sum(axis=1)
    per_class_correct = np.diag(cm)
    per_class_acc = np.divide(
        per_class_correct,
        np.maximum(per_class_total, 1),
        dtype=np.float64
    )

    top1 = top1_correct / max(total, 1)
    top5 = top5_correct / max(total, 1)
    return cm, per_class_acc, top1, top5, total


def save_per_class_csv(path: str, classes: List[str], per_class_acc: np.ndarray, cm: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "label_id", "n", "correct", "acc"])
        for i, cname in enumerate(classes):
            n = int(cm[i].sum())
            correct = int(cm[i, i])
            acc = float(per_class_acc[i])
            w.writerow([cname, i, n, correct, f"{acc:.6f}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", required=True, type=str, help="generated images root (class folders inside)")
    parser.add_argument("--ckpt", required=True, type=str, help="pretrained checkpoint path (e.g., ./pretrain/QD414k.pkl)")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--use_logits", default="img", choices=["img", "mix", "seq"],
                        help="which logits to use for prediction (recommend: img)")
    parser.add_argument("--out_cm", default="confusion_matrix_gen.npy", type=str)
    parser.add_argument("--out_csv", default="per_class_acc_gen.csv", type=str)
    parser.add_argument("--class_list", default=None, type=str,
                        help="optional txt file, one class name per line, order defines label id")
    # dummy seq 的形状（默认 QuickDraw414k 常见是 seq_len=100，每点5维）
    parser.add_argument("--dummy_seq_len", default=100, type=int)
    parser.add_argument("--dummy_seq_dim", default=5, type=int)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # 类别列表
    classes = _list_classes(args.gen_root, args.class_list)
    num_classes = len(classes)
    print(f"[DATA] classes={num_classes}, first10={classes[:10]}")
    print(f"[DATA] gen_root={args.gen_root}")

    # dataset / loader
    ds = GenFolderDataset(args.gen_root, classes, img_size=args.img_size)
    print(f"[DATA] num_images={len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in str(device)),
    )

    # model
    model = net()

    # 你原脚本里是按 hp.gpus 自动 DataParallel，这里保留一个简单版本：
    # 如果你想多卡：CUDA_VISIBLE_DEVICES=0,1 python eval_gen_images.py ...
    if "cuda" in str(device):
        model = model.to(device)

    # 如果检测到多张可见 GPU，就 DataParallel
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and "cuda" in str(device):
        model = nn.DataParallel(model)

    _load_pretrained(model, args.ckpt)

    params_total = sum(p.numel() for p in model.parameters())
    print("Number of parameter: %.2fM" % (params_total / 1e6))

    # eval
    cm, per_class_acc, top1, top5, total = eval_generated_images(
        model=model,
        loader=loader,
        device=device,
        num_classes=num_classes,
        use_logits=args.use_logits,
        dummy_seq_len=args.dummy_seq_len,
        dummy_seq_dim=args.dummy_seq_dim,
    )

    # save
    np.save(args.out_cm, cm)
    save_per_class_csv(args.out_csv, classes, per_class_acc, cm)

    print(f"[RESULT] total={total}  top1={top1:.6f}  top5={top5:.6f}")
    print(f"[SAVE] cm -> {args.out_cm}")
    print(f"[SAVE] per-class -> {args.out_csv}")

    # 打印每类准确率（可选：太多类会刷屏）
    print("Per-class acc (first 20):")
    for i in range(min(20, num_classes)):
        print(f"  {i:3d} {classes[i]:20s} acc={per_class_acc[i]:.4f} n={int(cm[i].sum())}")


if __name__ == "__main__":
    main()
