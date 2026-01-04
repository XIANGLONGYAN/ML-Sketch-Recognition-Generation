#!/usr/bin/env python3
"""
Per-class CLIP score (img-img): generated images vs real images.

- Assumes both real_path and fake_path contain per-category subfolders.
- For each category, it pairs images by sorted filename order (as in your original script).
  If counts differ, it uses the first min(N_real, N_fake) pairs and prints a warning.
- Outputs per-class CLIP score and saves a CSV.

Usage example:
  python clip_score_per_class_imgimg.py \
    --clip-model ViT-B/32 \
    --batch-size 64 \
    --device cuda:0 \
    --out_csv clip_per_class.csv \
    /path/to/real_images_root \
    /path/to/generated_images_root
"""

import os
import os.path as osp
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip

# If your project has hyper_params.py with hp.category, keep it.
# Otherwise, the script will fallback to scanning subfolders.
try:
    import sys
    sys.path.append("../")
    from hyper_params import hp  # type: ignore
except Exception:
    hp = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):  # type: ignore
        return x


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def is_image_file(p: str) -> bool:
    return osp.splitext(p)[1].lower() in IMAGE_EXTENSIONS


def list_images(folder: str) -> List[str]:
    if not osp.isdir(folder):
        return []
    files = [osp.join(folder, f) for f in os.listdir(folder)]
    files = [f for f in files if osp.isfile(f) and is_image_file(f)]
    return sorted(files)


def resolve_cat_dir(root: str, cat: str) -> Optional[str]:
    """
    Resolve category folder path under root with a few heuristics:
    - exact cat
    - cat without ".npz"
    - replace spaces with underscores
    - try quoting differences are irrelevant to filesystem calls
    """
    candidates = []
    candidates.append(cat)
    if cat.endswith(".npz"):
        candidates.append(cat[:-4])
    # also try adding/removing .npz
    if not cat.endswith(".npz"):
        candidates.append(cat + ".npz")

    # try underscore variants
    for c in list(candidates):
        candidates.append(c.replace(" ", "_"))
        candidates.append(c.replace("_", " "))

    # unique preserve order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        p = osp.join(root, c)
        if osp.isdir(p):
            return p
    return None


def get_categories(real_root: str, fake_root: str) -> List[str]:
    """
    Priority:
    1) hp.category if available
    2) intersection of subfolders under real_root and fake_root
    """
    if hp is not None and hasattr(hp, "category"):
        return list(hp.category)

    real_cats = {d for d in os.listdir(real_root) if osp.isdir(osp.join(real_root, d))}
    fake_cats = {d for d in os.listdir(fake_root) if osp.isdir(osp.join(fake_root, d))}
    cats = sorted(list(real_cats & fake_cats))
    return cats


class ImgImgPerClassDataset(Dataset):
    """
    Pairs images from (real_root/cat) and (fake_root/cat) by sorted order.
    Returns dict(real=img_tensor, fake=img_tensor)
    """
    def __init__(
        self,
        real_root: str,
        fake_root: str,
        cat: str,
        transform=None,
    ) -> None:
        super().__init__()
        self.real_dir = resolve_cat_dir(real_root, cat)
        self.fake_dir = resolve_cat_dir(fake_root, cat)
        self.cat = cat
        self.transform = transform

        if self.real_dir is None:
            raise FileNotFoundError(f"Cannot find real category folder for '{cat}' under {real_root}")
        if self.fake_dir is None:
            raise FileNotFoundError(f"Cannot find fake category folder for '{cat}' under {fake_root}")

        self.real_files = list_images(self.real_dir)
        self.fake_files = list_images(self.fake_dir)

        if len(self.real_files) == 0:
            raise RuntimeError(f"No real images found for '{cat}' in {self.real_dir}")
        if len(self.fake_files) == 0:
            raise RuntimeError(f"No fake images found for '{cat}' in {self.fake_dir}")

        self.n = min(len(self.real_files), len(self.fake_files))
        if len(self.real_files) != len(self.fake_files):
            print(
                f"[WARN] '{cat}': real={len(self.real_files)} fake={len(self.fake_files)}. "
                f"Using first min={self.n} pairs by sorted order."
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rpath = self.real_files[idx]
        fpath = self.fake_files[idx]

        rimg = Image.open(rpath).convert("RGB")
        fimg = Image.open(fpath).convert("RGB")

        if self.transform is not None:
            rimg = self.transform(rimg)
            fimg = self.transform(fimg)

        return {"real": rimg, "fake": fimg}


@torch.no_grad()
def forward_image(model, images: torch.Tensor) -> torch.Tensor:
    features = model.encode_image(images)
    return features


@torch.no_grad()
def calculate_clip_score_imgimg(dataloader: DataLoader, model) -> float:
    """
    Mean over samples of: logit_scale * cosine_similarity(real, fake)
    where cosine_similarity is computed in CLIP embedding space.
    """
    device = next(model.parameters()).device
    logit_scale = model.logit_scale.exp()

    score_sum = 0.0
    n = 0

    for batch in tqdm(dataloader):
        real = batch["real"].to(device)
        fake = batch["fake"].to(device)

        real_feat = forward_image(model, real)
        fake_feat = forward_image(model, fake)

        # normalize (cosine similarity)
        real_feat = real_feat / real_feat.norm(dim=1, keepdim=True).to(torch.float32)
        fake_feat = fake_feat / fake_feat.norm(dim=1, keepdim=True).to(torch.float32)

        # per-sample cosine similarity
        cos = (real_feat * fake_feat).sum(dim=1)  # [B]
        score = logit_scale * cos  # [B]

        score_sum += score.sum().item()
        n += score.shape[0]

    return score_sum / max(n, 1)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32", help="CLIP model to use")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers. Default min(8, num_cpus)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. Like cuda, cuda:0 or cpu")
    parser.add_argument("--out_csv", type=str, default="clip_per_class.csv",
                        help="Where to save per-class results (CSV)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Optional cap on number of pairs per class (after sorting).")
    parser.add_argument("real_path", type=str, help="Root folder of REAL images (per-class subfolders)")
    parser.add_argument("fake_path", type=str, help="Root folder of FAKE images (per-class subfolders)")
    args = parser.parse_args()

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # num_workers
    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count() or 0
        num_workers = min(num_cpus, 8) if num_cpus else 0
    else:
        num_workers = args.num_workers

    print(f"Loading CLIP model: {args.clip_model} on {device}")
    model, preprocess = clip.load(args.clip_model, device=device)
    model.eval()

    cats = get_categories(args.real_path, args.fake_path)
    if len(cats) == 0:
        raise RuntimeError("No categories found. Check your folder structure or hp.category.")

    results: List[Tuple[str, int, float]] = []
    total_score_sum = 0.0
    total_n = 0

    for cat in cats:
        try:
            ds = ImgImgPerClassDataset(args.real_path, args.fake_path, cat, transform=preprocess)
        except Exception as e:
            print(f"[SKIP] {cat}: {e}")
            continue

        # optional cap
        if args.max_per_class is not None and len(ds) > args.max_per_class:
            # truncate by slicing file lists (simple & deterministic)
            ds.real_files = ds.real_files[:args.max_per_class]
            ds.fake_files = ds.fake_files[:args.max_per_class]
            ds.n = args.max_per_class

        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

        score = calculate_clip_score_imgimg(dl, model)
        n = len(ds)
        print(f"[{cat}] N={n}  CLIP(img-img)={score:.4f}")

        results.append((cat, n, score))
        total_score_sum += score * n
        total_n += n

    # summary
    if total_n > 0:
        weighted_mean = total_score_sum / total_n
        mean_of_means = sum(s for _, _, s in results) / max(len(results), 1)
        print(f"\nOverall (weighted by N): {weighted_mean:.4f}")
        print(f"Overall (mean of class means): {mean_of_means:.4f}")
    else:
        print("\nNo classes evaluated successfully.")

    # save CSV
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("category,n,clip_img_img\n")
        for cat, n, s in results:
            f.write(f"{cat},{n},{s:.6f}\n")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
