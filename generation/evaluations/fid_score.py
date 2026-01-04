#!/usr/bin/env python3
"""Calculates Frechet Inception Distance (FID)

Extended:
- Compute per-class FID and overall FID (all classes merged) in one run.
- Folder mode assumes each class has a subfolder under given roots.
- Keeps compatibility with original:
    python fid_score.py PATH1 PATH2 --gpu 0

Notes:
- Per-class FID is only available when both inputs are directories.
- Overall FID supports directory / .npz / .txt like original.
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from PIL import Image

import sys
sys.path.append("../")
try:
    from hyper_params import hp
except Exception:
    hp = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):  # type: ignore
        return x

from evaluations.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "path",
    type=str,
    nargs=2,
    help=("Path to the generated images or to .npz statistic files"),
)
parser.add_argument("--batch-size", type=int, default=50, help="Batch size to use")
parser.add_argument(
    "--dims",
    type=int,
    default=2048,
    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    help=("Dimensionality of Inception features to use. By default, uses pool3 features"),
)
parser.add_argument("-c", "--gpu", default="", type=str, help="GPU to use (leave blank for CPU only)")
parser.add_argument(
    "--out_csv",
    type=str,
    default="fid_per_class.csv",
    help="Save per-class FID results to CSV (only when folder mode)",
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm", ".pgm"}


def imread(filename: str) -> np.ndarray:
    """Loads an image file into a (H, W, 3) uint8 ndarray."""
    return np.asarray(Image.open(filename).convert("RGB"), dtype=np.uint8)[..., :3]


def list_images_in_dir(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    p = pathlib.Path(folder)
    files = []
    for ext in IMAGE_EXTS:
        files.extend(list(p.glob(f"*{ext}")))
        files.extend(list(p.glob(f"*{ext.upper()}")))
    # sort for determinism
    return [str(x) for x in sorted(files)]


def get_categories_from_hp() -> Optional[List[str]]:
    if hp is None or not hasattr(hp, "category"):
        return None
    return list(hp.category)


def normalize_cat_name(cat: str) -> List[str]:
    """
    Try a few folder naming variants:
    - cat as-is
    - cat without ".npz"
    - cat with ".npz"
    """
    cand = []
    cand.append(cat)
    if cat.endswith(".npz"):
        cand.append(cat[:-4])
    else:
        cand.append(cat + ".npz")
    # unique preserve order
    out = []
    seen = set()
    for c in cand:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_cat_folder(root: str, cat: str) -> Optional[str]:
    """
    Find existing subfolder for this category under root.
    """
    for c in normalize_cat_name(cat):
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return None


def get_categories_by_scanning(real_root: str, fake_root: str) -> List[str]:
    """
    Fallback: use intersection of subfolder names.
    """
    real = {d for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))}
    fake = {d for d in os.listdir(fake_root) if os.path.isdir(os.path.join(fake_root, d))}
    return sorted(list(real & fake))


@torch.no_grad()
def get_activations(
    files: List[str],
    model: torch.nn.Module,
    batch_size: int = 50,
    dims: int = 2048,
    cuda: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Calculates the activations of the pool_3 layer for all images."""
    model.eval()

    if len(files) == 0:
        return np.empty((0, dims), dtype=np.float32)

    if batch_size > len(files):
        if verbose:
            print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims), dtype=np.float32)

    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print(f"\rPropagating batch {i // batch_size + 1}/{(len(files) + batch_size - 1) // batch_size}",
                  end="", flush=True)
        start = i
        end = min(i + batch_size, len(files))

        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]], dtype=np.float32)
        images = images.transpose((0, 3, 1, 2))  # (N, 3, H, W)
        images /= 255.0

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.detach().cpu().numpy().reshape(pred.size(0), -1)

    if verbose:
        print(" done")

    return pred_arr


def calculate_activation_statistics(
    files: List[str],
    model: torch.nn.Module,
    batch_size: int = 50,
    dims: int = 2048,
    cuda: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculation of the statistics used by the FID."""
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    if act.shape[0] == 0:
        raise RuntimeError("No activations: empty file list.")
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        print(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def make_dataset_txt(files: str) -> List[str]:
    img_paths = []
    with open(files, "r") as f:
        paths = f.readlines()
    for path in paths:
        img_paths.append(path.strip())
    return img_paths


def _compute_statistics_of_path(path: str, model, batch_size: int, dims: int, cuda: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original behavior:
    - .npz: load mu/sigma
    - .txt: list file paths
    - dir: gather images from all categories under path (hp.category-driven)
    """
    if path.endswith(".npz"):
        f = np.load(path)
        m, s = f["mu"][:], f["sigma"][:]
        return m, s

    if path.endswith(".txt"):
        files = make_dataset_txt(path)
        m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda)
        return m, s

    # directory (overall): gather all images across categories
    if not os.path.isdir(path):
        raise RuntimeError(f"Invalid path (not dir / npz / txt): {path}")

    files: List[str] = []
    cats = get_categories_from_hp()
    if cats is None:
        # fallback: just recursively collect images
        p = pathlib.Path(path)
        for ext in IMAGE_EXTS:
            files += [str(x) for x in p.rglob(f"*{ext}")]
            files += [str(x) for x in p.rglob(f"*{ext.upper()}")]
        files = sorted(files)
    else:
        for cat in cats:
            cat_dir = resolve_cat_folder(path, cat)
            if cat_dir is None:
                continue
            files.extend(list_images_in_dir(cat_dir))

    if len(files) == 0:
        raise RuntimeError(f"No images found under: {path}")

    m, s = calculate_activation_statistics(files, model, batch_size, dims, cuda)
    return m, s


def calculate_fid_given_paths(paths: List[str], batch_size: int, cuda: bool, dims: int) -> float:
    """Calculates the overall FID of two paths."""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid path: {p}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size, dims, cuda)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size, dims, cuda)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def calculate_fid_per_class(
    fake_root: str,
    real_root: str,
    batch_size: int,
    cuda: bool,
    dims: int,
) -> List[Tuple[str, int, int, float]]:
    """
    Per-class FID:
      FID_cat = FID(Real_cat, Fake_cat)
    Returns list of (cat, n_fake, n_real, fid)
    """
    if not (os.path.isdir(fake_root) and os.path.isdir(real_root)):
        raise RuntimeError("Per-class FID requires both paths to be directories.")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    cats = get_categories_from_hp()
    if cats is None:
        cats = get_categories_by_scanning(real_root, fake_root)

    results: List[Tuple[str, int, int, float]] = []

    for cat in cats:
        real_cat_dir = resolve_cat_folder(real_root, cat)
        fake_cat_dir = resolve_cat_folder(fake_root, cat)

        if real_cat_dir is None or fake_cat_dir is None:
            # skip silently or print warning
            print(f"[SKIP] {cat}: missing folder "
                  f"(real={real_cat_dir is not None}, fake={fake_cat_dir is not None})")
            continue

        real_files = list_images_in_dir(real_cat_dir)
        fake_files = list_images_in_dir(fake_cat_dir)

        if len(real_files) == 0 or len(fake_files) == 0:
            print(f"[SKIP] {cat}: empty images (real={len(real_files)}, fake={len(fake_files)})")
            continue

        m_r, s_r = calculate_activation_statistics(real_files, model, batch_size, dims, cuda)
        m_f, s_f = calculate_activation_statistics(fake_files, model, batch_size, dims, cuda)
        fid = calculate_frechet_distance(m_r, s_r, m_f, s_f)

        results.append((cat, len(fake_files), len(real_files), float(fid)))
        print(f"[{cat}] real={len(real_files)} fake={len(fake_files)}  FID={fid:.4f}")

    return results


def save_per_class_csv(rows: List[Tuple[str, int, int, float]], out_csv: str) -> None:
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("category,n_fake,n_real,fid\n")
        for cat, n_fake, n_real, fid in rows:
            f.write(f"{cat},{n_fake},{n_real},{fid:.6f}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # first compute per-class (only in folder mode)
    path0, path1 = args.path[0], args.path[1]
    cuda_flag = args.gpu != ""

    if os.path.isdir(path0) and os.path.isdir(path1):
        print("\n=== Per-class FID ===")
        per_class = calculate_fid_per_class(
            fake_root=path0,
            real_root=path1,
            batch_size=args.batch_size,
            cuda=cuda_flag,
            dims=args.dims,
        )
        if len(per_class) > 0:
            save_per_class_csv(per_class, args.out_csv)
            # report simple summaries
            macro_mean = sum(x[3] for x in per_class) / len(per_class)
            # weighted by real count (or fake, either is fine; here by min to be conservative)
            total_w = 0
            total_s = 0.0
            for cat, n_fake, n_real, fid in per_class:
                w = min(n_fake, n_real)
                total_w += w
                total_s += fid * w
            weighted_mean = total_s / max(total_w, 1)
            print(f"\nSaved per-class CSV: {args.out_csv}")
            print(f"Per-class macro mean FID: {macro_mean:.4f}")
            print(f"Per-class weighted mean FID (w=min(n_fake,n_real)): {weighted_mean:.4f}")
        else:
            print("No classes evaluated for per-class FID (check folder names / images).")
    else:
        print("\n[INFO] Not folder-folder mode, skip per-class FID.")

    # then compute overall FID (original behavior)
    print("\n=== Overall FID (all classes merged) ===")
    fid_value = calculate_fid_given_paths(args.path, args.batch_size, cuda_flag, args.dims)
    print("FID: ", fid_value)
