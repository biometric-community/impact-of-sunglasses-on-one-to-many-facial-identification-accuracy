"""Verification (clean vs sunglasses probe) + closed-set 1:N Rank-1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sungfr.augment.sunglasses import apply_sunglasses
from sungfr.data.celeba import CelebAIdDataset, resolve_celeba_root
from sungfr.data.pairs import build_calfw, build_cplfw, build_lfw
from sungfr.models.fr_net import FaceEmbedder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc, best_eer = 0.0, 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
        pos, neg = labels == 1, labels == 0
        if pos.any() and neg.any():
            fpr = float((pred[neg] == 1).mean())
            fnr = float((pred[pos] == 0).mean())
            best_eer = min(best_eer, 0.5 * (fpr + fnr))
    return best_acc, best_eer


@torch.no_grad()
def eval_pairs(
    model,
    ds,
    device,
    batch_size: int,
    num_workers: int,
    tag: str,
    sunglass_on: str = "none",
) -> Dict:
    """sunglass_on: none | probe (second image) | both | gallery (first)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc=tag, leave=False):
        a = a.to(device)
        b = b.to(device)
        if sunglass_on in ("gallery", "both"):
            a = apply_sunglasses(a)
        if sunglass_on in ("probe", "both"):
            b = apply_sunglasses(b)
        ea = model.embed(a)
        eb = model.embed(b)
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[{tag}] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


@torch.no_grad()
def eval_closed_set_1n(
    model,
    gallery_loader,
    probe_loader,
    device,
    probe_sunglasses: bool,
    tag: str,
) -> Dict:
    """Gallery embeddings clean; probe optionally with synthetic sunglasses. Rank-1."""
    model.eval()
    g_embs, g_labels = [], []
    for images, labels in tqdm(gallery_loader, desc=f"{tag}_gallery", leave=False):
        e = model.embed(images.to(device)).cpu().numpy()
        g_embs.append(e)
        g_labels.append(labels.numpy())
    G = np.concatenate(g_embs)
    gy = np.concatenate(g_labels)

    p_embs, p_labels = [], []
    for images, labels in tqdm(probe_loader, desc=f"{tag}_probe", leave=False):
        images = images.to(device)
        if probe_sunglasses:
            images = apply_sunglasses(images)
        e = model.embed(images).cpu().numpy()
        p_embs.append(e)
        p_labels.append(labels.numpy())
    P = np.concatenate(p_embs)
    py = np.concatenate(p_labels)

    # cosine: embeddings already L2-normalized
    sims = P @ G.T
    top1 = sims.argmax(axis=1)
    rank1 = float((gy[top1] == py).mean())
    # also gallery-sunglasses recovery (paper: add sunglasses to gallery)
    print(f"[{tag}] rank1={rank1:.4f} n_probe={len(py)} n_gallery={len(gy)}")
    return {
        "rank1": rank1,
        "n_probe": int(len(py)),
        "n_gallery": int(len(gy)),
        "probe_sunglasses": probe_sunglasses,
    }


def _load_models(ckpt_path: Path, device) -> Dict[str, FaceEmbedder]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    n = int(blob["num_classes"])
    out = {}
    for key in ("clean", "sunglass_aware"):
        m = FaceEmbedder(n, pretrained=False).to(device)
        m.load_state_dict(blob[key])
        m.eval()
        out[key] = m
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    models = _load_models(ckpt, device)

    image_size = int(cfg["model"].get("image_size", 112))
    bs = int(cfg["eval"].get("batch_size", 128))
    nw = int(cfg["train"].get("num_workers", 4))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None

    builders = {
        "lfw": build_lfw,
        "calfw": build_calfw,
        "cplfw": build_cplfw,
    }
    results: Dict = {"models": {}}

    for mname, model in models.items():
        mres: Dict = {"verification": {}, "identification": {}}
        for dname, builder in builders.items():
            try:
                ds = builder(cfg, image_size, max_p)
            except FileNotFoundError as e:
                print(f"[skip] {dname}: {e}")
                continue
            for mode in ("none", "probe", "both"):
                key = f"{dname}_{mode}"
                mres["verification"][key] = eval_pairs(
                    model, ds, device, bs, nw, f"{mname}/{key}", sunglass_on=mode
                )

        # closed-set 1:N on CelebA val (subjects disjoint from train by split)
        root = resolve_celeba_root(cfg)
        max_n = cfg["data"].get("max_images")
        max_n = int(max_n) if max_n is not None else None
        max_gal = cfg["eval"].get("max_gallery")
        max_gal = int(max_gal) if max_gal is not None else None
        val_ds = CelebAIdDataset(
            root,
            split="val",
            image_size=image_size,
            train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
            max_n=max_n,
            seed=int(cfg["train"].get("seed", 42)),
            train=False,
        )
        # build gallery: first image per identity; probes: remaining
        by_id: Dict[int, List[int]] = {}
        for i in range(len(val_ds)):
            _, sid = val_ds.samples[i]
            by_id.setdefault(sid, []).append(i)
        gal_idx, pr_idx = [], []
        for sid, idxs in by_id.items():
            if len(idxs) < 2:
                continue
            gal_idx.append(idxs[0])
            pr_idx.extend(idxs[1:])
        if max_gal is not None:
            gal_idx = gal_idx[:max_gal]
            keep = {val_ds.samples[i][1] for i in gal_idx}
            pr_idx = [i for i in pr_idx if val_ds.samples[i][1] in keep]
            if max_gal is not None and len(pr_idx) > max_gal * 2:
                pr_idx = pr_idx[: max_gal * 2]

        class _Sub(torch.utils.data.Dataset):
            def __init__(self, base, indices):
                self.base, self.indices = base, indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                return self.base[self.indices[i]]

        g_loader = DataLoader(_Sub(val_ds, gal_idx), batch_size=bs, shuffle=False, num_workers=nw)
        p_loader = DataLoader(_Sub(val_ds, pr_idx), batch_size=bs, shuffle=False, num_workers=nw)

        mres["identification"]["probe_clean"] = eval_closed_set_1n(
            model, g_loader, p_loader, device, False, f"{mname}/id_clean"
        )
        mres["identification"]["probe_sunglasses"] = eval_closed_set_1n(
            model, g_loader, p_loader, device, True, f"{mname}/id_sunglass"
        )
        # gallery sunglasses recovery: re-embed gallery with sunglasses
        mres["identification"]["gallery_sunglasses_probe_sunglasses"] = (
            _eval_gallery_sunglass_recovery(model, g_loader, p_loader, device, f"{mname}/id_recover")
        )
        results["models"][mname] = mres

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


@torch.no_grad()
def _eval_gallery_sunglass_recovery(model, gallery_loader, probe_loader, device, tag) -> Dict:
    model.eval()
    g_embs, g_labels = [], []
    for images, labels in tqdm(gallery_loader, desc=f"{tag}_gallery", leave=False):
        images = apply_sunglasses(images.to(device))
        e = model.embed(images).cpu().numpy()
        g_embs.append(e)
        g_labels.append(labels.numpy())
    G = np.concatenate(g_embs)
    gy = np.concatenate(g_labels)
    p_embs, p_labels = [], []
    for images, labels in tqdm(probe_loader, desc=f"{tag}_probe", leave=False):
        images = apply_sunglasses(images.to(device))
        e = model.embed(images).cpu().numpy()
        p_embs.append(e)
        p_labels.append(labels.numpy())
    P = np.concatenate(p_embs)
    py = np.concatenate(p_labels)
    top1 = (P @ G.T).argmax(axis=1)
    rank1 = float((gy[top1] == py).mean())
    print(f"[{tag}] rank1={rank1:.4f} (gallery+probe sunglasses)")
    return {
        "rank1": rank1,
        "n_probe": int(len(py)),
        "n_gallery": int(len(gy)),
        "probe_sunglasses": True,
        "gallery_sunglasses": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
