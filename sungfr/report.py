"""Figures + REPORT.md from real train/eval logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sungfr.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(train_s: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, key in enumerate(("clean_history", "aware_history")):
        hist = train_s.get(key, [])
        if not hist:
            continue
        ax.plot(
            [h["epoch"] for h in hist],
            [h["loss"] for h in hist],
            color=MULTI_SERIES_COLORS[i % len(MULTI_SERIES_COLORS)],
            linewidth=LINEWIDTH_MAIN,
            label=key.replace("_history", ""),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CelebA identity training")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_verification(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    models = eval_res.get("models", {})
    # group: LFW clean vs probe sunglasses for each model
    labels, vals, colors = [], [], []
    color_i = 0
    for mname, mres in models.items():
        ver = mres.get("verification", {})
        for mode, lab in (("none", "clean"), ("probe", "sunglass probe")):
            key = f"lfw_{mode}"
            if key in ver:
                labels.append(f"{mname}\n{lab}")
                vals.append(ver[key]["accuracy"] * 100)
                colors.append(MULTI_SERIES_COLORS[color_i % len(MULTI_SERIES_COLORS)])
                color_i += 1
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LFW accuracy (%)")
    ax.set_title("Verification: clean vs sunglasses probe")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_identification(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    models = eval_res.get("models", {})
    width = 0.25
    conditions = [
        ("probe_clean", "probe clean"),
        ("probe_sunglasses", "probe sunglasses"),
        ("gallery_sunglasses_probe_sunglasses", "gallery+probe sunglasses"),
    ]
    mnames = list(models.keys())
    x = np.arange(len(mnames))
    for i, (ckey, clab) in enumerate(conditions):
        vals = []
        for m in mnames:
            ident = models[m].get("identification", {})
            vals.append(ident.get(ckey, {}).get("rank1", 0.0) * 100)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=clab,
            color=MULTI_SERIES_COLORS[i % len(MULTI_SERIES_COLORS)],
            edgecolor="#002080",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(mnames)
    ax.set_ylabel("Rank-1 (%)")
    ax.set_title("Closed-set 1:N (CelebA val)")
    ax.legend()
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Impact of Sunglasses on One-to-Many Facial Identification Accuracy",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin of Tian et al. (TBIOM): measure degradation from synthetic sunglasses on",
        "probes and recovery via sunglasses-aware training / gallery sunglasses.",
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        f"![Training loss]({fig_dir.as_posix()}/train_loss.svg)",
        "",
        "## Verification (LFW / CALFW / CPLFW)",
        "",
        "| Model | Protocol | Accuracy | EER | N |",
        "|-------|----------|----------|-----|---|",
    ]
    for mname, mres in eval_res.get("models", {}).items():
        for key, v in sorted(mres.get("verification", {}).items()):
            lines.append(
                f"| {mname} | {key} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | {v['n_pairs']} |"
            )
    lines += [
        "",
        f"![LFW verification]({fig_dir.as_posix()}/verification_lfw.svg)",
        "",
        "## Closed-set identification (CelebA val)",
        "",
        "| Model | Condition | Rank-1 | N probe | N gallery |",
        "|-------|-----------|--------|---------|-----------|",
    ]
    for mname, mres in eval_res.get("models", {}).items():
        for key, v in mres.get("identification", {}).items():
            lines.append(
                f"| {mname} | {key} | {v['rank1']*100:.2f}% | {v['n_probe']} | {v['n_gallery']} |"
            )
    lines += [
        "",
        f"![Identification Rank-1]({fig_dir.as_posix()}/identification_rank1.svg)",
        "",
        "## Notes",
        "",
        "- Paper uses WebFace / commercial matchers and ND-MFAD; this rebuild trains ResNet-18 on CelebA",
        "  and evaluates LFW/CALFW/CPLFW + CelebA closed-set 1:N with synthetic eye-region sunglasses.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_train(train_s, fig_dir / "train_loss.svg")
    plot_verification(eval_res, fig_dir / "verification_lfw.svg")
    plot_identification(eval_res, fig_dir / "identification_rank1.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
