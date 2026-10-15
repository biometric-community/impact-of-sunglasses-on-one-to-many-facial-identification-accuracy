"""Embed a single image (optional synthetic sunglasses)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sungfr.augment.sunglasses import apply_sunglasses
from sungfr.models.fr_net import FaceEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", choices=["clean", "sunglass_aware"], default="clean")
    ap.add_argument("--sunglasses", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device_name = args.device or cfg.get("device", "cpu")
    device = torch.device(
        device_name if not str(device_name).startswith("cuda") or torch.cuda.is_available() else "cpu"
    )
    ckpt = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = FaceEmbedder(int(blob["num_classes"]), pretrained=False).to(device)
    model.load_state_dict(blob[args.model])
    model.eval()
    size = int(cfg["model"].get("image_size", 112))
    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    if args.sunglasses:
        x = apply_sunglasses(x)
    with torch.no_grad():
        emb = model.embed(x).cpu().numpy()[0]
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "embedding.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "sunglasses": args.sunglasses, "embedding": emb.tolist()}, f)
        f.write("\n")
    print(f"Wrote {out} dim={len(emb)}")


if __name__ == "__main__":
    main()
