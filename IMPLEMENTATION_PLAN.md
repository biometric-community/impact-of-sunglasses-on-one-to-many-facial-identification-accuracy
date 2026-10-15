# IMPLEMENTATION_PLAN — Sunglasses impact on 1:N identification

## Paper claims to cover

1. Sunglasses on probe degrade one-to-many identification accuracy.
2. Synthetic sunglasses augmentation can train a sunglasses-aware matcher.
3. Adding sunglasses to gallery images can recover accuracy when probes wear sunglasses.

## Lab twin design

| Component | Choice |
|-----------|--------|
| Train data | CelebA identities (WebFace substitute) |
| Backbone | ResNet-18 + 512-d embed + CE |
| Occlusion | Synthetic eye-region dark ellipses |
| Models | `clean` (no aug) vs `sunglass_aware` (p=0.5) |
| Verify | LFW / CALFW / CPLFW: clean, probe-sunglasses, both |
| Identify | CelebA val closed-set Rank-1: probe clean / probe sunglasses / gallery+probe sunglasses |

## Entry points

`scripts/smoke.sh` → `scripts/train_full.sh` (size gate) → `REPORT.md`
