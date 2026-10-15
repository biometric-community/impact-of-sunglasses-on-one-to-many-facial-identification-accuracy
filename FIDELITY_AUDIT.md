# FIDELITY_AUDIT — Sunglasses 1:N impact

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Synthetic sunglasses + dual models | 94% | clean vs sunglass_aware |
| 2 | Probe degradation (verify + 1:N) | 93% | LFW −4.8pp; Rank-1 drop |
| 3 | Gallery sunglasses recovery | 92% | Rank-1 0.35%→0.67% (clean) |
| 4 | Paper corpora / AdaFace | 72% | D1–D4 (CelebA/ResNet-18) |
| 5 | Size gate + REPORT + publish | 95% | Stop |

Confidence vs paper structure: **~89%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ **4.09 GiB** → `full_train`; 8 epochs × 2 models (**2180 s**)
- Clean LFW **61.28%** → sunglass probe **56.50%**; aware LFW clean **59.47%** / probe **59.82%**
- CelebA closed-set Rank-1 (clean): probe clean **1.11%** → sunglasses **0.35%** → gallery+probe **0.67%**
- Aware recovery (gallery+probe sunglasses): Rank-1 **0.83%**
