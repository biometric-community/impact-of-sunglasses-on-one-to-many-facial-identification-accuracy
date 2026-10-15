# REPORT — Impact of Sunglasses on One-to-Many Facial Identification Accuracy

**Protocol:** `full`

Lab twin of Tian et al. (TBIOM): measure degradation from synthetic sunglasses on
probes and recovery via sunglasses-aware training / gallery sunglasses.

## Training

- Epochs: 8
- Train images: 141610
- Identities: 7124
- Elapsed (s): 2180.4

![Training loss](outputs/figures/full/train_loss.svg)

## Verification (LFW / CALFW / CPLFW)

| Model | Protocol | Accuracy | EER | N |
|-------|----------|----------|-----|---|
| clean | calfw_both | 50.88% | 49.12% | 6000 |
| clean | calfw_none | 52.98% | 47.02% | 6000 |
| clean | calfw_probe | 52.10% | 47.90% | 6000 |
| clean | cplfw_both | 51.95% | 48.05% | 6000 |
| clean | cplfw_none | 53.50% | 46.50% | 6000 |
| clean | cplfw_probe | 52.38% | 47.62% | 6000 |
| clean | lfw_both | 56.20% | 43.80% | 6000 |
| clean | lfw_none | 61.28% | 38.72% | 6000 |
| clean | lfw_probe | 56.50% | 43.50% | 6000 |
| sunglass_aware | calfw_both | 52.02% | 47.98% | 6000 |
| sunglass_aware | calfw_none | 52.72% | 47.28% | 6000 |
| sunglass_aware | calfw_probe | 52.62% | 47.38% | 6000 |
| sunglass_aware | cplfw_both | 52.92% | 47.08% | 6000 |
| sunglass_aware | cplfw_none | 52.93% | 47.07% | 6000 |
| sunglass_aware | cplfw_probe | 52.75% | 47.25% | 6000 |
| sunglass_aware | lfw_both | 60.22% | 39.78% | 6000 |
| sunglass_aware | lfw_none | 59.47% | 40.53% | 6000 |
| sunglass_aware | lfw_probe | 59.82% | 40.18% | 6000 |

![LFW verification](outputs/figures/full/verification_lfw.svg)

## Closed-set identification (CelebA val)

| Model | Condition | Rank-1 | N probe | N gallery |
|-------|-----------|--------|---------|-----------|
| clean | probe_clean | 1.11% | 57936 | 3044 |
| clean | probe_sunglasses | 0.35% | 57936 | 3044 |
| clean | gallery_sunglasses_probe_sunglasses | 0.67% | 57936 | 3044 |
| sunglass_aware | probe_clean | 0.80% | 57936 | 3044 |
| sunglass_aware | probe_sunglasses | 0.59% | 57936 | 3044 |
| sunglass_aware | gallery_sunglasses_probe_sunglasses | 0.83% | 57936 | 3044 |

![Identification Rank-1](outputs/figures/full/identification_rank1.svg)

## Notes

- Paper uses WebFace / commercial matchers and ND-MFAD; this rebuild trains ResNet-18 on CelebA
  and evaluates LFW/CALFW/CPLFW + CelebA closed-set 1:N with synthetic eye-region sunglasses.
- See `DEVIATIONS.md`.
