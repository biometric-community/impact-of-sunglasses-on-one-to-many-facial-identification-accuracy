# DEVIATIONS — Sunglasses 1:N impact

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train corpus | WebFace / Glint-scale | CelebA identities | Local public data |
| D2 | Matcher | AdaFace / ArcFace commercial | ResNet-18 CE embedder | Compute / open weights |
| D3 | Eval 1:N | ND-MFAD, MORPH, IJB-C | CelebA closed-set + LFW/CALFW/CPLFW verify | ND-MFAD / MORPH restricted or unavailable |
| D4 | Sunglasses | StyleGAN / high-quality synthetic | Analytic eye-region ellipses | No StyleGAN pipeline locally |
| D5 | Blur / resolution ablations | Combined with sunglasses | Not in default run | Scope / runtime |
