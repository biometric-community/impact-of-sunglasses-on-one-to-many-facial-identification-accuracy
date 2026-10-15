# Impact of Sunglasses on One-to-Many Facial Identification Accuracy

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin measuring how **synthetic sunglasses** on probes degrade face verification / closed-set identification, and how **sunglasses-aware training** and **gallery sunglasses** recover accuracy.

**Repository:** https://github.com/biometric-community/impact-of-sunglasses-on-one-to-many-facial-identification-accuracy

**Paper:** [10.1109/TBIOM.2026.3666896](https://doi.org/10.1109/TBIOM.2026.3666896)

## Setup

```bash
bash scripts/setup_env.sh
```

Point configs at CelebA + LFW (+ optional CALFW/CPLFW) under `projects/datasets/`.

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
