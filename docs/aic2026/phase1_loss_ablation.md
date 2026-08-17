# AIC2026 UAV Segmentation - Phase 1 Loss Ablation Summary

## 1. Experiment setup
- Model: SegFormer-B0
- Framework: MMSegmentation 1.2.2
- Crop: 768×768
- Batch size: 2
- Train/val: 5930 / 1066
- Max iter: 80,000
- Val interval: 4,000
- Optimizer: AdamW
- Scheduler: LinearLR 0-1500 + PolyLR 1500-80000
- Seed: 2026
- Controlled variable: loss only
- Git commit: `7ef9e65`
- 5× RTX 4090 parallel runs

## 2. Final results

| Config | Final mIoU | Best mIoU | Best iter | aAcc | mAcc | Stable s/iter | Peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CE+Lovasz** | **76.28** | **76.28** | 80000 | 87.61 | 87.36 | 0.0977 | 2200 MB |
| CE | 75.26 | 75.26 | 80000 | 87.32 | 85.96 | 0.0934 | 1992 MB |
| CE+Dice | 75.15 | 75.26 | 76000 | 87.23 | 86.04 | 0.0969 | 2193 MB |
| WCE | 73.19 | 73.19 | 80000 | 86.35 | **89.37** | 0.0922 | 1990 MB |
| Focal | 59.40 | 65.16 | 64000 | 81.13 | 67.52 | 0.0940 | 2019 MB |

## 3. Per-class IoU

| Class | CE+Lovasz | CE | CE+Dice | WCE | Focal |
|---|---:|---:|---:|---:|---:|
| Background | **67.80** | 67.31 | 67.03 | 64.17 | 56.71 |
| Building | **80.56** | 79.99 | 80.25 | 79.43 | 73.23 |
| Road | **76.36** | 75.69 | 75.28 | 75.28 | 60.90 |
| Water | **87.93** | 86.89 | 86.14 | 87.14 | 80.50 |
| Barren | **55.28** | 53.93 | 53.74 | 48.65 | 0.00 |
| Vegetation | **87.52** | 87.38 | 87.32 | 86.78 | 83.34 |
| Agricultural | 78.43 | **78.51** | 78.31 | 77.99 | 69.90 |
| Vehicle | **76.37** | 72.41 | 73.16 | 66.04 | 50.66 |

## 4. Main conclusions
1. **CE+Lovasz is the current best baseline**: 76.28 mIoU, +1.02 over CE.
2. **CE and CE+Dice are essentially tied in this single run**. Dice does not show a clear gain.
3. **WCE accelerates minority-class learning early but hurts final balance**. Its highest mAcc with lower mIoU is consistent with a recall-heavy / over-prediction tendency, but precision/confusion analysis is needed before claiming this definitively.
4. **Focal is unstable in the current configuration**: final 59.40, peak 65.16 at 64k, severe validation oscillation and 6 non-finite logged loss values.
5. Healthy runs remain near their best at the end, so 80k is a reasonable first-round budget.

## 5. Interpretation cautions
- One split + one seed is not enough for statistical significance.
- Lovasz helping Vehicle is consistent with an IoU-oriented objective, but one run cannot prove causality.
- Focal failure cannot be attributed to alpha=0.25 alone; sigmoid one-vs-all formulation, imbalance and optimization instability may all contribute.

## 6. Recommended next stage
- **A. Repeatability check:** CE and CE+Lovasz with 1–2 additional seeds.
- **B. Error analysis:** confusion matrix, precision/recall, qualitative overlays for Vehicle/Barren/Road.
- **C. Stronger baseline:** SegFormer-B2 + CE+Lovasz under the same split/protocol.
- **D. Domain-gap validation:** add a source/scene-aware secondary validation split before aggressive tuning.
- Lower priority: WCE retuning, Focal rescue, CE+Dice coefficient sweep.
