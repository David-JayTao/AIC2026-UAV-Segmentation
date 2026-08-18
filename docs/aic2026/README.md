# AIC2026 UAV Segmentation — experiment index

Three completed phases on the AIC2026 UAV semantic-segmentation task (8 classes,
5930 / 1066 train/val split, SegFormer + MMSegmentation 1.2.2). Every phase varies
**one** factor and holds the rest of the protocol fixed, so the results compose.

Read them in order — each phase's design is a direct consequence of the previous
phase's findings.

## Current best

| | |
|---|---|
| Model | **SegFormer-B2** (MiT-B2, ImageNet-pretrained) |
| Loss | **CE + Lovasz** |
| Crop | **768 × 768**, batch 2 |
| Schedule | AdamW, LinearLR 0→1500 + PolyLR 1500→80000, 80k iterations |
| Best validated performance | **79.60 ± 0.07 mIoU** (3 seeds, val split) |
| Config | [`configs/aic2026/segformer_b2_768_ce_lovasz.py`](../../configs/aic2026/segformer_b2_768_ce_lovasz.py) |
| Cost | ~3.55 h and ~4.4 GB peak on one RTX 4090 |

mIoU is on the 1066-image val split, which is also the `save_best` selection set —
optimistic as a held-out estimate, but fair between arms since every arm was
selected the same way.

## The three phases

### Phase 1 — Loss ablation → CE+Lovasz

[**phase1_loss_ablation.md**](phase1_loss_ablation.md)

Five losses on SegFormer-B0, single seed, loss as the only varied factor:
CE+Lovasz **76.28** > CE 75.26 ≈ CE+Dice 75.15 > WCE 73.19 ≫ Focal 59.40.
Focal was unstable in this configuration; WCE traded mIoU for recall.

**Outcome:** CE+Lovasz adopted provisionally, +1.02 over CE — but on one seed, so
repeatability was the open question.

### Phase 2 — Multi-seed + error analysis → the gain is real, and it is recall

[**phase2_lovasz_multiseed.md**](phase2_lovasz_multiseed.md)

CE and CE+Lovasz repeated over seeds 2026 / 2027 / 2028, then a full error analysis
on the seed-2026 checkpoints.

- **+1.02 ± 0.05 mIoU, 3/3 seeds, non-overlapping ranges.** The paired delta is ~3×
  tighter than the within-loss spread — seed noise is common-mode and cancels.
- **42.9 % of the gain is Vehicle** (Vehicle + Road = 58 %): a small-object and
  thin-structure win, not a uniform one.
- **The mechanism is recall, not precision** (+1.40 vs +0.17) — Lovasz recovers
  regions CE leaks into Background.
- **The Phase 1 Barren signal was noise**: it sign-flips across seeds with ~5× the
  variance of any other class.
- **Two loss-independent ceilings found:** boundary accuracy ~52 % against ~89 % in
  region interiors, and ~86 % of Vehicle instances below 100 px missed under *both*
  losses. Neither is fixable by choosing a loss.

**Outcome:** CE+Lovasz confirmed as the baseline. The two ceilings look capacity- and
resolution-bound, which is what motivated Phase 3.

### Phase 3 — Backbone scaling → SegFormer-B2 is the new baseline

[**phase3_experiment_record.md**](phase3_experiment_record.md) — what was run and how
each run was verified
[**phase3_backbone_scaling.md**](phase3_backbone_scaling.md) — the results analysis
[**phase3_group_update.md**](phase3_group_update.md) — short summary, in Chinese

MiT-B0 → B1 → B2 with CE+Lovasz held fixed, three seeds at B2, plus a plain-CE
control arm at B2.

- **B2 + CE+Lovasz: 79.60 ± 0.07 mIoU** over 3 seeds, **+3.48 ± 0.19 paired against
  B0** — roughly 3.4× the size of the Phase 2 loss effect, and additive with it.
- **Ladder at seed 2026:** 76.28 → 77.49 → 79.61 (+1.21 then +2.12).
- **The Lovasz gain survives scaling:** +1.20 at B2 against +1.02 at B0. The B2 CE arm
  is single-seed, so this is one observation consistent with Phase 2, not a measured
  effect size.
- **The ladder is two axes, not one.** B0→B1 is pure width and costs no wall clock;
  B1→B2 is pure depth and costs +54 % per iteration. B2/B3/B4/B5 all share
  `embed_dims=64`, so the entire remaining ladder is on the expensive axis.
- **Largest per-class gains** (B0→B2, paired): Barren +5.13 ± 2.44, Vehicle
  +4.73 ± 0.18, Road +4.54 ± 0.90. Vehicle and Road are tight across seeds; Barren has
  by far the widest spread and should be read with care — consistent with Phase 2's
  finding that Barren is the noisiest class.
- **B3 / B4 are not recommended next:** projected +0.56 to +1.08 mIoU for 1.52× the
  wall clock, roughly 3× worse than the measured B1→B2 rate of +1.68 mIoU per extra
  GPU-hour.

**Outcome:** baseline updated to **SegFormer-B2 + CE+Lovasz**. Phase 4 priorities are
small-object Vehicle recall, higher resolution / multi-scale, boundary quality, and a
domain-gap validation split — not more backbone depth.

## Configs

| config | phase | what it is |
|---|---|---|
| [`segformer_b0_768_formal_base.py`](../../configs/aic2026/segformer_b0_768_formal_base.py) | 1 | the base every arm inherits — split, crop, batch, optimizer, scheduler, augmentation, 80k / val 4000 |
| [`loss_ce.py`](../../configs/aic2026/loss_ce.py) | 1 | plain CE (adds only a `work_dir`, so the CE definition lives in the base alone) |
| [`loss_ce_lovasz.py`](../../configs/aic2026/loss_ce_lovasz.py) | 1 | CE + Lovasz |
| [`segformer_b1_768_ce_lovasz.py`](../../configs/aic2026/segformer_b1_768_ce_lovasz.py) | 3 | MiT-B1, CE+Lovasz |
| [`segformer_b2_768_ce_lovasz.py`](../../configs/aic2026/segformer_b2_768_ce_lovasz.py) | 3 | MiT-B2, CE+Lovasz — **current baseline** |
| [`segformer_b2_768_ce.py`](../../configs/aic2026/segformer_b2_768_ce.py) | 3 | MiT-B2, plain CE — the loss-control arm |

Each new config overrides only its one factor and restates nothing, so the controlled
variables cannot drift. This was verified mechanically per phase by resolving every
config with `Config.fromfile(...)` and diffing the flattened leaf keys.

## What is not in this repository

Training logs, checkpoints, `work_dirs`, the dataset, and the analysis scripts that
compute these numbers all live outside the repo on the training host
(`/root/autodl-tmp`). The documents here are the record; the per-run artifact
inventory is in [phase3_experiment_record.md](phase3_experiment_record.md) §8.

## Milestone tags

| tag | phase |
|---|---|
| `phase1-loss-ablation` | Phase 1 complete |
| `phase2-lovasz-multiseed` | Phase 2 complete |
| `phase3-backbone-scaling` | Phase 3 complete |
