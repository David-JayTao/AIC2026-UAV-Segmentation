# AIC2026 UAV Segmentation — Phase 3 experiment record

Formal 80k backbone-capacity-scaling experiments: SegFormer MiT-B1 and
MiT-B2 against the established MiT-B0 baseline, plus a loss-control arm
at B2. Five single-GPU runs, one experiment per GPU.

This document records **what was run and how it was checked**. The
interpretation of the numbers is in
[phase3_backbone_scaling.md](phase3_backbone_scaling.md); the short version for
sharing is [phase3_group_update.md](phase3_group_update.md).

| | |
|---|---|
| Record generated | 2026-08-18T13:05:43+08:00 |
| Launched at | 2026-08-18T00:35:19+08:00 |
| Launch manifest | `phase3/manifest_formal.txt` |
| Repo | `AIC2026-UAV-Segmentation` (this repository) |
| Branch | `main` |
| Commit at launch | `3f11376` (recorded in the manifest) |
| Commit now | `3f11376` |
| Python | `/root/miniconda3/envs/aic-seg/bin/python` |
| MMSegmentation | 1.2.2 |
| PyTorch / CUDA | 2.1.0 / 12.1 |
| Data root | `/root/autodl-tmp/AIC2026-UAV-data/train/train` |
| MMSeg source modified | **no** |
| Phase 1/2 configs modified | **no** |
| Logs / checkpoints committed | **no** (all under `/root/autodl-tmp`, outside the repo) |

## 1. Experiment matrix

| GPU | run | config | backbone | loss | seed | work_dir |
|---|---|---|---|---|---|---|
| 0 | `phase3_b2_lovasz_2026` | `segformer_b2_768_ce_lovasz.py` | MiT-B2 | CE+Lovasz | 2026 | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2026` |
| 1 | `phase3_b2_lovasz_2027` | `segformer_b2_768_ce_lovasz.py` | MiT-B2 | CE+Lovasz | 2027 | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2027` |
| 2 | `phase3_b2_lovasz_2028` | `segformer_b2_768_ce_lovasz.py` | MiT-B2 | CE+Lovasz | 2028 | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2028` |
| 3 | `phase3_b2_ce_2026` | `segformer_b2_768_ce.py` | MiT-B2 | CE | 2026 | `/root/autodl-tmp/work_dirs/phase3_b2_ce_2026` |
| 4 | `phase3_b1_lovasz_2026` | `segformer_b1_768_ce_lovasz.py` | MiT-B1 | CE+Lovasz | 2026 | `/root/autodl-tmp/work_dirs/phase3_b1_lovasz_2026` |

The matrix is a two-factor design with one factor varied at a time:

* **capacity arm** — B0 (already had 3 seeds from Phase 2) → B1 → B2, loss
  held at CE+Lovasz. Answers "does capacity help, and how much".
* **loss arm** — B2 CE versus B2 CE+Lovasz at the shared seed 2026.
  Answers "does the Phase 2 loss result survive at higher capacity".
* **seed arm** — B2 CE+Lovasz at 2026 / 2027 / 2028, so the capacity
  effect can be compared against seed noise rather than asserted.

## 2. Training protocol (held fixed, inherited not restated)

Every controlled variable is **inherited** from
`configs/aic2026/segformer_b0_768_formal_base.py` through the Phase 1/2
loss configs. The new configs restate none of them, so they cannot drift:

| variable | value | source |
|---|---|---|
| train/val split | unchanged from Phase 1 | base config |
| crop size | 768 × 768 | base config |
| batch size | 2 train / 1 val | base config |
| optimizer | AdamW, lr 6e-5, wd 0.01 | base config |
| paramwise_cfg | custom_keys unchanged (pos_block/norm/head) | base config |
| scheduler | LinearLR 0→1500 then PolyLR 1500→80000 | base config |
| augmentation | unchanged from Phase 1 | base config |
| max_iters | 80,000 | base config |
| val_interval | 4,000 (→ 20 validations per run) | base config |
| AMP | AmpOptimWrapper | base config |
| reduce_zero_label | True (label 0 Ignore→255, 1–8 → 0–7) | base config |
| classes | 8 | base config |
| init | ImageNet pretrained MiT-B1/B2 from iteration 0 | new config (backbone `init_cfg`) |
| resume | no — `--resume` never passed, all work_dirs empty at launch | `launch.sh` |

Only two things were passed on the command line, per run:
`randomness.seed` and `work_dir`. Nothing else was overridden.

Verified mechanically, not by eye — `phase3/config_diff.txt` and
`phase3/config_diff_b2ce.txt` resolve each config with
`Config.fromfile(...).to_dict()`, flatten it to leaf keys, and diff:

```
config_diff.txt      : RESULT: ALL CONTROLLED VARIABLES IDENTICAL
config_diff_b2ce.txt : RESULT: ALL CHECKS PASSED
```

## 3. Configs used

| config | status in git | md5 | backbone hyper-parameters |
|---|---|---|---|
| `segformer_b1_768_ce_lovasz.py` | tracked (commit `3f11376`) | `88857bb553a4` | embed_dims 64, num_layers [2,2,2,2], heads [1,2,5,8] → 8 blocks |
| `segformer_b2_768_ce_lovasz.py` | tracked (commit `3f11376`) | `9be6f1108f08` | embed_dims 64, num_layers [3,4,6,3], heads [1,2,5,8] → 16 blocks |
| `segformer_b2_768_ce.py` | tracked (commit `bb12304`) | `dbadfe321b8c` | embed_dims 64, num_layers [3,4,6,3], heads [1,2,5,8] → 16 blocks |

All three backbone hyper-parameter sets are copied verbatim from the
official MMSeg v1.2.2 SegFormer configs; none are hand-guessed.

### 3.1 The B2 + plain-CE config (GPU 3)

This arm needed a CE definition *exactly* identical to Phase 1 plain CE.
That was achieved structurally rather than by copying. Phase 1 CE is
`configs/aic2026/loss_ce.py`, which is three lines and adds nothing to the
base — it only sets a `work_dir`. The single `loss_decode` definition
therefore lives in the base config alone:

```python
loss_decode=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    avg_non_ignore=True,
    loss_weight=1.0,
),
```

`segformer_b2_768_ce.py` inherits `loss_ce.py` and overrides only the
backbone and the decode-head input channels. It never restates the loss,
so identity is guaranteed by construction, not by a matching diff:

| check | result |
|---|---|
| `loss_decode` B2-CE vs B0-CE | byte-identical (both a single `dict`, not a list) |
| `backbone` B2-CE vs B2-CE+Lovasz | byte-identical |
| B2-LOV vs B2-CE resolved leaf keys | 200 keys, 17 differ — all `loss_decode` + `work_dir` |
| B0-CE vs B2-CE resolved leaf keys | 188 keys, 6 differ — `checkpoint`, `embed_dims`, `init_cfg.checkpoint`, `num_layers`, `in_channels`, `work_dir` |
| controlled variables across all 5 configs | 44 keys identical |
| train/val pipelines | identical |
| parameter count B2-CE vs B2-LOV | identical (24,725,192) |

## 4. Pre-flight gates (before any 80k run started)

| gate | scope | artifact | result |
|---|---|---|---|
| config integrity / controlled-variable diff | B1, B2 CE+Lovasz | `config_diff.txt` | pass |
| pretrained tensor-shape verification | B1, B2 | `verify_pretrained.txt` | pass |
| 5-iteration smoke | B1, B2 CE+Lovasz | `smoke_b1.log`, `smoke_b2.log` | pass (exit 0) |
| 200-iteration timing probe | B1, B2 | `timing_summary.txt` | pass |
| resolved-config diff | **B2 plain CE** | `config_diff_b2ce.txt` | ALL CHECKS PASSED |
| 5-iteration smoke | **B2 plain CE** | `smoke_b2ce.log` | pass (exit 0) |

The B2+CE smoke run produced a strong incidental cross-check. Its
`loss_ce` at iteration 5 (2.0547) and its validation mIoU (5.53) both match
the B2+CE+Lovasz smoke run exactly — the same initialisation and the same
data order, which is what "only the loss differs" has to mean.
`loss_lovasz` is correctly absent from the CE run's log fields.

## 5. Launch

`phase3/launch.sh`, one detached `screen` session per run, each pinned to
one physical GPU with `CUDA_VISIBLE_DEVICES`, each writing its own launch
log, pid file and exit-code file.

The script refuses to start if any target `work_dir` already contains a
`*.pth`, which is the third independent guarantee of no-resume (the other
two: `--resume` is never passed and defaults to False; all five
work_dirs were empty at launch).

| GPU | GPU UUID | run | started | wrapper pid |
|---|---|---|---|---|
| 0 | `GPU-8386c3c1-3b8e-77a4-9438-a7e788d21e37` | `phase3_b2_lovasz_2026` | 2026-08-18T00:35:19+08:00 | 8254 |
| 1 | `GPU-20c70abf-4f1a-3092-3559-8b9b3d751eb8` | `phase3_b2_lovasz_2027` | 2026-08-18T00:35:21+08:00 | 8333 |
| 2 | `GPU-506c8067-9d3b-dba9-bcdb-c0a3cbe24289` | `phase3_b2_lovasz_2028` | 2026-08-18T00:35:24+08:00 | 8484 |
| 3 | `GPU-91dbf18b-4d49-6792-cf76-5a573e9c70d2` | `phase3_b2_ce_2026` | 2026-08-18T00:35:26+08:00 | 8765 |
| 4 | `GPU-ee7d3795-811d-bd73-56a4-be3e578088f6` | `phase3_b1_lovasz_2026` | 2026-08-18T00:35:28+08:00 | 9053 |

## 6. Health gates

### 6.1 The 100-iteration gate (live)

Run once all five processes were past iteration 100, while they were still
alive, because liveness and GPU mapping can only be verified live:
`/proc/<pid>/environ` and the `nvidia-smi` compute-app list both vanish on
exit. Artifact: `health_check_100.txt`. Result: **ALL 5 RUNS HEALTHY**.

| checked | how |
|---|---|
| process alive | real python child resolved from the screen wrapper pid via `pgrep -P` + `/proc/<k>/cmdline` |
| GPU mapping correct | `CUDA_VISIBLE_DEVICES` read out of `/proc/<pid>/environ`, cross-checked against the per-GPU pid list from `nvidia-smi` |
| GPU exclusivity | exactly one compute pid per GPU; pid absent from every other GPU |
| loss finite | every numeric field on every `Iter(train)` line parsed and `math.isfinite`-tested |
| no NaN/Inf | word-boundary regex, so `INFO` / `InfiniteSampler` / `metainfo` do not false-positive |
| no OOM / traceback | explicit pattern match |
| seed + work_dir applied | read back out of the log, not assumed from the CLI |
| pretrained loaded | checkpoint filename confirmed in the log, no key/shape warnings |
| loss terms | `loss_lovasz` present iff the arm is CE+Lovasz |

### 6.2 The completion gate (post-hoc)

Re-run after all five exited. Artifact: `health_check_final.txt`.
Result: **ALL 5 RUNS HEALTHY**. In this mode a dead process and an idle
GPU are the *correct* state, so those two checks report `N/A` and defer to
the live gate; everything log-derived is re-checked over the full 80k.

| run | exit code | last iter | validations | pretrained | logged loss values | non-finite | NaN/Inf tokens | fatal | resumed |
|---|---|---|---|---|---|---|---|---|---|
| `phase3_b2_lovasz_2026` | 0 | 80,000 | 20 | `mit_b2_20220624-66e8bf70.pth` | 4,800 | 0 | 0 | none | no |
| `phase3_b2_lovasz_2027` | 0 | 80,000 | 20 | `mit_b2_20220624-66e8bf70.pth` | 4,800 | 0 | 0 | none | no |
| `phase3_b2_lovasz_2028` | 0 | 80,000 | 20 | `mit_b2_20220624-66e8bf70.pth` | 4,800 | 0 | 0 | none | no |
| `phase3_b2_ce_2026` | 0 | 80,000 | 20 | `mit_b2_20220624-66e8bf70.pth` | 3,200 | 0 | 0 | none | no |
| `phase3_b1_lovasz_2026` | 0 | 80,000 | 20 | `mit_b1_20220624-02e5a6a1.pth` | 4,800 | 0 | 0 | none | no |

Across the five Phase 3 runs: **22,400 logged loss values, 0 non-finite,
0 NaN/Inf tokens, 0 tracebacks,
0 OOM events, 0 resumes**. All five reached exactly 80,000 iterations and
completed all 20 validations.

## 7. Timeline and wall clock

| run | GPU | first log line | last log line | wall | s/iter | peak mem |
|---|---|---|---|---|---|---|
| `phase3_b2_lovasz_2026` | 0 | 08-18 00:35:20 | 08-18 04:07:08 | 3.53 h | 0.1501 | 4,426 MB |
| `phase3_b2_lovasz_2027` | 1 | 08-18 00:35:22 | 08-18 04:04:03 | 3.48 h | 0.1477 | 4,432 MB |
| `phase3_b2_lovasz_2028` | 2 | 08-18 00:35:24 | 08-18 04:14:14 | 3.65 h | 0.1553 | 4,436 MB |
| `phase3_b2_ce_2026` | 3 | 08-18 00:35:27 | 08-18 04:06:17 | 3.51 h | 0.1492 | 4,280 MB |
| `phase3_b1_lovasz_2026` | 4 | 08-18 00:35:29 | 08-18 02:53:25 | 2.30 h | 0.0975 | 2,785 MB |

All five ran concurrently, one per GPU, from 2026-08-18 00:35:20 to
2026-08-18 04:14:14 — **3.65 h of
elapsed time for 16.47 GPU-hours of work**.

Timing note: wall clock and s/iter here come from the logs' own
timestamps, **not** from mmengine's `time` field. With the default
`LogProcessor(window_size=10)` that field is a *rolling* mean over the
last 10 iterations, so it cannot be integrated to a total. (The Phase 3
step-1 record describes it as a cumulative mean; that description is
wrong, though the numeric conclusion it drew from it still holds.)

## 8. Artifacts on disk

Nothing below is in the repo; all of it is under `/root/autodl-tmp`.

| run | work_dir | checkpoints kept | best checkpoint | size |
|---|---|---|---|---|
| `phase3_b2_lovasz_2026` | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2026` | iter_72000.pth, iter_76000.pth, iter_80000.pth | `best_mIoU_iter_72000.pth` | 1.0 GB |
| `phase3_b2_lovasz_2027` | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2027` | iter_72000.pth, iter_76000.pth, iter_80000.pth | `best_mIoU_iter_80000.pth` | 1.0 GB |
| `phase3_b2_lovasz_2028` | `/root/autodl-tmp/work_dirs/phase3_b2_lovasz_2028` | iter_72000.pth, iter_76000.pth, iter_80000.pth | `best_mIoU_iter_80000.pth` | 1.0 GB |
| `phase3_b2_ce_2026` | `/root/autodl-tmp/work_dirs/phase3_b2_ce_2026` | iter_72000.pth, iter_76000.pth, iter_80000.pth | `best_mIoU_iter_80000.pth` | 1.0 GB |
| `phase3_b1_lovasz_2026` | `/root/autodl-tmp/work_dirs/phase3_b1_lovasz_2026` | iter_72000.pth, iter_76000.pth, iter_80000.pth | `best_mIoU_iter_72000.pth` | 0.6 GB |

Total Phase 3 checkpoint footprint: **4.8 GB** (`max_keep_ckpts=3` plus one best per run).

Verification and analysis artifacts, all in `/root/autodl-tmp/phase3/`:

| file | what it is |
|---|---|
| `manifest_formal.txt` | launch manifest — commit, GPU UUIDs, pids, start times |
| `config_diff.txt` | controlled-variable diff for B1 / B2 CE+Lovasz |
| `config_diff_b2ce.txt` | resolved-config diff for the B2 plain-CE arm |
| `verify_pretrained.txt` | pretrained checkpoint tensor-shape verification |
| `smoke_b1.log / smoke_b2.log / smoke_b2ce.log` | 5-iteration smoke runs |
| `timing_summary.txt` | 200-iteration timing probe |
| `health_check_100.txt` | the live 100-iteration health gate |
| `health_check_final.txt` | the post-completion health gate |
| `parse_runs.py` | log + scalars.json parser with cross-checks |
| `phase3_stats.py` | seed statistics and paired comparisons |
| `phase3_projection.py` | B3 / B4 cost and accuracy projection |
| `phase3_plots.py` | chart generation |
| `phase3_record.py` | generates this document |
| `phase3_report.py` | generates `PHASE3_RESULTS_ANALYSIS.md` |
| `health_check.py` | the health checker used for both gates |
| `plots/` | the seven charts, published to `docs/aic2026/assets/` |
| `verify_b2ce_for_commit.py` | the pre-commit gate for the B2 plain-CE config |
| `publish_docs.py` | copies these documents and charts into `docs/aic2026/` |
| `PHASE3_EXPERIMENT_RECORD.md` | this document, published as `phase3_experiment_record.md` |
| `PHASE3_RESULTS_ANALYSIS.md` | the results analysis, published as `phase3_backbone_scaling.md` |

## 9. How the reported numbers are derived

Two independent sources are combined and cross-checked against each other:

* `work_dir/*/vis_data/scalars.json` — authoritative `(step, mIoU, aAcc,
  mAcc)` for each of the 20 validations.
* the launch log — the per-class IoU tables, which do **not** appear in
  `scalars.json` at all.

The k-th per-class table is aligned to the k-th `scalars.json` step, and
the mean of the 8 per-class IoUs is compared against that step's reported
mIoU (tolerance 0.02). A mismatch is reported, not silently dropped.
Alignment errors across all runs: **0**.

"Final mIoU" always means the iteration-80000 validation, never the best
checkpoint. One run (`phase3_b2_lovasz_2026`) peaked slightly higher at
72k (79.71) than at 80k (79.61); the 80k value is the one reported, for
all runs, so the comparison stays like-for-like.

## 10. Deviations, caveats and open items

| item | detail |
|---|---|
| `segformer_b2_768_ce.py` postdates the launch commit | It was created for the GPU-3 arm after commit `3f11376`, so the commit recorded in the launch manifest does **not** contain the config that run trained from. It has since been committed as `bb12304`, unchanged since launch (md5 `dbadfe321b8c`). Before committing, `verify_b2ce_for_commit.py` re-resolved the repo file and compared its 188 leaf keys against **both** copies of the resolved config archived in the GPU-3 work_dir (`work_dir/segformer_b2_768_ce.py` and `20260818_003526/vis_data/config.py`): identical apart from `work_dir` and `launcher`, the two keys `tools/train.py` writes into the Config object itself at launch. The run is therefore reproducible from the repo. |
| B1 and the B2-CE arm are single-seed | Only seed 2026. The capacity arm has 3 seeds at B0 and at B2, so the B0→B2 comparison is paired and seed-controlled; the B0→B1→B2 ladder and the B2 CE-vs-CE+Lovasz contrast are single-seed at one end and are reported as such. |
| step-1 record error | `phase3_config_smoke_record.md` describes mmengine's `time` field as a cumulative mean; it is a rolling mean (window 10). All timing in this record and in the analysis comes from log timestamps instead. |
| record/remote discrepancy | `phase3_config_smoke_record.md` states commit `3f11376` was not pushed, but `origin/main` was already at `3f11376` (0 ahead, 0 behind) when Phase 3 was archived. Noted for the record. |

---

*Record generated by `phase3_record.py` on 2026-08-18T13:05:45+08:00.*
