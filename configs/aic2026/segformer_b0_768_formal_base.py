_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
]

# ============================================================
# AIC2026 UAV Semantic Segmentation
# SegFormer-B0, crop=768, formal train/val split
#
# Control variables:
# model / split / crop / augment / optimizer / scheduler fixed
# Only LOSS changes in subsequent experiments.
# ============================================================

crop_size = (768, 768)

# data_root = (
#     r'C:/LYX/01_Workspace/AIC2026-UAV-data/'
#     r'2026-低空图像语义分割赛道-训练集/train/train'
# )

import os

data_root = os.getenv(
    'AIC2026_DATA_ROOT',
    'data/aic2026/train/train'
)



# Official:
# 0 Ignore
# 1 Background
# 2 Building
# 3 Road
# 4 Water
# 5 Barren
# 6 Vegetation
# 7 Agricultural
# 8 Vehicle
#
# Internally:
# reduce_zero_label=True
# 0 -> 255 Ignore
# 1~8 -> 0~7

classes = (
    'Background',
    'Building',
    'Road',
    'Water',
    'Barren',
    'Vegetation',
    'Agricultural',
    'Vehicle',
)

palette = [
    [0, 0, 0],
    [220, 20, 60],
    [128, 64, 128],
    [0, 0, 255],
    [210, 180, 140],
    [0, 128, 0],
    [255, 215, 0],
    [255, 0, 255],
]

metainfo = dict(
    classes=classes,
    palette=palette,
)

# -------------------------
# Data pipeline
# -------------------------

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),

    dict(
        type='RandomCrop',
        crop_size=crop_size,
        cat_max_ratio=0.75,
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
    ),

    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,

    sampler=dict(
        type='InfiniteSampler',
        shuffle=True,
    ),

    dataset=dict(
        type='BaseSegDataset',

        data_root=data_root,

        ann_file='splits/train.txt',

        data_prefix=dict(
            img_path='images',
            seg_map_path='masks',
        ),

        img_suffix='.png',
        seg_map_suffix='.png',

        metainfo=metainfo,

        # Official 0 Ignore -> MMSeg 255 Ignore
        reduce_zero_label=True,

        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='BaseSegDataset',

        data_root=data_root,

        ann_file='splits/val.txt',

        data_prefix=dict(
            img_path='images',
            seg_map_path='masks',
        ),

        img_suffix='.png',
        seg_map_suffix='.png',

        metainfo=metainfo,

        reduce_zero_label=True,

        pipeline=val_pipeline,

        test_mode=True,
    ),
)

test_dataloader = val_dataloader

# -------------------------
# Model
# -------------------------

checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b0_20220624-7e0fe6dd.pth'
)
model = dict(

    data_preprocessor=dict(
        size=crop_size,
    ),

    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
        ),
    ),

    decode_head=dict(
        num_classes=8,
        ignore_index=255,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0,
        ),
    ),

    test_cfg=dict(mode='whole'),
)
# -------------------------
# Evaluation
# -------------------------

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)

test_evaluator = val_evaluator

# -------------------------
# Optimizer
# -------------------------

optim_wrapper = dict(
    type='AmpOptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),

    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }
    ),
)

# -------------------------
# 80k training
# -------------------------

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=80000,
    val_interval=4000,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500,
    ),

    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False,
    ),

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3,
    ),
)

randomness = dict(seed=2026)

work_dir = './work_dirs/aic2026_segformer_b0_768_ce'