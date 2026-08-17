_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
]

# ============================================================
# AIC2026 UAV Segmentation - official Examples smoke baseline
# PURPOSE: verify full train/val pipeline, NOT official score.
# ============================================================

crop_size = (512, 512)

# Official labels:
# 0 ignore
# 1 background
# 2 building
# 3 road
# 4 water
# 5 barren
# 6 forest
# 7 farmland
# 8 vehicle
#
# reduce_zero_label=True:
# 0 -> 255(ignore), 1~8 -> 0~7

classes = (
    'background',
    'building',
    'road',
    'water',
    'barren',
    'forest',
    'farmland',
    'vehicle',
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

metainfo = dict(classes=classes, palette=palette)

#data_root = r'C:/LYX/01_Workspace/AIC2026-UAV-data/Examples/Examples/Examples'

import os

data_root = os.getenv(
    'AIC2026_EXAMPLES_ROOT',
    'data/aic2026/examples'
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 1.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.90),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(
            img_path='images',
            seg_map_path='masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=True,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BaseSegDataset',
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(
            img_path='images',
            seg_map_path='masks'),
        img_suffix='.png',
        seg_map_suffix='.png',
        metainfo=metainfo,
        reduce_zero_label=True,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
)

test_evaluator = val_evaluator

# MiT-B0 ImageNet pretrained backbone
checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b0_20220624-7e0fe6dd.pth'
)

model = dict(
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
    ),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
        )
    ),
    decode_head=dict(
        num_classes=8,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
)

optim_wrapper = dict(
    type='OptimWrapper',
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

# Smoke only: 100 iterations
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=100,
    val_interval=25,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=10,
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=10,
        end=100,
        by_epoch=False,
    ),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=25,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=2,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

randomness = dict(seed=42)

work_dir = './work_dirs/aic2026_segformer_b0_examples_smoke'
