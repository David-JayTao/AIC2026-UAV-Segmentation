_base_ = ['./loss_ce_lovasz.py']

# ============================================================
# AIC2026 UAV Semantic Segmentation - Phase 3
# Backbone capacity scaling: SegFormer MiT-B2 + CE+Lovasz
#
# Control variable = backbone capacity ONLY.
#
# Everything else is inherited unchanged through
#   loss_ce_lovasz.py -> segformer_b0_768_formal_base.py
# i.e. dataset / train-val split / crop=768 / batch_size=2 /
# augmentation / AdamW lr=6e-5 / paramwise_cfg /
# LinearLR(0-1500)+PolyLR(1500-80000) / max_iters=80000 /
# val_interval=4000 / seed=2026 / CE+Lovasz loss_decode.
#
# Backbone hyper-parameters and the ImageNet pretrained
# checkpoint are copied verbatim from the official MMSeg v1.2.2
# config:
#   configs/segformer/segformer_mit-b2_8xb2-160k_ade20k-512x512.py
# Nothing is hand-guessed.
# ============================================================

checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b2_20220624-66e8bf70.pth'
)

model = dict(

    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint,
        ),

        # official MiT-B2
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3],
    ),

    decode_head=dict(
        in_channels=[64, 128, 320, 512],
    ),
)

work_dir = './work_dirs/aic2026_b2_768_ce_lovasz'
