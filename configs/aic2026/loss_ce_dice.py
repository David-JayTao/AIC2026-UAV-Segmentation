_base_ = ['./segformer_b0_768_formal_base.py']

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0,
                loss_name='loss_ce',
            ),

            dict(
                type='DiceLoss',
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
                loss_name='loss_dice',
            ),
        ],
    ),
)

work_dir = './work_dirs/aic2026_b0_768_ce_dice'