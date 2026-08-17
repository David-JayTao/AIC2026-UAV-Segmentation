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
                type='LovaszLoss',
                loss_type='multi_class',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=1.0,
                loss_name='loss_lovasz',
            ),
        ],
    ),
)

work_dir = './work_dirs/aic2026_b0_768_ce_lovasz'