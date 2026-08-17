_base_ = ['./segformer_b0_768_formal_base.py']

model = dict(
    decode_head=dict(
        loss_decode=dict(
            _delete_=True,

            type='FocalLoss',
            use_sigmoid=True,

            gamma=2.0,
            alpha=0.25,

            loss_weight=1.0,
            loss_name='loss_focal',
        ),
    ),
)

work_dir = './work_dirs/aic2026_b0_768_focal'