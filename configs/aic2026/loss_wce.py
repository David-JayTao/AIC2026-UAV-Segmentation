_base_ = ['./segformer_b0_768_formal_base.py']

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0,

            class_weight=[
                0.422,  # Background
                0.521,  # Building
                0.749,  # Road
                0.886,  # Water
                1.651,  # Barren
                0.416,  # Vegetation
                0.840,  # Agricultural
                2.513,  # Vehicle
            ],
        ),
    ),
)

work_dir = './work_dirs/aic2026_b0_768_wce'