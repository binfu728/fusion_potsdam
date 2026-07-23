custom_imports = dict(
    imports=["custom_datasets.customPotsdam", "custom_models.dinov3_backbone"],
    allow_failed_imports=False,
)

_base_ = [
    "/mnt/ht2-nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py",
]

# ── 路径 ───────────────────────────────────────────────────────────────
DINO_CKPT = "/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/9999.pth"
DATA_ROOT = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/"

# ── backbone: 纯 DINOv3 vit_large（从 fusion ckpt 取 backbone 权重，冻结）──
img_size = 512
num_classes = 5         # potsdam: 原始标签 1-5 (0/6 ignored → remapped to 0-4)

data_preprocessor = dict(
    type="SegDataPreProcessor", _delete_=True,
    mean=None, std=None, bgr_to_rgb=False,
    pad_val=0, seg_pad_val=255,
    size=(img_size, img_size), test_cfg=dict(size_divisor=32))

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="DINOv3BackboneMmseg",
        arch="vit_large",
        patch_size=16,
        interaction_indexes=[5, 11, 17, 23],
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
        img_size=img_size,
        checkpoint=DINO_CKPT,
        freeze_backbone=True,       # True=冻结ViT只训adapter+m2f; False=全参微调
        finetune_vit=False,         # True=forward中ViT参与反传(需freeze_backbone=False)
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        num_queries=50,
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss", use_sigmoid=False,
            loss_weight=2.0, reduction="mean", class_weight=[1.0] * (num_classes + 1)),
    ),
)

# ── 数据流水线（potsdam HR 数据，无 fusion/olmoearth/fake_h5）───────────
train_pipeline = [
    dict(type="LoadCustomRaster", img_size=img_size),
    dict(type="CustomRandomRotate90", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="CustomNormalize"),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadCustomRaster", img_size=img_size),
    dict(type="CustomNormalize"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=4,
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="train", pipeline=train_pipeline))
val_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=4,
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="val", pipeline=val_pipeline))
test_dataloader = dict(
    _delete_=True, batch_size=4, num_workers=4,
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="val", pipeline=val_pipeline))

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"], _delete_=True)
test_evaluator = val_evaluator

# ── optimizer & schedule ──────────────────────────────────────────────
optim_wrapper = dict(
    _delete_=True, type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 冻结模式: lr_mult=0.0; 全参微调: 改成 lr_mult=0.1 给 ViT 较小学习率
            "backbone.backbone": dict(lr_mult=0.0),   # ViT
        },
        norm_decay_mult=0.0))

max_iters = 40000
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, begin=0, end=3000, by_epoch=False),
    dict(type="PolyLR", eta_min=0, power=0.9, begin=3000, end=max_iters, by_epoch=False)]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000,
                    save_best="mIoU", max_keep_ckpts=1),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False))

find_unused_parameters = True
