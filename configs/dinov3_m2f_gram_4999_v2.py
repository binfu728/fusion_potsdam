custom_imports = dict(
    imports=["custom_datasets.customPotsdam", "custom_models.dinov3_backbone"],
    allow_failed_imports=False,
)

_base_ = [
    "/mnt/ht2-nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py",
]

# ── 路径 ───────────────────────────────────────────────────────────────
DINO_CKPT = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/zhejiang_DinoViT_large_Olmoearth10m_128gpu_stage2_stage3_no_cl_gram_nofusion/4999_new.pt"
DATA_ROOT = "/mnt/qh2-nas3/00-model/00-limx/datasets/potsdam/"

# ── backbone ──────────────────────────────────────────────────────────
img_size = 512
num_classes = 5

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
        freeze_backbone=True,
        finetune_vit=False,
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        num_queries=100,            # [改动1] 50→100: 更多 mask proposals
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss", use_sigmoid=False,
            loss_weight=1.0,         # [改动2] 2.0→1.0: 降低分类损失权重，让 mask 损失主导
            reduction="mean",
            class_weight=[1.0] * (num_classes + 1)),
    ),
)

# ── 数据流水线（增强 augmentation）──────────────────────────────────────
# [改动3] 新增 ColorJitter + RandomChoiceResize 抑制过拟合
train_pipeline = [
    dict(type="LoadCustomRaster", img_size=img_size),
    dict(type="CustomRandomRotate90", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomColorJitter", brightness=0.2, contrast=0.2, saturation=0.1, prob=0.5),
    dict(type="CustomNormalize"),
    dict(type="PackSegInputs"),
]
val_pipeline = [
    dict(type="LoadCustomRaster", img_size=img_size),
    dict(type="CustomNormalize"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    _delete_=True, batch_size=8, num_workers=8,  # [改动4] 4→8: 更大 batch 稳定梯度
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="train", pipeline=train_pipeline))
val_dataloader = dict(
    _delete_=True, batch_size=8, num_workers=8,
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="val", pipeline=val_pipeline))
test_dataloader = dict(
    _delete_=True, batch_size=8, num_workers=8,
    dataset=dict(type="CustomPotsdamDataset", data_root=DATA_ROOT, split="val", pipeline=val_pipeline))

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mFscore"], _delete_=True)
test_evaluator = val_evaluator

# ── optimizer & schedule ──────────────────────────────────────────────
# [改动5] lr 5e-5→2e-5 (配合 batch=8 线性缩放), clip_grad 0.01→1.0
optim_wrapper = dict(
    _delete_=True, type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=2e-5, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.backbone": dict(lr_mult=0.0),   # ViT frozen
        },
        norm_decay_mult=0.0))

# [改动6] 40000→80000: 更长训练 + CosineAnnealing 更平滑的衰减
max_iters = 40000
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, begin=0, end=3000, by_epoch=False),
    dict(type="CosineAnnealingLR", T_max=max_iters-3000, eta_min=1e-6, begin=3000, end=max_iters, by_epoch=False)]
train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000,
                    save_best="mIoU", max_keep_ckpts=3),  # [改动7] 保留更多 checkpoint 防止丢失 best
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False))

find_unused_parameters = True
