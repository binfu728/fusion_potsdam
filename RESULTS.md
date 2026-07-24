# Potsdam Segmentation Experiments

## 实验设置

- **数据集**: Potsdam (3456 train / 2016 val), 512x512 tiles
- **类别**: 5 类 (impervious_surface, building, low_vegetation, tree, car), 标签 0/6 → ignore (255)
- **Backbone**: DINOv3 ViT-Large (patch=16, n_storage_tokens=4, mask_k_bias=True)
- **Decoder**: Mask2Former Head (50 queries)
- **img_size**: 480 (fusion) / 512 (HR-only)
- **预训练权重**: stage2+stage3-zhejiang (浙江遥感数据 SSL 预训练)

---

## 结果总表

### 1. HR 分支 (纯 DINOv3, 无 olmoearth)

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **HR 9999** | `stage2+stage3-zhejiang/9999.pth` | **88.43** | 93.73 | 22000 | ✅ 完成 |
| **HR 23999** | `stage2+stage3-zhejiang/23999.pth` | **88.11** | 93.56 | 14000 | ✅ 完成 |
| **HR 31999** | `stage2+stage3-zhejiang/31999.pth` | **87.98** | 93.47 | 14000 | ✅ 完成 |

### 2. Fusion 分支 (context = learnable masked token)

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **fusion 9999** | `stage2+stage3-zhejiang/9999.pth` | **87.85** | 93.41 | 22000 | ✅ 完成 |
| **fusion 23999** | `stage2+stage3-zhejiang/23999.pth` | **87.54** | 93.24 | 14000 | ✅ 完成 |
| **fusion 31999** | `stage2+stage3-zhejiang/31999.pth` | **87.64** | 93.29 | 14000 | ✅ 完成 |

### 3. No Contrastive Loss — HR 分支

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **nocont HR 9999** | `stage2+stage3-zhejiang_no_cl/9999.pth` | **88.49** | 93.77 | 14000 | ✅ 完成 |
| **nocont HR 23999** | `stage2+stage3-zhejiang_no_cl/23999.pth` | **88.12** | 93.57 | 14000 | ✅ 完成 |

### 4. No Contrastive Loss — Fusion 分支

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **nocont fusion 9999** | `stage2+stage3-zhejiang_no_cl/9999.pth` | **87.85** | 93.41 | 22000 | ✅ 完成 |
| **nocont fusion 23999** | `stage2+stage3-zhejiang_no_cl/23999.pth` | **87.66** | 93.30 | 14000 | ✅ 完成 |

### 5. Fusion 分支 (context = olmoearth RGB embedding)

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **fusion 31999 (olmov1)** | `stage2+stage3-zhejiang/31999.pth` | **87.64** | 93.29 | 14000 | ✅ 完成 |

---

## 汇总对比

| # | 实验名称 | Backbone | Context | Contrastive | mIoU | mFscore |
|---|---------|----------|---------|-------------|------|---------|
| 1 | nocont HR 9999 | HR | — | No | **88.49** | 93.77 |
| 2 | HR 9999 | HR | — | Yes | 88.43 | 93.73 |
| 3 | nocont HR 23999 | HR | — | No | 88.12 | 93.57 |
| 4 | HR 23999 | HR | — | Yes | 88.11 | 93.56 |
| 5 | HR 31999 | HR | — | Yes | 87.98 | 93.47 |
| 6 | fusion 9999 | fusion | mask_token | Yes | 87.85 | 93.41 |
| 7 | nocont fusion 9999 | fusion | mask_token | No | 87.85 | 93.41 |
| 8 | fusion 31999 (mask) | fusion | mask_token | Yes | 87.64 | 93.29 |
| 9 | fusion 31999 (olmov1) | fusion | olmoearth RGB | Yes | 87.64 | 93.29 |
| 10 | nocont fusion 23999 | fusion | mask_token | No | 87.66 | 93.30 |
| 11 | fusion 23999 | fusion | mask_token | Yes | 87.54 | 93.24 |

> 按 mIoU 降序排列。HR 分支整体优于 fusion 分支 (~0.5-0.7 mIoU)。

---

## 关键发现

1. **HR > Fusion**: 纯 DINOv3 backbone 始终优于 fusion backbone，差距约 0.5-0.9 mIoU
2. **Checkpoint 越早越好**: 9999.pth 在多数实验中优于 23999/31999.pth（best mIoU 出现在更早的 iter）
3. **No Contrastive Loss 微弱优势**: nocont 在 HR 分支上有 ~0.05-0.06 mIoU 的提升；在 fusion 分支上差异不明显
4. **Context 类型无显著差异**: fusion 31999 使用 learnable masked token (87.64) vs olmoearth RGB embedding (87.64) 结果一致
5. **最佳模型**: nocont HR 9999 (88.49 mIoU / 93.77 mFscore)

---

## 目录映射

| 实验名称 | work_dir | Config |
|---------|----------|--------|
| HR 9999 | `dinov3_m2f_9999` | `dinov3_m2f_9999.py` |
| HR 23999 (=v2) | `dinov3_m2f_v2` | `dinov3_m2f_v2.py` |
| HR 31999 | `dinov3_m2f_31999` | `dinov3_m2f_31999.py` |
| fusion 9999 | `fusion_vits_m2f_9999` | `fusion_vits_m2f_9999.py` |
| fusion 23999 (mask) | `fusion_vits_m2f_v2` | `fusion_vits_m2f_v2.py` |
| fusion 31999 (mask) | `fusion_vits_m2f_31999` | `fusion_vits_m2f_31999.py` |
| nocont HR 9999 | `dinov3_m2f_nocont` | `dinov3_m2f_nocont_*.py` |
| nocont HR 23999 | `dinov3_m2f_nocont_23999` | `dinov3_m2f_nocont_23999.py` |
| nocont fusion 9999 | `fusion_vits_m2f_nocont_9999` | `fusion_vits_m2f_nocont_9999.py` |
| nocont fusion 23999 | `fusion_vits_m2f_nocont_23999` | `fusion_vits_m2f_nocont_23999.py` |
| fusion 31999 (olmov1) | `fusion_vits_m2f_olmov1` | `fusion_vits_m2f_olmov1.py` |
