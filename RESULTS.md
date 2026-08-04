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
| **fusion 9999** (context = mask_token) | `stage2+stage3-zhejiang/9999.pth` | **87.85** | 93.41 | 22000 | ✅ 完成 |
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

### 6. Gram Loss (no contrastive, no fusion) — HR 分支

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| gram HR 4999 | `gram_nofusion/4999_new.pt` | 88.39 | 93.75 | 14000 | ✅ 完成 |

---

## 汇总对比

| # | 实验名称 | Backbone | Pretrain | Context | Contrastive | mIoU | mFscore | mean mIoU | mean mFscore |
|---|---------|----------|----------|---------|-------------|------|---------|-----------|-------------|
| 1 | **DINOv3 LVD-1689M (ori)** | HR | ImageNet | — | — | **89.00** | 94.07 | 87.72 | 93.32 |
| 2 | nocont HR 9999 | HR | zhejiang | — | No | 88.49 | 93.77 | 86.62 | 92.66 |
| 3 | HR 9999 | HR | zhejiang | — | Yes | 88.43 | 93.73 | 87.15 | 92.99 |
| 4 | gram HR 4999 | HR | gram | — | No (gram) | 88.39 | 93.75 | 86.39 | 92.56 |
| 5 | nocont HR 23999 | HR | zhejiang | — | No | 88.12 | 93.57 | 86.42 | 92.56 |
| 6 | HR 23999 | HR | zhejiang | — | Yes | 88.11 | 93.56 | 86.39 | 92.53 |
| 7 | HR 31999 | HR | zhejiang | — | Yes | 87.98 | 93.47 | 86.04 | 92.31 |
| 8 | fusion 9999 | fusion | zhejiang | mask_token | Yes | 87.85 | 93.41 | 86.22 | 92.43 |
| 9 | nocont fusion 9999 | fusion | zhejiang | mask_token | No | 87.85 | 93.41 | 86.62 | 92.69 |
| 10 | nocont fusion 23999 | fusion | zhejiang | mask_token | No | 87.66 | 93.30 | 86.28 | 92.49 |
| 11 | fusion 31999 (mask) | fusion | zhejiang | mask_token | Yes | 87.64 | 93.29 | 85.29 | 91.86 |
| 12 | fusion 31999 (olmov1) | fusion | zhejiang | olmoearth RGB | Yes | 87.64 | 93.29 | 86.18 | 92.43 |
| 13 | fusion 23999 | fusion | zhejiang | mask_token | Yes | 87.54 | 93.24 | 86.08 | 92.37 |
| 14 | **DINOv3 SAT-493M (sat)** | HR | SAT RS | — | — | **86.85** | 92.85 | 85.76 | 92.20 |

> 按 mIoU 降序排列。mean = iter≥6000 后所有验证结果的均值。

---

## 基线对比 (Official DINOv3 Pretraining)

| 实验 | Checkpoint | mIoU | mFscore | Best Iter | 状态 |
|------|-----------|------|---------|-----------|------|
| **DINOv3 LVD-1689M (ImageNet)** | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | **89.00** | 94.07 | 14000 | ✅ 完成 |
| **DINOv3 SAT-493M (Remote Sensing)** | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | **86.85** | 92.85 | 22000 | ✅ 完成 |

> 基线使用官方 DINOv3 预训练权重（无 fusion/olmoearth），与本项目 zhejiang SSL 权重对比。

---

## 关键发现

1. **ImageNet LVD-1689M 最优**: 官方 ImageNet 预训练 (89.00 mIoU) 超越所有实验，但仅用 ViT HR backbone + 冻结
2. **zhejiang SSL > SAT RS**: 浙江遥感 SSL (88.43-88.49) 显著优于官方遥感 SAT-493M (86.85)，差距 ~1.6 mIoU
3. **HR > Fusion**: 纯 DINOv3 backbone 始终优于 fusion backbone，差距约 0.5-0.9 mIoU
4. **Checkpoint 越早越好**: 9999.pth 在多数实验中优于 23999/31999.pth（best mIoU 出现在更早的 iter）
5. **No Contrastive Loss 微弱优势**: nocont 在 HR 分支上有 ~0.05-0.06 mIoU 的提升；在 fusion 分支上差异不明显
6. **Context 类型无显著差异**: fusion 31999 使用 learnable masked token (87.64) vs olmoearth RGB embedding (87.64) 结果一致
7. **最佳模型**: DINOv3 LVD-1689M (89.00 mIoU / 94.07 mFscore), 其次是 nocont HR 9999 (88.49 mIoU / 93.77 mFscore)

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
| DINOv3 LVD-1689M (ImageNet) | `dinov3_m2f_ori` | `dinov3_m2f_ori.py` |
| DINOv3 SAT-493M (Remote Sensing) | `dinov3_m2f_sat` | `dinov3_m2f_sat.py` |
| gram HR 4999 | `dinov3_m2f_gram_4999` | `dinov3_m2f_gram_4999.py` |
