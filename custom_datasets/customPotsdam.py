import cv2
import numpy as np
from pathlib import Path
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS, DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@TRANSFORMS.register_module()
class RandomColorJitter(BaseTransform):
    """对 RGB 图像做随机亮度/对比度/饱和度扰动（在 CustomNormalize 之前运行）。"""
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results
        img = results["img"].astype(np.float32)

        # Brightness
        if self.brightness > 0:
            b = np.random.uniform(-self.brightness, self.brightness) * 255
            img = np.clip(img + b, 0, 255)

        # Contrast
        if self.contrast > 0:
            mean = img.mean()
            c = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            img = np.clip((img - mean) * c + mean, 0, 255)

        # Saturation (gray blend)
        if self.saturation > 0:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = np.stack([gray] * 3, axis=-1).astype(np.float32)
            s = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            img = np.clip(gray * (1 - s) + img * s, 0, 255)

        results["img"] = img.astype(np.uint8)
        return results


@TRANSFORMS.register_module()
class LoadCustomRaster(BaseTransform):
    """第一步加载：读取、BGR→RGB、Resize + 标签重映射。

    标签重映射规则（potsdam 原始标签 0-6）：
      0 → 255 (ignore)
      1 → 0  (impervious_surface)
      2 → 1  (building)
      3 → 2  (low_vegetation)
      4 → 3  (tree)
      5 → 4  (car)
      6 → 255 (ignore)
    """
    # 查找表：原始标签 0-6 → 训练标签（255=ignore）
    _REMAP_LUT = np.array([255, 0, 1, 2, 3, 4, 255], dtype=np.int64)

    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_path = results['seg_map_path']
        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)

        ori_h, ori_w = img.shape[:2]

        if self.img_size != ori_h or self.img_size != ori_w:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(ann, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 标签重映射：0/6→255(ignore)，1-5→0-4
        gt_seg_map = self._REMAP_LUT[ann.astype(np.int64)]

        results["img"] = img
        results["gt_seg_map"] = gt_seg_map
        results["img_shape"] = (self.img_size, self.img_size)
        results["ori_shape"] = (self.img_size, self.img_size)
        results["seg_fields"] = results.get("seg_fields", []) + ["gt_seg_map"]

        return results


@TRANSFORMS.register_module()
class CustomRandomRotate90(BaseTransform):
    """强数据增强：随机 90/180/270 度旋转"""
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results

        k = np.random.randint(1, 4)
        results["img"] = np.ascontiguousarray(np.rot90(results["img"], k))

        for key in results.get("seg_fields", []):
            results[key] = np.ascontiguousarray(np.rot90(results[key], k))

        return results


@TRANSFORMS.register_module()
class CustomNormalize(BaseTransform):
    """内置 potsdam 全局均值/方差，放在加载器最后一步直接计算。"""
    def __init__(self):
        # 来自 potsdam_norm.txt（R/G/B 顺序，对应 RGB）
        self.mean = np.array([97.61828308705, 92.50345435337714, 85.8699012576488], dtype=np.float32)
        self.std  = np.array([36.295481104983764, 35.3808408869616, 36.78625007116312], dtype=np.float32)

    def transform(self, results: dict) -> dict:
        img = results["img"].astype(np.float32)
        results["img"] = (img - self.mean) / self.std
        return results


@TRANSFORMS.register_module()
class LoadOlmoEarthEmbedding(BaseTransform):
    """从 .npy 加载 olmoearth 预计算 embedding。

    路径规则：{embed_root}/{split}/{basename}.npy
    basename = img_path 去掉前缀和 .png 后缀的部分。
    embedding shape: (H_emb, W_emb, 768)，通常为 (128, 128, 768)。
    """
    def __init__(self, embed_root: str):
        self.embed_root = embed_root

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        basename = Path(img_path).stem   # e.g. "2_10_0_0_512_512"
        npy_path = Path(self.embed_root) / f"{basename}.npy"
        embed = np.load(str(npy_path)).astype(np.float32)   # (H, W, 768)
        results['olmoearth_embedding'] = embed
        return results


@DATASETS.register_module()
class CustomPotsdamDataset(BaseSegDataset):
    """Potsdam 数据集：img_dir/{split}/*.png, ann_dir/{split}/*.png。

    目录结构：
        data_root/
        ├── img_dir/{train,val}/<basename>.png
        ├── ann_dir/{train,val}/<basename>.png
        ├── train.txt
        └── val.txt
    split ∈ {train, val}。标注文件名与图像相同（均为 .png）。
    """

    METAINFO = dict(
        classes=['impervious_surface', 'building', 'low_vegetation', 'tree', 'car'],
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0]],
    )

    def __init__(self, data_root: str, split: str = 'train', pipeline=None, **kwargs):
        self._custom_root = Path(data_root)
        self._split = split

        super().__init__(
            data_root=data_root,
            ann_file="",
            img_suffix='.png',
            seg_map_suffix='.png',
            pipeline=pipeline,
            **kwargs)

    def load_data_list(self) -> list:
        txt_path = self._custom_root / f"{self._split}.txt"

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        img_dir = self._custom_root / "img_dir" / self._split
        ann_dir = self._custom_root / "ann_dir" / self._split

        samples = []
        for basename in lines:
            img_path = img_dir / f"{basename}{self.img_suffix}"
            ann_path = ann_dir / f"{basename}{self.seg_map_suffix}"

            if img_path.exists() and ann_path.exists():
                samples.append({
                    'img_path': str(img_path),
                    'seg_map_path': str(ann_path),
                    'label_map': self.label_map,
                    'reduce_zero_label': self.reduce_zero_label,
                    'seg_fields': []
                })
        return samples
