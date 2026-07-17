"""纯 DINOv3 backbone（从 fusion checkpoint 中只取 HR ViT）+ DINOv3_Adapter → mmseg。

相比 fusion_backbone.py：去掉 olmoearth / embed_head / fusion，只保留 HR ViT +
标准 4 层 adapter 提取（无 -1 sentinel）。backbone 冻结，只训 adapter + m2f。
"""
import sys as _sys
import types as _types

_MOCKS = {
    "dinov3.models.RS_vision_transformer": {},
    "dinov3.eval.utils": {"ModelWithIntermediateLayers": _types.new_class("_MIL", ())},
}
for _name, _attrs in _MOCKS.items():
    if _name not in _sys.modules:
        _mod = _types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        _sys.modules[_name] = _mod

_LIMX_ROOT = "/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main"
if _LIMX_ROOT not in _sys.path:
    _sys.path.insert(0, _LIMX_ROOT)

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmseg.registry import MODELS

from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn


# ── mmcv MSDeformAttn shim（与 fusion_backbone.py 一致）──────────────────
class _MmcvMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model, num_levels=n_levels, num_heads=n_heads,
            num_points=n_points, batch_first=True)

    def init_weights(self):
        self.attn.init_weights()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        return self.attn(
            query=query, value=input_flatten, identity=torch.zeros_like(query),
            query_pos=None, key_padding_mask=input_padding_mask,
            reference_points=reference_points, spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index)


@MODELS.register_module()
class DINOv3BackboneMmseg(BaseModule):
    """纯 DINOv3 backbone：vit_large + 标准 DINOv3_Adapter（4 层）。

    从 fusion checkpoint 中只取 backbone.* 权重加载到 ViT。
    ViT 冻结（DINOv3_Adapter 默认 requires_grad_(False) + no_grad），
    只训练 adapter (SPM+InteractionBlocks) + m2f head。
    """

    _DEFAULT_INTERACTION_INDEXES = {
        "vit_small": [2, 5, 8, 11],
        "vit_base":  [2, 5, 8, 11],
        "vit_large": [5, 11, 17, 23],
    }

    def __init__(
        self,
        arch="vit_large",
        patch_size=16,
        interaction_indexes=None,
        n_storage_tokens=0,
        layerscale_init=1e-5,
        img_size=512,
        checkpoint=None,
        freeze_backbone=True,
        finetune_vit=False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=None)

        _factory = {"vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large}[arch]
        self.backbone = _factory(
            patch_size=patch_size, img_size=img_size,
            n_storage_tokens=n_storage_tokens, layerscale_init=layerscale_init)
        self.backbone.init_weights()

        if checkpoint:
            self._load_backbone_checkpoint(checkpoint)
            # Mark backbone as already initialized so that mmengine's
            # BaseModule.init_weights() (called by Runner at train start)
            # does NOT re-run DinoVisionTransformer.init_weights(), which
            # would overwrite the loaded checkpoint weights with
            # trunc_normal_(std=0.02) random init.
            self.backbone.is_init = True

        if interaction_indexes is None:
            interaction_indexes = self._DEFAULT_INTERACTION_INDEXES[arch]

        self.adapter = DINOv3_Adapter(
            self.backbone,
            interaction_indexes=interaction_indexes,
            with_cp=False)

        self._replace_msda_with_mmcv()

        # ── 冻结 / 全参微调控制 ──
        # DINOv3_Adapter.__init__ 默认 requires_grad_(False) + forward 用 no_grad
        # finetune_vit=True  → forward 改用 nullcontext（梯度流过 ViT）
        # freeze_backbone=False → ViT 参数 requires_grad=True
        self.adapter.finetune_vit = finetune_vit
        if not freeze_backbone or finetune_vit:
            self.backbone.requires_grad_(True)

        self.embed_dim = self.backbone.embed_dim

    def _load_backbone_checkpoint(self, checkpoint):
        """从 fusion checkpoint 中只取 backbone.* 加载到 ViT。"""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for c in ("model", "state_dict", "teacher", "student"):
                if c in ckpt and isinstance(ckpt[c], dict):
                    ckpt = ckpt[c]
                    break
        # 只保留 backbone.* 前缀，去掉前缀后加载
        sd = {}
        for k, v in ckpt.items():
            if isinstance(v, torch.Tensor) and k.startswith("backbone."):
                sd[k[len("backbone."):]] = v
        inc = self.backbone.load_state_dict(sd, strict=False)
        print(f"[DINOv3Backbone] checkpoint loaded: "
              f"matched={len(sd)-len(inc.unexpected_keys)} "
              f"missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")

    def _replace_msda_with_mmcv(self):
        for parent in self.adapter.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, MSDeformAttn):
                    rep = _MmcvMSDeformAttn(
                        d_model=child.d_model, n_levels=child.n_levels,
                        n_heads=child.n_heads, n_points=child.n_points, ratio=child.ratio)
                    rep.init_weights()
                    setattr(parent, name, rep)

    def forward(self, x):
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
