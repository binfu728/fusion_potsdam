"""Fusion backbone (DINOv3 + OlmoEarth + Fusion ViT) adapted for mmseg Mask2Former.

Key architecture:
  FusionBackboneWrapper ─ wraps HR ViT + embed_head + MultiLayerCustomEncoder,
    exposes the DINOv3_Adapter contract (embed_dim, patch_size,
    get_intermediate_layers).

  FusionDINOv3_Adapter(DINOv3_Adapter) ─ overrides forward to accept an
    optional ``olmoearth`` context that flows through to the wrapper's
    get_intermediate_layers.

  FusionBackboneMmseg(BaseModule) ─ the mmseg-registered backbone that
    instantiates 1+2, swaps native MSDeformAttn → mmcv, and returns 4
    multi-scale feature maps (strides 4/8/16/32, all ``embed_dim`` channels).

Layer selection (vit_small, 12 blocks):
  interaction_indexes = [2, 5, 8, -1]
  3 DINOv3 tap layers ([2, 5, 8]) + fusion output (-1 sentinel).

Dependencies:
  - limx dinov3 (imported via sys.path.insert(0, LIMX_ROOT)).
  - The local dinov3/ copy (non-fusion) is intentionally shadowed.
  - olmoearth_pretrain is NOT needed (stub + UPHead reimplementation).
"""

import sys as _sys
import types as _types

# ── pre-populate sys.modules to short-circuit problematic import chains ──
_MOCKS = {
    "dinov3.models.RS_vision_transformer": {},
    "dinov3.eval.utils": {"ModelWithIntermediateLayers": _types.new_class("_ModelWithIntermediateLayers", ())},
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

# ── now safe to import from limx ──
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
 
from einops import rearrange
from omegaconf import OmegaConf

from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmseg.registry import MODELS

from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
from dinov3.models.croma_vit_crosself_integration_opimize import MultiLayerCustomEncoder
from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
    DINOv3_Adapter,
    deform_inputs,
)
from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn


# ═══════════════════════════════════════════════════════════════════════════
# 1. mmcv MSDeformAttn shim (verbatim copy from mmseg_dino_agri)
# ═══════════════════════════════════════════════════════════════════════════

class _MmcvMSDeformAttn(nn.Module):
    """Drop-in replacement that delegates to mmcv's MultiScaleDeformableAttention."""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):  # noqa: ARG002
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )

    def init_weights(self):
        self.attn.init_weights()

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        return self.attn(
            query=query,
            value=input_flatten,
            identity=torch.zeros_like(query),
            query_pos=None,
            key_padding_mask=input_padding_mask,
            reference_points=reference_points,
            spatial_shapes=input_spatial_shapes,
            level_start_index=input_level_start_index,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. UPHead (MLP head: Linear + LayerNorm, matches 23999.pth embed_head)
# ═══════════════════════════════════════════════════════════════════════════

class UPHead(nn.Module):
    """Project olmoearth embedding → HR ViT embedding dim.

    Matches the 23999.pth checkpoint which stores:
      ``embed_head.decoder.0.*`` = Linear(in_dim, out_dim)
      ``embed_head.decoder.1.*`` = LayerNorm(out_dim)
    """

    def __init__(self, in_dim, out_dim, modalities=None):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(out_dim, eps=1e-6),
        )

    def forward(self, embed):
        return self.decoder(embed)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Fusion backbone wrapper
# ═══════════════════════════════════════════════════════════════════════════

_FUSION_SENTINEL = -1

class FusionBackboneWrapper(nn.Module):
    """Wraps HR ViT + embed_head + MultiLayerCustomEncoder into a single module
    that satisfies the DINOv3_Adapter backbone contract (``.embed_dim``,
    ``.patch_size``, ``.get_intermediate_layers``).

    ``get_intermediate_layers`` accepts an integer sentinel ``-1`` in the
    ``n`` list to indicate the *fusion output* slot — all other integer
    entries are treated as DINOv3 block indices.

    When ``olmoearth`` is ``None``, the wrapper feeds the fusion encoder's
    learnable ``fusion_mask_token`` as context (equivalent to "olmoearth
    dropped" in pretraining), so the backbone is callable without OlmoEarth
    data.
    """

    def __init__(
        self,
        arch="vit_small",
        patch_size=16,
        n_storage_tokens=0,
        layerscale_init=1e-5,
        mask_k_bias=False,
        fusion_cfg=None,
        olmoearth_embed=768,
        img_size=480,
        checkpoint=None,
    ):
        super().__init__()
        _factory = {"vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large}[arch]
        self.vit = _factory(
            patch_size=patch_size,
            img_size=img_size,
            n_storage_tokens=n_storage_tokens,
            layerscale_init=layerscale_init,
            mask_k_bias=mask_k_bias,
        )
        self.vit.init_weights()

        self.embed_dim = self.vit.embed_dim
        self.patch_size = patch_size
        self.n_blocks = len(self.vit.blocks)
        self.n_storage_tokens = n_storage_tokens

        fusion_cfg = fusion_cfg or {}
        self.fusion = MultiLayerCustomEncoder(
            dim=fusion_cfg.get("dim", self.embed_dim),
            depth=fusion_cfg.get("depth", 3),
            num_heads=fusion_cfg.get("num_heads", 8),
            num_patches_q=fusion_cfg.get("num_patches_q", 900),
            num_patches_kv=fusion_cfg.get("num_patches_kv", 144),
            ff_mult=fusion_cfg.get("ff_mult", 4),
        )
        self.fusion.init_weights()

        self.embed_head = UPHead(in_dim=olmoearth_embed, out_dim=self.embed_dim)

        if checkpoint:
            self.load_fusion_checkpoint(checkpoint)

    def load_fusion_checkpoint(self, checkpoint):
        """Load a released fusion SSLMetaArch_QH2 checkpoint into the wrapper.

        Checkpoint is a flat state_dict with prefixes:
          ``backbone.*``         → ``self.vit.*``
          ``embed_head.*``       → ``self.embed_head.*``
          ``fusion_backbone.*``  → ``self.fusion.*``
        Keys are cleaned of ``model.``/``student.``/``_orig_mod.`` prefixes.
        """
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

        # unwrap nested container (model / state_dict / teacher / student)
        if isinstance(ckpt, dict):
            for container in ("model", "state_dict", "teacher", "student"):
                if container in ckpt and isinstance(ckpt[container], dict):
                    ckpt = ckpt[container]
                    break
            # else: assume flat top-level tensor dict

        def _remap(k):
            k = (k.replace("_orig_mod.", "")
                  .replace("_checkpoint_wrapped_module.", ""))
            for p in ("model.student.", "student.", "model."):
                if k.startswith(p):
                    k = k[len(p):]
                    break
            if k.startswith("backbone."):
                return "vit." + k[len("backbone."):]
            if k.startswith("fusion_backbone."):
                return "fusion." + k[len("fusion_backbone."):]
            # embed_head.* stays as-is
            return k

        remapped = {_remap(k): v for k, v in ckpt.items()
                    if isinstance(v, torch.Tensor)}

        # ── 处理 fusion_mask_token 形状不匹配 ─────────────────────────────────
        # MultiLayerCustomEncoder 初始化 fusion_mask_token 为 (1,1,dim)，
        # 但 checkpoint 中为 (1,num_patches_kv,dim)。在 load_state_dict
        # 之前将模型的参数 resize 到 checkpoint 的形状，避免 RuntimeError。
        ckpt_fusion_mask = remapped.get("fusion.fusion_mask_token")
        if ckpt_fusion_mask is not None:
            model_param = self.fusion.fusion_mask_token
            if ckpt_fusion_mask.shape != model_param.shape:
                new_token = ckpt_fusion_mask.clone().to(model_param.device)
                self.fusion.fusion_mask_token = nn.Parameter(new_token)

        incomp = self.load_state_dict(remapped, strict=False)
        print(f"[load_fusion_checkpoint] "
              f"matched={len(remapped) - len(incomp.unexpected_keys)} "
              f"missing={len(incomp.missing_keys)} "
              f"unexpected={len(incomp.unexpected_keys)}")
        return incomp

    def get_intermediate_layers(
        self,
        x,
        *,
        n=1,
        reshape=False,
        return_class_token=False,
        return_extra_tokens=False,
        norm=True,
        olmoearth=None,
    ):
        if isinstance(n, (list, tuple)):
            n_list = list(n)
        else:
            n_list = [n]

        # partition into DINO block indices vs fusion sentinel
        dino_idx = sorted(set(i for i in n_list if isinstance(i, int) and 0 <= i < self.n_blocks))
        fusion_positions = {k for k, i in enumerate(n_list) if isinstance(i, int) and i == _FUSION_SENTINEL}
        has_fusion = len(fusion_positions) > 0

        # collect indices the ViT must produce (plus last block for fusion input)
        viT_needed = set(dino_idx)
        if has_fusion:
            viT_needed.add(self.n_blocks - 1)
        viT_needed = sorted(viT_needed)

        # ── run HR ViT once ──
        vit_layers = self.vit.get_intermediate_layers(
            x, n=viT_needed, return_class_token=True, norm=norm, reshape=False,
        )
        # vit_layers: tuple of (patches, cls) ordered by viT_needed
        idx_to_layer = dict(zip(viT_needed, vit_layers))

        # ── fusion output ──
        fusion_feat = fusion_cls = None
        if has_fusion:
            g_patch, _ = idx_to_layer[self.n_blocks - 1]
            B = g_patch.shape[0]

            if olmoearth is not None:
                context = self.embed_head(olmoearth)          # [B,H,W,1024]
                context = rearrange(context, "b h w d -> b (h w) d")
                fusion_feat = self.fusion(g_patch, context)
            else:
                # olmoearth=None (mmseg calls backbone(x) with image only):
                # pass context=None → fusion encoder uses its own learnable
                # fusion_mask_token (loaded from checkpoint) as the KV context,
                # equivalent to "olmoearth dropped" in pretraining.
                fusion_feat = self.fusion(g_patch, context=None)
            fusion_cls = fusion_feat.mean(dim=1)

        # ── assemble in original order ──
        outputs = []
        class_tokens = []
        extra_tokens_list = []

        for k, idx in enumerate(n_list):
            if k in fusion_positions:
                outputs.append(fusion_feat)
                class_tokens.append(fusion_cls)
            else:
                p, c = idx_to_layer[idx]
                outputs.append(p)
                class_tokens.append(c)
            extra_tokens_list.append(torch.empty(
                0, self.embed_dim, device=outputs[0].device,
            ).expand(outputs[0].shape[0], -1, -1))

        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        if return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        if not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens_list))
        # both
        return tuple(zip(outputs, class_tokens, extra_tokens_list))


# ═══════════════════════════════════════════════════════════════════════════
# 5. Fusion adapter (subclass of DINOv3_Adapter, adds olmoearth interface)
# ═══════════════════════════════════════════════════════════════════════════

class FusionDINOv3_Adapter(DINOv3_Adapter):
    """Same as DINOv3_Adapter except ``forward`` accepts ``olmoearth``."""

    def forward(self, x, olmoearth=None):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        c = torch.cat([c2, c3, c4], dim=1)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, _, _, _ = x.shape

        grad_ctx = (
            contextlib.nullcontext()
            if getattr(self, "finetune_vit", False)
            else torch.no_grad()
        )
        with torch.autocast("cuda", torch.bfloat16), grad_ctx:
            all_layers = self.backbone.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True, olmoearth=olmoearth,
            )

        x_for_shape, _ = all_layers[0]
        bs, _, dim = x_for_shape.shape
        del x_for_shape

        outs = list()
        for i, layer in enumerate(self.interactions):
            x_patch, cls_tok = all_layers[i]
            _, c, _ = layer(
                x_patch, c, cls_tok,
                deform_inputs1, deform_inputs2,
                H_c, W_c, H_toks, W_toks,
            )
            outs.append(x_patch.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        c2_p = c[:, 0:c2.size(1), :]
        c3_p = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4_p = c[:, c2.size(1) + c3.size(1):, :]

        c2_p = c2_p.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3_p = c3_p.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4_p = c4_p.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2_p) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2_p, c3_p, c4_p = c1 + x1, c2_p + x2, c3_p + x3, c4_p + x4

        f1 = self.norm1(c1)
        f2 = self.norm2(c2_p)
        f3 = self.norm3(c3_p)
        f4 = self.norm4(c4_p)

        return {"1": f1, "2": f2, "3": f3, "4": f4}


# ═══════════════════════════════════════════════════════════════════════════
# 6. mmseg-registered backbone
# ═══════════════════════════════════════════════════════════════════════════

@MODELS.register_module()
class FusionBackboneMmsegV2(BaseModule):
    """Fusion backbone v2 — registered as an mmseg MODEL.

    v2 changes vs v1:
      - UPHead now has Linear + LayerNorm (matches 23999.pth embed_head).
      - OlmoEarth stub removed; when olmoearth=None the fusion encoder
        uses its own learnable fusion_mask_token (loaded from checkpoint).
      - Expects n_storage_tokens=4 (matches 23999.pth).

    Returns a tuple of 4 feature maps (strides 4/8/16/32), all
    ``embed_dim`` channels.
    """

    _DEFAULT_INTERACTION_INDEXES = {
        "vit_small": [2, 5, 8, _FUSION_SENTINEL],
        "vit_base":  [2, 5, 8, _FUSION_SENTINEL],
        "vit_large": [5, 11, 17, _FUSION_SENTINEL],
    }

    def __init__(
        self,
        arch="vit_small",
        patch_size=16,
        interaction_indexes=None,
        n_storage_tokens=0,
        layerscale_init=1e-5,
        mask_k_bias=False,
        fusion_cfg=None,
        olmoearth_embed=768,
        img_size=480,
        checkpoint=None,
        freeze_backbone=False,
        finetune_vit=False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=None)

        self.wrapper = FusionBackboneWrapper(
            arch=arch,
            patch_size=patch_size,
            n_storage_tokens=n_storage_tokens,
            layerscale_init=layerscale_init,
            mask_k_bias=mask_k_bias,
            fusion_cfg=fusion_cfg,
            olmoearth_embed=olmoearth_embed,
            img_size=img_size,
            checkpoint=checkpoint,
        )

        if interaction_indexes is None:
            interaction_indexes = self._DEFAULT_INTERACTION_INDEXES.get(
                arch, [2, 5, 8, _FUSION_SENTINEL]
            )

        self.adapter = FusionDINOv3_Adapter(
            self.wrapper,
            interaction_indexes=interaction_indexes,
            with_cp=False,
        )

        self._replace_msda_with_mmcv()

        self.adapter.finetune_vit = finetune_vit
        if not freeze_backbone or finetune_vit:
            self.adapter.backbone.requires_grad_(True)
        for name, param in self.adapter.named_parameters():
            if not freeze_backbone or "backbone" not in name:
                param.requires_grad = True

        self.embed_dim = self.wrapper.embed_dim

    def _replace_msda_with_mmcv(self):
        for parent in self.adapter.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, MSDeformAttn):
                    replacement = _MmcvMSDeformAttn(
                        d_model=child.d_model,
                        n_levels=child.n_levels,
                        n_heads=child.n_heads,
                        n_points=child.n_points,
                        ratio=child.ratio,
                    )
                    replacement.init_weights()
                    setattr(parent, name, replacement)

    def forward(self, x, olmoearth=None):
        out = self.adapter(x, olmoearth=olmoearth)
        return (out["1"], out["2"], out["3"], out["4"])
