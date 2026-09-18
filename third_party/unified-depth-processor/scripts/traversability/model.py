"""通行性预测模型。

Architecture:
    RGB (B, 3, H, W)  +  azimuth_map (B, 1, H, W)   [H, W divisible by 16]
      -> DINOv3 ViT-L/16 backbone -> patch tokens (B, h*w, 1024)
         (optionally: multi-scale tokens from blocks 6,12,18,24 fused via FPN)
      -> Reshape to 2D feature map (B, 1024, h, w)   [h=H/16, w=W/16]
      -> Lightweight CNN decoder -> (B, 128, h*4, w*4)
      -> Angle projector (4-head cross-attention with azimuth encoding) -> (B, 360, 64)
      -> Radar distance head -> radar_dist (B, 360) + exist_logit (B, 360)

The azimuth map provides per-pixel horizontal angle from camera intrinsics,
enabling the cross-attention to learn geometry-aware angle-to-spatial mapping
across pinhole, fisheye, and equirectangular cameras.

Input resolution is adaptive: the dataset auto-computes (H, W) from the
native aspect ratio scaled to --img-size (long edge), never upscaling,
rounded to the nearest multiple of 16.

Multi-scale mode (--multi-scale): extracts tokens from blocks 6, 12, 18, 24
and fuses them with a lightweight FPN (lateral 1x1 convs + top-down addition)
before the decoder, providing richer local detail for near-obstacle boundary
estimation.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone


# ---------------------------------------------------------------------------
# Multi-scale FPN fusion
# ---------------------------------------------------------------------------

class MultiScaleFusion(nn.Module):
    """Lightweight FPN that fuses tokens from 4 backbone blocks into one map.

    Each scale's tokens are reshaped to (B, embed_dim, h, w), projected to
    fpn_dim via a 1x1 conv, then fused top-down (coarse-to-fine addition).
    The result is a single (B, embed_dim, h, w) feature map at the finest
    spatial resolution (same as the input patch grid).

    Args:
        embed_dim: Backbone token dimension (1024 for ViT-L/16).
        fpn_dim: Internal FPN channel dimension (default 256).
        num_scales: Number of scales to fuse (default 4).
    """

    def __init__(self, embed_dim: int = 1024, fpn_dim: int = 256, num_scales: int = 4):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.num_scales = num_scales

        # Lateral 1x1 projections: embed_dim -> fpn_dim for each scale
        self.lateral = nn.ModuleList([
            nn.Conv2d(embed_dim, fpn_dim, 1) for _ in range(num_scales)
        ])
        # Output projection: fpn_dim -> embed_dim to keep decoder interface unchanged
        self.out_proj = nn.Conv2d(fpn_dim, embed_dim, 1)

    def forward(self, multi_scale_tokens: List[torch.Tensor], h: int, w: int) -> torch.Tensor:
        """Fuse multi-scale tokens into a single 2D feature map.

        Args:
            multi_scale_tokens: List of (B, h*w, embed_dim) tensors from
                coarsest to finest (blocks 6, 12, 18, 24).
            h: Spatial height of patch grid (H_img / patch_size).
            w: Spatial width of patch grid (W_img / patch_size).

        Returns:
            (B, embed_dim, h, w) fused feature map.
        """
        # Reshape all scales to 2D and apply lateral projections
        maps = []
        for i, tokens in enumerate(multi_scale_tokens):
            feat = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
            maps.append(self.lateral[i](feat))  # (B, fpn_dim, h, w)

        # Top-down fusion: start from coarsest (index 0), add to finer scales
        out = maps[0]
        for i in range(1, self.num_scales):
            out = out + maps[i]

        return self.out_proj(out)  # (B, embed_dim, h, w)


class LightweightDecoder(nn.Module):
    """CNN decoder that upsamples patch features to higher resolution.

    in_dim -> 256 -> out_dim with 2x bilinear upsample at each stage.
    Default: 1024 -> 256 -> 128 (4x spatial upscale total).
    Uses GroupNorm (groups=32) instead of BatchNorm for stability at small batch sizes.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 128,
        hidden_dim: int = 256,
        use_dropout_slot: bool = False,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ]
        if use_dropout_slot:
            layers.append(nn.Dropout2d(p=0.0))
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1),
            nn.GroupNorm(min(32, out_dim), out_dim),
            nn.GELU(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AngleProjector(nn.Module):
    """Projects 2D spatial features to per-angle features via multi-head cross-attention.

    Uses learnable angle queries with sinusoidal position encoding
    to attend over spatial feature maps.  When an azimuth map is provided,
    it is encoded and added to the spatial features before key/value
    projection, giving the attention geometric awareness of which image
    regions correspond to which angular directions.

    Args:
        spatial_dim: Input spatial feature dimension.
        angle_dim: Output per-angle feature dimension.
        num_angles: Number of angular bins (default 360).
        num_heads: Number of attention heads (default 4).
    """

    def __init__(self,
                 spatial_dim: int = 128,
                 angle_dim: int = 64,
                 num_angles: int = 360,
                 num_heads: int = 4):
        super().__init__()
        self.num_angles = num_angles
        self.angle_dim = angle_dim
        self.spatial_dim = spatial_dim
        self.num_heads = num_heads
        assert angle_dim % num_heads == 0, "angle_dim must be divisible by num_heads"
        self.head_dim = angle_dim // num_heads

        # Angle queries (learnable + sinusoidal position encoding)
        self.angle_queries = nn.Parameter(
            torch.randn(num_angles, angle_dim) * 0.02
        )
        self.register_buffer(
            'angle_pos',
            self._sinusoidal_encoding(num_angles, angle_dim),
        )

        # Azimuth encoder: maps (B, 1, H, W) azimuth map to (B, spatial_dim, H, W)
        self.azimuth_encoder = nn.Sequential(
            nn.Conv2d(1, spatial_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(spatial_dim // 4, spatial_dim, 1),
        )

        # Multi-head key / value / query projections
        self.key_proj = nn.Linear(spatial_dim, angle_dim)
        self.value_proj = nn.Linear(spatial_dim, angle_dim)
        self.query_proj = nn.Linear(angle_dim, angle_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(angle_dim, angle_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(angle_dim),
            nn.GELU(),
        )

    @staticmethod
    def _sinusoidal_encoding(n: int, dim: int) -> torch.Tensor:
        angles = torch.arange(n, dtype=torch.float32) / n * 2 * math.pi
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32)
            * -(math.log(10000.0) / (half - 1))
        )
        outer = angles.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(outer), torch.cos(outer)], dim=-1)

    def forward(self,
                feat_2d: torch.Tensor,
                azimuth_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            feat_2d: (B, C, H, W) spatial features from decoder.
            azimuth_map: (B, 1, H_img, W_img) per-pixel azimuth in radians.
                Resized to (H, W) internally.  None to skip azimuth encoding.

        Returns:
            (B, num_angles, angle_dim) per-angle features.
        """
        B, C, H, W = feat_2d.shape

        # Add azimuth encoding to spatial features
        if azimuth_map is not None:
            az = F.interpolate(
                azimuth_map, size=(H, W),
                mode='bilinear', align_corners=False,
            )  # (B, 1, H, W)
            az_enc = self.azimuth_encoder(az)  # (B, spatial_dim, H, W)
            feat_2d = feat_2d + az_enc

        feat_flat = feat_2d.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        S = H * W

        keys = self.key_proj(feat_flat)      # (B, S, angle_dim)
        values = self.value_proj(feat_flat)  # (B, S, angle_dim)

        queries = self.query_proj(
            self.angle_queries + self.angle_pos
        ).unsqueeze(0).expand(B, -1, -1)    # (B, num_angles, angle_dim)

        # Reshape to multi-head: (B, num_heads, seq, head_dim)
        def split_heads(t, seq):
            return t.view(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(queries, self.num_angles)  # (B, H, num_angles, head_dim)
        k = split_heads(keys, S)                   # (B, H, S, head_dim)
        v = split_heads(values, S)                 # (B, H, S, head_dim)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, num_angles, S)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, num_angles, head_dim)
        # Merge heads: (B, num_angles, angle_dim)
        out = out.transpose(1, 2).contiguous().view(B, self.num_angles, self.angle_dim)

        return self.out_proj(out)


class TraversabilityModel(nn.Module):
    """End-to-end traversability prediction model.

    Input:  RGB image (B, 3, H, W) where H, W are divisible by 16,
            and optional azimuth_map (B, 1, H, W).
    Output: radar_dist (B, 360), exist_logit (B, 360), raw_dist (B, 360).

    Args:
        backbone_path: Path to DINOv3 pretrained weights.
        backbone_type: ViT variant to load (e.g. 'vitb16', 'vitl16').
        freeze_backbone: Freeze backbone parameters.
        unfreeze_last_n: Unfreeze last N backbone blocks (when frozen).
        decoder_dim: Decoder output channel dimension.
        angle_dim: Per-angle feature dimension.
        num_angles: Number of angular bins.
        use_multi_scale: If True, extract tokens from evenly-spaced quarter
            blocks and fuse with a lightweight FPN before the decoder.
    """

    def __init__(self,
                 backbone_path: str,
                 backbone_type: str = 'vitb16',
                 freeze_backbone: bool = True,
                 unfreeze_last_n: int = 0,
                 decoder_dim: int = 128,
                 decoder_hidden_dim: int = 256,
                 angle_dim: int = 64,
                 num_angles: int = 360,
                 use_multi_scale: bool = True,
                 fpn_dim: int = 256,
                 decoder_dropout: bool = False):
        super().__init__()
        self.num_angles = num_angles
        self.use_multi_scale = use_multi_scale

        # Backbone
        self.backbone = DINOv3Backbone(
            weights_path=backbone_path,
            model_name=backbone_type,
            freeze=freeze_backbone,
            unfreeze_last_n=unfreeze_last_n,
        )
        embed_dim = self.backbone.embed_dim
        n_blocks = self.backbone.n_blocks

        # MS block indices: evenly-spaced quarter points (0-based)
        # e.g. vitb16 (12 blocks) -> (2,5,8,11); vitl16 (24) -> (5,11,17,23)
        self._ms_block_indices = tuple(n_blocks // 4 * i - 1 for i in range(1, 5))

        # Optional multi-scale FPN fusion
        if use_multi_scale:
            self.ms_fusion = MultiScaleFusion(
                embed_dim=embed_dim,
                fpn_dim=fpn_dim,
                num_scales=len(self._ms_block_indices),
            )

        # Decoder
        self.decoder = LightweightDecoder(
            embed_dim,
            decoder_dim,
            hidden_dim=decoder_hidden_dim,
            use_dropout_slot=decoder_dropout,
        )

        # Angle projector (with azimuth encoding)
        self.angle_projector = AngleProjector(decoder_dim, angle_dim, num_angles)

        # Radar distance prediction head
        self.radar_dist_mlp = nn.Sequential(
            nn.Linear(angle_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.dist_out = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.exist_logit = nn.Linear(64, 1)  # raw logit, no sigmoid

    def forward(self, x: torch.Tensor, azimuth_map: torch.Tensor = None):
        """
        Args:
            x: RGB images (B, 3, H, W).
            azimuth_map: Per-pixel azimuth (B, 1, H, W) in radians.
                None to skip azimuth encoding (backward compatible).

        Returns:
            radar_dist:  (B, 360) combined obstacle distance (for inference).
            exist_logit: (B, 360) obstacle existence logit (pre-sigmoid).
            raw_dist:    (B, 360) raw distance from Softplus (for training loss).
        """
        B = x.shape[0]
        H_img, W_img = x.shape[2], x.shape[3]
        patch_size = self.backbone.patch_size
        h, w = H_img // patch_size, W_img // patch_size

        # Backbone: extract features (no_grad when fully frozen)
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if backbone_frozen else torch.enable_grad():
            if self.use_multi_scale:
                ms_tokens = self.backbone.forward_multi_scale(
                    x, block_indices=self._ms_block_indices
                )  # list of 4 x (B, h*w, 1024)
                feat_2d = self.ms_fusion(ms_tokens, h, w)  # (B, 1024, h, w)
            else:
                patch_tokens = self.backbone(x)  # (B, h*w, 1024)
                feat_2d = patch_tokens.transpose(1, 2).reshape(B, -1, h, w)

        # Decode
        feat_dec = self.decoder(feat_2d)  # (B, decoder_dim, h*4, w*4)

        # Angle projection (with azimuth encoding)
        angle_feat = self.angle_projector(feat_dec, azimuth_map)

        # Radar distance head
        radar_feat = self.radar_dist_mlp(angle_feat)       # (B, 360, 64)
        raw_dist = self.dist_out(radar_feat).squeeze(-1)   # (B, 360)
        exist_logit = self.exist_logit(radar_feat).squeeze(-1)  # (B, 360)

        # Return raw_dist directly (no combination logic)
        # Downstream applications can use exist_logit separately if needed
        return raw_dist, exist_logit, raw_dist
