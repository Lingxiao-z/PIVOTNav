"""对比基线模型，用于与 TraversabilityModel 进行消融实验对比。

Two lightweight baseline models sharing the same forward() signature as
TraversabilityModel so they can be swapped in via train.py --model flag
with zero changes to the training loop, loss, or dataset.

ResNetFC:
    ResNet-50 (torchvision ImageNet pretrained) + Global Average Pooling
    + FC head -> (raw_dist, exist_logit, raw_dist).
    End-to-end baseline with no geometric awareness; used to validate
    whether the DINOv3 + cross-attention design is necessary.

DINOv3MLP:
    Same DINOv3 backbone as TraversabilityModel, but replaces the
    AngleProjector (cross-attention over 360 angle queries) with a simple
    mean-pool + MLP.  Used as ablation to isolate the contribution of the
    geometry-aware AngleProjector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone


# ---------------------------------------------------------------------------
# ResNet-FC baseline
# ---------------------------------------------------------------------------

class ResNetFC(nn.Module):
    """ResNet-50 + Global Average Pooling + FC prediction head.

    Accepts azimuth_map for API compatibility but ignores it.

    Args:
        freeze_backbone: If True, freeze all ResNet parameters.
        num_angles: Number of angular output bins (default 360).
        pretrained: Load ImageNet pretrained weights via torchvision.
    """

    def __init__(self,
                 freeze_backbone: bool = True,
                 num_angles: int = 360,
                 pretrained: bool = True):
        super().__init__()
        self.num_angles = num_angles

        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)
        # Remove the classification head; keep up to avgpool
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = resnet.avgpool  # AdaptiveAvgPool2d(1)
        feat_dim = 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Single linear block that outputs dist + exist concatenated
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GroupNorm(16, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_angles * 2),
        )

    def forward(self,
                x: torch.Tensor,
                azimuth_map: torch.Tensor = None) -> tuple:
        """
        Args:
            x: RGB images (B, 3, H, W).
            azimuth_map: Ignored (API compatibility with TraversabilityModel).

        Returns:
            Tuple of (raw_dist, exist_logit, raw_dist):
                raw_dist:    (B, num_angles) Softplus obstacle distance.
                exist_logit: (B, num_angles) existence logit (pre-sigmoid).
        """
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if backbone_frozen else torch.enable_grad():
            feat = self.backbone(x)         # (B, 2048, h, w)
            feat = self.pool(feat)          # (B, 2048, 1, 1)
            feat = feat.flatten(1)          # (B, 2048)

        out = self.head(feat)               # (B, num_angles * 2)
        raw_dist = F.softplus(out[:, :self.num_angles])   # (B, num_angles)
        exist_logit = out[:, self.num_angles:]             # (B, num_angles)

        return raw_dist, exist_logit, raw_dist


# ---------------------------------------------------------------------------
# DINOv3-MLP baseline
# ---------------------------------------------------------------------------

class DINOv3MLP(nn.Module):
    """DINOv3 backbone + token mean-pool + MLP prediction head.

    Uses the same DINOv3Backbone as TraversabilityModel but replaces the
    AngleProjector (geometry-aware cross-attention) with a simple global
    mean-pool followed by an MLP.  Accepts azimuth_map for API compatibility
    but ignores it.

    Args:
        weights_path: Path to DINOv3 pretrained weights (.pth).
        backbone_type: ViT variant (e.g. 'vitb16', 'vitl16').
        freeze_backbone: If True, freeze backbone parameters.
        unfreeze_last_n: Unfreeze last N transformer blocks.
        num_angles: Number of angular output bins (default 360).
    """

    def __init__(self,
                 weights_path: str,
                 backbone_type: str = 'vitb16',
                 freeze_backbone: bool = True,
                 unfreeze_last_n: int = 0,
                 num_angles: int = 360):
        super().__init__()
        self.num_angles = num_angles

        self.backbone = DINOv3Backbone(
            weights_path=weights_path,
            model_name=backbone_type,
            freeze=freeze_backbone,
            unfreeze_last_n=unfreeze_last_n,
        )
        embed_dim = self.backbone.embed_dim

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_angles * 2),
        )

    def forward(self,
                x: torch.Tensor,
                azimuth_map: torch.Tensor = None) -> tuple:
        """
        Args:
            x: RGB images (B, 3, H, W).
            azimuth_map: Ignored (API compatibility with TraversabilityModel).

        Returns:
            Tuple of (raw_dist, exist_logit, raw_dist):
                raw_dist:    (B, num_angles) Softplus obstacle distance.
                exist_logit: (B, num_angles) existence logit (pre-sigmoid).
        """
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        with torch.no_grad() if backbone_frozen else torch.enable_grad():
            tokens = self.backbone(x)           # (B, h*w, embed_dim)

        feat = tokens.mean(dim=1)               # (B, embed_dim)
        out = self.head(feat)                   # (B, num_angles * 2)
        raw_dist = F.softplus(out[:, :self.num_angles])   # (B, num_angles)
        exist_logit = out[:, self.num_angles:]             # (B, num_angles)

        return raw_dist, exist_logit, raw_dist
