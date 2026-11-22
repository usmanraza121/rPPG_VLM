
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DINOv2 VISUAL FEATURE EXTRACTOR
# ============================================================

class DINOv2FeatureExtractor(nn.Module):
    """
    Extract frame-level visual features using pretrained DINOv2.

    Input:
        [N, 3, H, W]

    Output:
        [N, 2 * embed_dim]
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        frame_chunk_size=32,
    ):
        super().__init__()

        self.frame_chunk_size = frame_chunk_size
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        # Load pretrained DINOv2
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )

        self.embed_dim = self.backbone.embed_dim

        # Freeze backbone initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Optionally unfreeze the last transformer blocks
            if unfreeze_last_n_blocks > 0:
                for block in self.backbone.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True

                for param in self.backbone.norm.parameters():
                    param.requires_grad = True

        # Learnable spatial attention over patch tokens
        self.spatial_attention = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, x):
        """
        x: [N, 3, H, W]
        """

        features = []

        for start in range(0, x.shape[0], self.frame_chunk_size):

            frame_batch = x[
                start:start + self.frame_chunk_size
            ]
            # Resize frames to 126x126 for DINOv2
            frame_batch = F.interpolate(
                        frame_batch,
                        size=(126, 126),
                        mode="bilinear",
                        align_corners=False,
                                    )

            # Frozen backbone does not need gradient computation
            if self.freeze_backbone and self.unfreeze_last_n_blocks == 0:

                with torch.no_grad():
                    output = self.backbone.forward_features(
                        frame_batch
                    )

            else:
                output = self.backbone.forward_features(
                    frame_batch
                )

            # [N, embed_dim]
            cls_token = output["x_norm_clstoken"]

            # [N, num_patches, embed_dim]
            patch_tokens = output["x_norm_patchtokens"]

            # Learnable spatial attention
            attention = self.spatial_attention(patch_tokens)

            # [N, num_patches, 1]
            attention = torch.softmax(attention, dim=1)

            # [N, embed_dim]
            spatial_features = torch.sum(
                attention * patch_tokens,
                dim=1,
            )

            # Combine global and spatial representations
            frame_features = torch.cat(
                [cls_token, spatial_features],
                dim=-1,
            )

            features.append(frame_features)

        return torch.cat(features, dim=0)


# ============================================================
# 2. TEMPORAL CONVOLUTION BLOCK
# ============================================================

class TemporalBlock(nn.Module):
    """
    Dilated residual temporal convolution block.

    Input/output:
        [B, C, T]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp1 = padding

        self.norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp2 = padding

        self.norm2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.residual(x)

        out = self.conv1(x)

        if self.chomp1 > 0:
            out = out[:, :, :-self.chomp1]

        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)

        if self.chomp2 > 0:
            out = out[:, :, :-self.chomp2]

        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)

        return self.activation(out + residual)


# ============================================================
# 3. TEMPORAL CONVOLUTIONAL NETWORK
# ============================================================

class TCN(nn.Module):
    """
    Dilated residual TCN.

    Input:
        [B, C, T]

    Output:
        [B, C_out, T]
    """

    def __init__(
        self,
        input_channels=512,
        channels=(512, 512, 512, 512, 512, 512),
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        layers = []

        in_channels = input_channels

        dilations = [2 ** i for i in range(len(channels))]

        for out_channels, dilation in zip(channels, dilations):

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

            in_channels = out_channels

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
# 4. COMPLETE DINOV2-TCN rPPG MODEL
# ============================================================

# class DINOv2TCN(nn.Module):
    """
    DINOv2 + TCN for remote photoplethysmography estimation.

    Input:
        [B, D, C, H, W]

    Output:
        [B, D]

    Example:
        Input:  [16, 160, 3, 128, 128]
        Output: [16, 160]
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        feature_dim=512,
        tcn_channels=(512, 512, 512, 512, 512, 512),
        kernel_size=3,
        dropout=0.2,
        frame_chunk_size=32,
    ):
        super().__init__()

        # ----------------------------------------------------
        # DINOv2
        # ----------------------------------------------------

        self.visual_encoder = DINOv2FeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            frame_chunk_size=frame_chunk_size,
        )

        embed_dim = self.visual_encoder.embed_dim

        # ----------------------------------------------------
        # Feature projection
        # CLS + spatial attention features
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(

            nn.Linear(embed_dim * 2, feature_dim),

            nn.LayerNorm(feature_dim),

            nn.GELU(),

            nn.Dropout(dropout),
        )

        # ----------------------------------------------------
        # Temporal Convolutional Network
        # ----------------------------------------------------

        self.temporal_encoder = TCN(
            input_channels=feature_dim,
            channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # rPPG regression head
        # ----------------------------------------------------

        self.regression_head = nn.Sequential(

            nn.Conv1d(
                in_channels=tcn_channels[-1],
                out_channels=128,
                kernel_size=1,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=128,
                out_channels=1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        """
        Args:
            x: [B, D, C, H, W]

        Returns:
            predicted_rppg: [B, D]
        """

        B, D, C, H, W = x.shape
        # print(f'Input shape: {x.shape}')
        if C > 3:
            x = x[:, :, 3:, :, :] # use last 3 channels
            # print(f' x-shape: {x.shape}')
        B, D, C, H, W = x.shape

        if C != 3:
            raise ValueError(
                f"Expected RGB input with 3 channels, got {C}"
            )

        # ----------------------------------------------------
        # Step 1: Flatten temporal dimension
        # ----------------------------------------------------

        # [B, D, C, H, W] -> [B*D, C, H, W]

        x = x.reshape(B * D, C, H, W)

        # ----------------------------------------------------
        # Step 2: DINOv2 feature extraction
        # ----------------------------------------------------

        # [B*D, 2*embed_dim]

        visual_features = self.visual_encoder(x)

        # ----------------------------------------------------
        # Step 3: Feature projection
        # ----------------------------------------------------

        # [B*D, feature_dim]

        visual_features = self.feature_projection(
            visual_features
        )

        # Restore temporal dimension
        # [B*D, F] -> [B, D, F]

        visual_features = visual_features.reshape(
            B, D, -1
        )

        # ----------------------------------------------------
        # Step 4: Prepare TCN input
        # ----------------------------------------------------

        # [B, D, F] -> [B, F, D]

        temporal_features = visual_features.transpose(1, 2)

        # ----------------------------------------------------
        # Step 5: Temporal feature learning
        # ----------------------------------------------------

        temporal_features = self.temporal_encoder(
            temporal_features
        )

        # ----------------------------------------------------
        # Step 6: rPPG regression
        # ----------------------------------------------------

        predicted_rppg = self.regression_head(
            temporal_features
        )

        # [B, 1, D] -> [B, D]

        predicted_rppg = predicted_rppg.squeeze(1)

        return predicted_rppg

from .dinotcn import DINOTCN
DINOv2TCN = DINOTCN

if __name__ == "__main__":
    # Test the DINOv2TCN model with dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DINOv2TCN(
        model_name="dinov2_vitb14",
        pretrained=True,
        freeze_backbone=True,
        unfreeze_last_n_blocks=0,
        feature_dim=512,
        tcn_channels=(512, 512, 512, 512, 512, 512),
        kernel_size=3,
        dropout=0.2,
        frame_chunk_size=32,
    )
    model.to(device)
    # Dummy input: [B, D, C, H, W]
    dummy_input = torch.randn(2, 160, 3, 128, 128)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expected: [B, D]