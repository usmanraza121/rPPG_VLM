import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm across channels independently at each time step."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B, C, D] -> [B, D, C] -> [B, C, D]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualTCNBlock(nn.Module):
    """Two dilated temporal convolutions with a residual connection."""

    def __init__(
        self,
        channels,
        dilation,
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        # Symmetric padding preserves sequence length.
        padding = dilation * (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            ChannelLayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            ChannelLayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class DINOTCN(nn.Module):
    """
    DINOv2 + TCN model for video-based rPPG waveform estimation.

    Input:
        video: [B, D, 3, H, W]

    Output:
        rppg: [B, D]

    Default architecture:
        DINOv2 ViT-S/14
        -> spatial mean of patch tokens: 384 features/frame
        -> projection: 384 -> 128
        -> six residual TCN blocks, dilations [1, 2, 4, 8, 16, 32]
        -> 1x1 temporal convolution
        -> rPPG waveform

    input_format:
        "rgb01"    : RGB values in [0, 1].
        "rgb255"   : RGB values in [0, 255], including uint8.
        "imagenet" : RGB already normalized with ImageNet mean/std.

    freeze_backbone:
        True  : train projection, TCN, and waveform head.
        False : also fine-tune DINOv2.

    This is a noncausal model intended for complete video clips.
    """

    def __init__(
        self,
        backbone_name="dinov2_vits14",
        pretrained=True,
        freeze_backbone=True,
        hidden_dim=128,
        num_tcn_blocks=6,
        kernel_size=3,
        dropout=0.1,
        encoder_size=140,
        frame_chunk_size=32,
        # input_format="rgb01",
        input_format="rgb255",
        checkpoint_backbone=True,
    ):
        super().__init__()

        if hidden_dim < 2:
            raise ValueError("hidden_dim must be >= 2.")

        if num_tcn_blocks < 1:
            raise ValueError("num_tcn_blocks must be >= 1.")

        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be >= 1.")

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        if encoder_size < 14 or encoder_size % 14 != 0:
            raise ValueError("encoder_size must be a positive multiple of 14.")

        if input_format not in ("rgb01", "rgb255", "imagenet"):
            raise ValueError(
                "input_format must be 'rgb01', 'rgb255', or 'imagenet'."
            )

        if freeze_backbone and not pretrained:
            raise ValueError(
                "A frozen backbone must use pretrained weights."
            )

        self.freeze_backbone = freeze_backbone
        self.encoder_size = encoder_size
        self.frame_chunk_size = frame_chunk_size
        self.input_format = input_format
        self.checkpoint_backbone = checkpoint_backbone

        # First construction downloads the repository and pretrained weights.
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=pretrained,
            trust_repo=True,
        )

        self.backbone.requires_grad_(not freeze_backbone)

        # This model never uses masked image tokens.
        # Freeze the unused parameter, including when fine-tuning with DDP.
        if getattr(self.backbone, "mask_token", None) is not None:
            self.backbone.mask_token.requires_grad_(False)

        if freeze_backbone:
            self.backbone.eval()

        feature_dim = self.backbone.embed_dim

        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )

        self.tcn = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=hidden_dim,
                    dilation=2 ** i,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for i in range(num_tcn_blocks)
            ]
        )

        # Unrestricted scalar prediction at each time step.
        self.head = nn.Conv1d(
            hidden_dim,
            1,
            kernel_size=1,
        )

        # Two convolutions per block; default receptive field = 253 frames.
        self.receptive_field = (
            1
            + 2 * (kernel_size - 1) * (2 ** num_tcn_blocks - 1)
        )

    def train(self, mode=True):
        super().train(mode)

        # Calling model.train() must not reactivate a frozen backbone.
        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def _encode_chunk(self, frames):
        """
        frames: [N, 3, H, W]
        returns: [N, feature_dim]
        """
        frames = frames.to(dtype=self.rgb_mean.dtype)

        if self.input_format == "rgb255":
            frames = frames / 255.0

        if frames.shape[-2:] != (self.encoder_size, self.encoder_size):
            frames = F.interpolate(
                frames,
                size=(self.encoder_size, self.encoder_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.input_format != "imagenet":
            frames = (frames - self.rgb_mean) / self.rgb_std

        features = self.backbone.forward_features(frames)

        # [N, number_of_patches, feature_dim]
        patch_tokens = features["x_norm_patchtokens"]

        # Average spatial patches, preserving one feature vector per frame.
        return patch_tokens.mean(dim=1)

    def forward(self, video):
        B, D, C, H, W = video.shape
        if C > 3:
                video = video[:, :, 3:, :, :] # use last 3 channels
                    # print(f' x-shape: {x.shape}')
                B, D, C, H, W = video.shape
        
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(
                f"Expected [B, D, 3, H, W], got {tuple(video.shape)}."
            )

        B, D, C, H, W = video.shape

        if min(B, D, H, W) < 1:
            raise ValueError("Input dimensions must be nonzero.")

        # Flatten only batch and time; frame order is preserved.
        frames = video.reshape(B * D, C, H, W)

        frame_features = []

        # Resize and encode a limited number of frames at a time.
        for chunk in frames.split(self.frame_chunk_size, dim=0):
            if self.freeze_backbone:
                # Keep gradients enabled for the projection and TCN below.
                with torch.no_grad():
                    features = self._encode_chunk(chunk)

            elif (
                self.training
                and torch.is_grad_enabled()
                and self.checkpoint_backbone
            ):
                # Non-reentrant checkpointing supports parameter gradients
                # even when input video.requires_grad is False.
                features = checkpoint(
                    self._encode_chunk,
                    chunk,
                    use_reentrant=False,
                )

            else:
                features = self._encode_chunk(chunk)

            frame_features.append(features)

        # [B*D, feature_dim] -> [B, D, feature_dim]
        features = torch.cat(frame_features, dim=0).reshape(B, D, -1)

        # [B, D, feature_dim] -> [B, D, hidden_dim]
        features = self.projection(features)

        # Conv1d expects [B, channels, time].
        features = features.transpose(1, 2).contiguous()

        # Temporal length remains D throughout the TCN.
        features = self.tcn(features)          # [B, hidden_dim, D]
        rppg = self.head(features).squeeze(1)   # [B, D]

        return rppg

if __name__ == "__main__":
    # Test the DINOv2TCN model with dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DINOTCN(
    freeze_backbone=True,
    input_format="rgb01",
    frame_chunk_size=32,
    )
    
    model.to(device)
    # Dummy input: [B, D, C, H, W]
    dummy_input = torch.randn(2, 160, 3, 128, 128)
    dummy_input = dummy_input.to(device)

    # Forward pass
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Expected: [B, D]