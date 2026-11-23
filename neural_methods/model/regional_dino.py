"""Regional pulse estimation with optional DINOv2 context for fusion.

This is an experimental architecture, not a validated novelty claim. The
regions are fixed image-grid cells, not anatomical regions or skin masks.
DINOv2 supplies context to the reliability gate; it never predicts a waveform.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class _ChannelLayerNorm(nn.Module):
    """Normalize channels at each time independently, without batch statistics."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value):
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class _TemporalBlock(nn.Module):
    """Noncausal temporal block whose padding preserves constant sequences."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                      padding=padding, padding_mode="replicate"),
            _ChannelLayerNorm(channels),
            nn.GELU(),
            # One channel mask over the whole sequence avoids temporal noise
            # being injected into an otherwise static input during training.
            nn.Dropout1d(dropout),
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                      padding=padding, padding_mode="replicate"),
            _ChannelLayerNorm(channels),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )

    def forward(self, value):
        return value + self.net(value)


class RegionalDINO(nn.Module):
    """Estimate rPPG from local RGB changes and explicitly weighted regions.

    ``video`` is finite raw RGB in [0, 255], shaped [B, T, 3, H, W].
    Actual range verification belongs in the trainer's input audit; this model
    validates shape and finiteness without guessing the input normalization.

    The shared pulse path processes each grid crop independently. Its inputs
    include RGB, relative frame differences and unnormalized local RGB traces.
    There is no BatchNorm or waveform standard-deviation normalization.
    DINO sees sparse full frames, retains patch-grid context, and only affects
    the slowly varying reliability weights. Its patch features include global
    transformer context, so gate weights are not causal skin attributions.

    ``region_mask`` optionally marks valid grid cells with a [B, R] bool tensor.
    Every example must retain at least one region. Invalid features and
    waveforms are excluded from fusion, and their returned weights are zero.

    Returns [B, T], or, with ``return_aux=True``, a dictionary containing
    waveform, regional_waveforms, region_weights, quality_logits and
    valid_regions. The waveform is exactly the weighted sum of the returned
    regional waveforms. A shared DC correction centers that sum in time.

    Only constructing learned fusion with ``use_dino=True`` and no injected
    ``backbone`` can access torch.hub. Uniform fusion bypasses DINO entirely.
    Importing this module has no download or device effects.
    ``finetune_blocks=2`` trains only the last two blocks and final norm;
    the default zero keeps the complete DINO backbone frozen and in eval mode.
    """

    def __init__(
        self,
        use_dino=True,
        fusion="learned",
        grid_size=3,
        hidden_dim=48,
        context_dim=32,
        num_tcn_blocks=4,
        kernel_size=3,
        dropout=0.1,
        backbone_name="dinov2_vits14",
        encoder_size=140,
        context_stride=8,
        frame_chunk_size=64,
        finetune_blocks=0,
        checkpoint_backbone=True,
        backbone=None,
    ):
        super().__init__()
        for name, value, minimum in (
            ("grid_size", grid_size, 1), ("hidden_dim", hidden_dim, 2),
            ("context_dim", context_dim, 1),
            ("num_tcn_blocks", num_tcn_blocks, 1),
            ("context_stride", context_stride, 1),
            ("frame_chunk_size", frame_chunk_size, 1),
            ("finetune_blocks", finetune_blocks, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if (not isinstance(kernel_size, int) or isinstance(kernel_size, bool)
                or kernel_size < 1 or kernel_size % 2 != 1):
            raise ValueError("kernel_size must be a positive odd integer.")
        if (not isinstance(encoder_size, int) or isinstance(encoder_size, bool)
                or encoder_size < 14 or encoder_size % 14 != 0):
            raise ValueError("encoder_size must be a positive multiple of 14.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if fusion not in ("learned", "uniform"):
            raise ValueError("fusion must be 'learned' or 'uniform'.")
        if finetune_blocks and (not use_dino or fusion == "uniform"):
            raise ValueError("finetune_blocks requires learned fusion and use_dino=True.")

        self.use_dino = bool(use_dino and fusion == "learned")
        self.fusion = fusion
        self.grid_size = grid_size
        self.num_regions = grid_size ** 2
        self.hidden_dim = hidden_dim
        self.encoder_size = encoder_size
        self.context_stride = context_stride
        self.frame_chunk_size = frame_chunk_size
        self.finetune_blocks = finetune_blocks
        self.checkpoint_backbone = bool(checkpoint_backbone)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_tcn_blocks - 1)
        self.quality_smoothing = 2 * context_stride + 1

        # Small per-region images bound memory without averaging away the RGB
        # trace skip. Crops are separated before any learned spatial operation.
        self.spatial_size = 12
        self.pulse_stem = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(16, 24, 3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.pulse_projection = nn.Conv1d(24 + 6, hidden_dim, 1)
        self.temporal_encoder = nn.Sequential(*[
            _TemporalBlock(hidden_dim, kernel_size, 2 ** i, dropout)
            for i in range(num_tcn_blocks)
        ])
        self.waveform_head = nn.Conv1d(hidden_dim, 1, 1)
        # Direct access to local color traces bypasses learned image features
        # and channel normalization. The skip is learned, not amplitude-scaled.
        self.trace_skip = nn.Conv1d(6, 1, 1, bias=False)

        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.backbone = None
        self.context_projection = None
        if self.use_dino:
            self.backbone = backbone if backbone is not None else torch.hub.load(
                "facebookresearch/dinov2", backbone_name,
                pretrained=True, trust_repo=True,
            )
            feature_dim = getattr(self.backbone, "embed_dim", None)
            if not isinstance(feature_dim, int) or feature_dim < 1:
                raise ValueError("DINO backbone must expose a positive integer embed_dim.")
            self.backbone.requires_grad_(False)
            if finetune_blocks:
                blocks = getattr(self.backbone, "blocks", None)
                norm = getattr(self.backbone, "norm", None)
                if blocks is None or norm is None or finetune_blocks > len(blocks):
                    raise ValueError("finetune_blocks exceeds available backbone blocks, or final norm is missing.")
                for block in list(blocks)[-finetune_blocks:]:
                    block.requires_grad_(True)
                norm.requires_grad_(True)
            # Masked-image tokens are unused by forward_features here.
            if getattr(self.backbone, "mask_token", None) is not None:
                self.backbone.mask_token.requires_grad_(False)
            self.context_projection = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, context_dim), nn.GELU(),
            )
            self._set_backbone_mode(self.training)

        # Raw RGB and relative-change first/second moments provide local
        # quality cues; no region-aggregated feature can leak a masked crop.
        gate_dim = 12 + (context_dim if self.use_dino else 0)
        self.quality_head = nn.Sequential(
            nn.Conv1d(gate_dim, context_dim, 1), nn.GELU(),
            nn.Conv1d(context_dim, 1, 1),
        )
        # Begin learned fusion at equal weights, then learn quality differences.
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)

    def _set_backbone_mode(self, mode):
        if self.backbone is None:
            return
        # Frozen prefix and all non-block components stay deterministic.
        self.backbone.eval()
        if self.finetune_blocks and mode:
            for block in list(self.backbone.blocks)[-self.finetune_blocks:]:
                block.train(True)
            self.backbone.norm.train(True)

    def train(self, mode=True):
        super().train(mode)
        self._set_backbone_mode(mode)
        return self

    def _validate_mask(self, video, region_mask):
        batch = video.shape[0]
        if region_mask is None:
            return torch.ones(batch, self.num_regions, dtype=torch.bool, device=video.device)
        if (not isinstance(region_mask, torch.Tensor)
                or region_mask.dtype != torch.bool
                or tuple(region_mask.shape) != (batch, self.num_regions)):
            raise ValueError(f"region_mask must be a bool tensor of shape [{batch}, {self.num_regions}].")
        valid = region_mask.to(device=video.device)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("Every video must have at least one valid region.")
        return valid

    def _local_features(self, rgb):
        """Return spatial features, RGB/change traces and local quality moments."""
        batch, time, _, height, width = rgb.shape
        previous = torch.cat((rgb[:, :1], rgb[:, :-1]), dim=1)
        relative = (rgb - previous) / (rgb + previous + 1e-3)
        stem_inputs, traces, qualities = [], [], []
        for row in range(self.grid_size):
            top, bottom = row * height // self.grid_size, (row + 1) * height // self.grid_size
            for col in range(self.grid_size):
                left, right = col * width // self.grid_size, (col + 1) * width // self.grid_size
                local_rgb = rgb[:, :, :, top:bottom, left:right]
                local_diff = relative[:, :, :, top:bottom, left:right]
                trace = torch.cat((local_rgb.mean((-1, -2)), local_diff.mean((-1, -2))), dim=-1)
                traces.append(trace)
                local = torch.cat((local_rgb, local_diff), dim=2).reshape(batch * time, 6, bottom - top, right - left)
                stem_inputs.append(F.adaptive_avg_pool2d(local, self.spatial_size))
                # Moments retain brightness, saturation/heterogeneity and motion
                # cues without interpreting them as calibrated signal quality.
                qualities.append(torch.cat((
                    trace,
                    local_rgb.square().mean((-1, -2)),
                    local_diff.square().mean((-1, -2)),
                ), dim=-1))
        # [B, R, T, channels, h, w] -> shared, independent region encoder.
        crops = torch.stack([value.reshape(batch, time, 6, self.spatial_size, self.spatial_size)
                             for value in stem_inputs], dim=1)
        spatial = self.pulse_stem(crops.reshape(-1, 6, self.spatial_size, self.spatial_size))
        spatial = spatial.flatten(1).reshape(batch, self.num_regions, time, 24)
        return spatial, torch.stack(traces, dim=1), torch.stack(qualities, dim=1)

    def _encode_context_chunk(self, frames):
        frames = F.interpolate(frames, size=(self.encoder_size, self.encoder_size),
                               mode="bilinear", align_corners=False)
        output = self.backbone.forward_features((frames - self.rgb_mean) / self.rgb_std)
        tokens = output["x_norm_patchtokens"]
        if tokens.ndim != 3:
            raise ValueError("DINO patch tokens must have shape [frames, patches, channels].")
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError("Square input requires a square DINO patch-token grid.")
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
        return F.adaptive_avg_pool2d(grid, self.grid_size).flatten(2).transpose(1, 2)

    def _context_features(self, rgb, valid):
        batch, time = rgb.shape[:2]
        # A DINO token attends to the entire image. Zero invalid input cells
        # before encoding as well as excluding their output features, so a
        # masked crop cannot alter the contextual gate of a remaining region.
        height, width = rgb.shape[-2:]
        pixel_valid = rgb.new_empty(batch, 1, height, width)
        for row in range(self.grid_size):
            top, bottom = row * height // self.grid_size, (row + 1) * height // self.grid_size
            for col in range(self.grid_size):
                left, right = col * width // self.grid_size, (col + 1) * width // self.grid_size
                region = row * self.grid_size + col
                pixel_valid[:, :, top:bottom, left:right] = valid[:, region, None, None, None]
        rgb = rgb * pixel_valid[:, None]
        indices = torch.arange(0, time, self.context_stride, device=rgb.device)
        # Include the endpoint so interpolation refers to actual frame times.
        if int(indices[-1]) != time - 1:
            indices = torch.cat((indices, indices.new_tensor([time - 1])))
        sampled = rgb.index_select(1, indices).reshape(-1, *rgb.shape[2:])
        encoded = []
        for chunk in sampled.split(self.frame_chunk_size):
            if self.finetune_blocks == 0:
                with torch.no_grad():
                    features = self._encode_context_chunk(chunk)
            elif self.training and torch.is_grad_enabled() and self.checkpoint_backbone:
                features = checkpoint(self._encode_context_chunk, chunk, use_reentrant=False)
            else:
                features = self._encode_context_chunk(chunk)
            encoded.append(features)
        context = torch.cat(encoded, dim=0).reshape(batch, indices.numel(), self.num_regions, -1)
        context = self.context_projection(context).permute(0, 2, 3, 1)
        if indices.numel() == 1:
            context = context.expand(-1, -1, -1, time)
        else:
            # Linear interpolation with the exact (possibly shortened) last gap.
            target = torch.arange(time, device=rgb.device)
            right = torch.searchsorted(indices, target, right=False).clamp(1, indices.numel() - 1)
            left = right - 1
            alpha = (target - indices[left]).to(context.dtype) / (indices[right] - indices[left]).to(context.dtype)
            context = context[..., left] * (1 - alpha) + context[..., right] * alpha
        return context.masked_fill(~valid[:, :, None, None], 0)

    def forward(self, video, return_aux=False, region_mask=None):
        if (not isinstance(video, torch.Tensor) or video.ndim != 5
                or video.shape[2] != 3):
            shape = tuple(video.shape) if isinstance(video, torch.Tensor) else type(video).__name__
            raise ValueError(f"Expected raw RGB video [B, T, 3, H, W], got {shape}.")
        batch, time, _, height, width = video.shape
        if batch < 1 or time < 1 or min(height, width) < self.grid_size:
            raise ValueError("Batch/time must be nonzero and image dimensions must cover the region grid.")
        if video.is_complex() or not bool(torch.isfinite(video).all()):
            raise ValueError("video must contain finite real RGB values.")
        valid = self._validate_mask(video, region_mask)
        rgb = video.to(dtype=self.rgb_mean.dtype) / 255.0
        spatial, traces, quality = self._local_features(rgb)
        invalid = ~valid[:, :, None, None]
        spatial = spatial.masked_fill(invalid, 0)
        traces = traces.masked_fill(invalid, 0)
        quality = quality.masked_fill(invalid, 0)

        pulse_features = torch.cat((spatial, traces), dim=-1)
        # Remove static appearance, while leaving actual fluctuation amplitude.
        pulse_features = pulse_features - pulse_features.mean(dim=2, keepdim=True)
        pulse_features = pulse_features.permute(0, 1, 3, 2).reshape(batch * self.num_regions, -1, time)
        hidden = self.temporal_encoder(self.pulse_projection(pulse_features))
        regional = self.waveform_head(hidden).reshape(batch, self.num_regions, time)
        centered_traces = traces - traces.mean(dim=2, keepdim=True)
        skip = self.trace_skip(centered_traces.permute(0, 1, 3, 2).reshape(batch * self.num_regions, 6, time))
        regional = regional + skip.reshape(batch, self.num_regions, time)
        regional = regional - regional.mean(dim=-1, keepdim=True)
        regional = regional.masked_fill(~valid[:, :, None], 0)

        if self.fusion == "uniform":
            logits = regional.new_zeros(batch, self.num_regions, time)
            weights = valid[:, :, None].to(regional.dtype).expand(-1, -1, time)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            gate_features = quality.permute(0, 1, 3, 2)
            if self.use_dino:
                gate_features = torch.cat((gate_features, self._context_features(rgb, valid)), dim=2)
            logits = self.quality_head(gate_features.reshape(batch * self.num_regions, -1, time))
            pad = self.quality_smoothing // 2
            logits = F.avg_pool1d(F.pad(logits, (pad, pad), mode="replicate"),
                                  self.quality_smoothing, stride=1)
            logits = logits.reshape(batch, self.num_regions, time)
            weights = logits.masked_fill(~valid[:, :, None], -torch.inf).softmax(dim=1)
            logits = logits.masked_fill(~valid[:, :, None], 0)

        # With time-varying weights, individually centered regions can produce
        # nonzero fused DC. Shift every valid region by the same DC so the
        # returned explicit weighted sum is centered, with no amplitude scaling.
        fused_dc = (weights * regional).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        regional = (regional - fused_dc).masked_fill(~valid[:, :, None], 0)
        waveform = (weights * regional).sum(dim=1)
        if not return_aux:
            return waveform
        return {
            "waveform": waveform,
            "regional_waveforms": regional,
            "region_weights": weights,
            "quality_logits": logits,
            "valid_regions": valid,
        }
