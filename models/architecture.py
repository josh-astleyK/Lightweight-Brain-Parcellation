"""
UMamba architecture variants for 134-class whole-brain parcellation.

Three models are provided:
  build_base()  — UMamba (baseline), filters (32,64,128,256,256,256), ~44 M params
  build_fwr()   — UMamba-FWR (filter-width reduced), filters (16,32,64,128,128,128), ~11 M params
  build_lite()  — UMamba-lite (FWR + squeeze-and-excitation), ~11 M params

All three share the same UMambaBot backbone: a 6-stage residual U-Net encoder
with a Mamba SSM bottleneck and symmetric residual decoder.  The Mamba layer
operates in channel-token mode at the 5³ bottleneck (160 / 2⁵ = 5).
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

try:
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD
    from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
except ImportError as e:
    raise ImportError(
        "dynamic-network-architectures is required.  "
        "Install with: pip install dynamic-network-architectures\n"
        f"Original error: {e}"
    )

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError(
        "mamba-ssm is required.  "
        "Install with: pip install causal-conv1d mamba-ssm\n"
        f"Original error: {e}"
    )

NUM_CLASSES = 134
PATCH = (160, 160, 160)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock3d(nn.Module):
    """3-D channel squeeze-and-excitation gate (reduction=16, floor at 4)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        w = self.pool(x).view(B, C)
        w = self.fc(w).view(B, C, 1, 1, 1)
        return x * w


class UpsampleLayer(nn.Module):
    def __init__(self, conv_op, input_channels, output_channels, scale):
        super().__init__()
        self.conv  = conv_op(input_channels, output_channels, kernel_size=1)
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.conv(x)


class MambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, channel_token: bool = False):
        super().__init__()
        self.dim = dim
        self.channel_token = channel_token
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _forward_patch_token(self, x):
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        return self.mamba(self.norm(x_flat)).transpose(-1, -2).reshape(B, C, *img_dims)

    def _forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        return self.mamba(self.norm(x_flat)).reshape(B, n_tokens, *img_dims)

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        if self.channel_token:
            return self._forward_channel_token(x)
        return self._forward_patch_token(x)


class BasicResBlock(nn.Module):
    """Two-conv residual block with optional 1×1 projection and SE gate."""

    def __init__(self, conv_op, input_channels, output_channels,
                 norm_op, norm_op_kwargs,
                 kernel_size=3, padding=1, stride=1,
                 use_1x1conv=False,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 use_se: bool = False, se_reduction: int = 16):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        self.conv1 = conv_op(input_channels, output_channels, kernel_size,
                             stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        self.conv3 = (conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
                      if use_1x1conv else None)
        self.se = SEBlock3d(output_channels, reduction=se_reduction) if use_se else None

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


class UNetResEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides, n_blocks_per_stage,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, return_skips=False,
                 use_se: bool = False):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]
        stem_channels = features_per_stage[0]

        self.stem = nn.Sequential(
            BasicResBlock(conv_op, input_channels, stem_channels,
                          norm_op, norm_op_kwargs,
                          kernel_size=kernel_sizes[0],
                          padding=self.conv_pad_sizes[0],
                          stride=1, use_1x1conv=True,
                          nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                          use_se=use_se),
            *[BasicBlockD(conv_op, stem_channels, stem_channels,
                          kernel_sizes[0], stride=1, conv_bias=conv_bias,
                          norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                          nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
              for _ in range(n_blocks_per_stage[0] - 1)],
        )
        input_channels = stem_channels

        stages = []
        for s in range(n_stages):
            stages.append(nn.Sequential(
                BasicResBlock(conv_op, input_channels, features_per_stage[s],
                              norm_op, norm_op_kwargs,
                              kernel_size=kernel_sizes[s],
                              padding=self.conv_pad_sizes[s],
                              stride=strides[s], use_1x1conv=True,
                              nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                              use_se=use_se),
                *[BasicBlockD(conv_op, features_per_stage[s], features_per_stage[s],
                              kernel_sizes[s], stride=1, conv_bias=conv_bias,
                              norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                              nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)
                  for _ in range(n_blocks_per_stage[s] - 1)],
            ))
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class UNetResDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage, use_se: bool = False):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_enc = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_enc - 1)

        stages, upsample_layers, seg_layers = [], [], []
        for s in range(1, n_stages_enc):
            ch_below = encoder.output_channels[-s]
            ch_skip = encoder.output_channels[-(s + 1)]
            stride_up = encoder.strides[-s]
            upsample_layers.append(
                UpsampleLayer(encoder.conv_op, ch_below, ch_skip, stride_up))
            stages.append(nn.Sequential(
                BasicResBlock(encoder.conv_op, 2 * ch_skip, ch_skip,
                              encoder.norm_op, encoder.norm_op_kwargs,
                              kernel_size=encoder.kernel_sizes[-(s + 1)],
                              padding=encoder.conv_pad_sizes[-(s + 1)],
                              stride=1, use_1x1conv=True,
                              nonlin=encoder.nonlin,
                              nonlin_kwargs=encoder.nonlin_kwargs,
                              use_se=use_se),
                *[BasicBlockD(encoder.conv_op, ch_skip, ch_skip,
                              encoder.kernel_sizes[-(s + 1)], stride=1,
                              conv_bias=encoder.conv_bias,
                              norm_op=encoder.norm_op,
                              norm_op_kwargs=encoder.norm_op_kwargs,
                              nonlin=encoder.nonlin,
                              nonlin_kwargs=encoder.nonlin_kwargs)
                  for _ in range(n_conv_per_stage[s - 1] - 1)],
            ))
            seg_layers.append(encoder.conv_op(ch_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.deep_supervision = False

    def forward(self, skips):
        lres = skips[-1]
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres)
            x = torch.cat((x, skips[-(s + 2)]), dim=1)
            x = self.stages[s](x)
            lres = x
        return self.seg_layers[-1](lres)


class UMambaBot(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage,
                 conv_op, kernel_sizes, strides, n_conv_per_stage,
                 num_classes, n_conv_per_stage_decoder,
                 bottleneck_spatial_size,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 use_se: bool = False):
        super().__init__()
        self.encoder = UNetResEncoder(
            input_channels, n_stages, features_per_stage,
            conv_op, kernel_sizes, strides, n_conv_per_stage,
            conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            return_skips=True, use_se=use_se,
        )
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder,
                                      use_se=use_se)

        bottleneck_channels = features_per_stage[-1] if isinstance(features_per_stage, (list, tuple)) \
            else features_per_stage
        n_spatial_tokens = int(np.prod(bottleneck_spatial_size))
        # If there are fewer spatial positions than channels, scan over channels instead of space.
        # For a 160³ input with 6 stages: bottleneck is 5³ = 125 positions vs 128/256 channels.
        channel_token = n_spatial_tokens <= bottleneck_channels
        mamba_dim = n_spatial_tokens if channel_token else bottleneck_channels
        self.mamba_layer = MambaLayer(dim=mamba_dim, channel_token=channel_token)

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def _build(filters, num_classes, patch_size, use_se):
    n_stages = 6
    strides  = [(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1)   # first stage no downsampling
    kernels  = [(3, 3, 3)] * n_stages
    bottleneck_size = tuple(p // (2 ** (n_stages - 1)) for p in patch_size)  # 5³ for 160³ input
    return UMambaBot(
        input_channels=1, n_stages=n_stages,
        features_per_stage=list(filters), conv_op=nn.Conv3d,
        kernel_sizes=kernels, strides=strides,
        n_conv_per_stage=2, num_classes=num_classes,
        n_conv_per_stage_decoder=2, bottleneck_spatial_size=bottleneck_size,
        conv_bias=True, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        use_se=use_se,
    )


def build_base(num_classes: int = NUM_CLASSES, patch_size: tuple = PATCH) -> UMambaBot:
    """UMamba baseline — filters (32,64,128,256,256,256), ~44 M parameters."""
    return _build((32, 64, 128, 256, 256, 256), num_classes, patch_size, use_se=False)


def build_fwr(num_classes: int = NUM_CLASSES, patch_size: tuple = PATCH) -> UMambaBot:
    """UMamba-FWR — filter-width reduced (16,32,64,128,128,128), ~11 M parameters."""
    return _build((16, 32, 64, 128, 128, 128), num_classes, patch_size, use_se=False)


def build_lite(num_classes: int = NUM_CLASSES, patch_size: tuple = PATCH) -> UMambaBot:
    """UMamba-lite — FWR + squeeze-and-excitation, ~11 M parameters."""
    return _build((16, 32, 64, 128, 128, 128), num_classes, patch_size, use_se=True)
