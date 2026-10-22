from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):

    def __init__(self, kernels=(7, 15, 31), branch_dim=16, out_dim=64, dropout=0.1):
        super().__init__()
        kernels = tuple(int(k) for k in kernels)
        if not kernels:
            raise ValueError("At least one temporal kernel is required")

        self.kernels = kernels
        self.branches = nn.ModuleList()
        for k in kernels:
            if k <= 0:
                raise ValueError(f"Temporal kernel must be positive, got {k}")
            total_pad = k - 1
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            self.branches.append(nn.Sequential(
                nn.ZeroPad2d((pad_left, pad_right, 0, 0)),
                nn.Conv2d(1, branch_dim, kernel_size=(1, k), padding=0, bias=False),
                nn.BatchNorm2d(branch_dim),
                nn.GELU(),
            ))
        in_dim = branch_dim * len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, C, T] -> [B, 1, C, T] -> [B, D, C, T]
        x = x.unsqueeze(1)
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class TemporalTokenReducer(nn.Module):

    def __init__(self, target_tokens: int):
        super().__init__()
        self.target_tokens = int(target_tokens)

    def forward(self, x):
        # x: [B, D, C, T]
        if self.target_tokens <= 0 or self.target_tokens >= x.shape[-1]:
            return x
        time_points = x.shape[-1]
        if time_points % self.target_tokens == 0:
            factor = time_points // self.target_tokens
            return F.avg_pool2d(x, kernel_size=(1, factor), stride=(1, factor))
        return F.adaptive_avg_pool2d(x, output_size=(x.shape[-2], self.target_tokens))


class SpatialEncoder(nn.Module):
    def __init__(self, num_channels, in_dim=64, embed_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, kernel_size=(num_channels, 1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # [B, D, C, T'] -> [B, E, 1, T'] -> [B, T', E]
        x = self.net(x).squeeze(2)
        return x.transpose(1, 2)


class EEGCrossSubjectNet(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        time_segments: int = 400,
        transformer_tokens: int = 100,
        temporal_kernels=(7, 15, 31),
        temporal_branch_dim: int = 16,
        temporal_dim: int = 64,
        embed_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % transformer_heads != 0:
            raise ValueError("embed_dim must be divisible by transformer_heads")
        if time_segments <= 0:
            raise ValueError("time_segments must be positive")
        if transformer_tokens < 0:
            raise ValueError("transformer_tokens must be >= 0 (0 disables reduction)")
        if transformer_tokens > time_segments:
            raise ValueError(
                f"transformer_tokens ({transformer_tokens}) cannot exceed "
                f"time_segments ({time_segments})"
            )

        self.num_channels = num_channels
        self.time_segments = time_segments
        self.transformer_tokens = time_segments if transformer_tokens == 0 else transformer_tokens

        self.temporal = MultiScaleTemporalEncoder(
            kernels=temporal_kernels,
            branch_dim=temporal_branch_dim,
            out_dim=temporal_dim,
            dropout=dropout,
        )
        self.token_reducer = TemporalTokenReducer(self.transformer_tokens)
        self.spatial = SpatialEncoder(num_channels, temporal_dim, embed_dim, dropout)


        self.pos = nn.Parameter(torch.zeros(1, time_segments, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _position_encoding(self, token_count: int):
        if token_count == self.pos.shape[1]:
            return self.pos
        # [1, T, E] -> [1, E, T] -> interpolate -> [1, T', E]
        return F.interpolate(
            self.pos.transpose(1, 2),
            size=token_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def encode(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}")
        if x.shape[2] != self.time_segments:
            raise ValueError(f"Expected {self.time_segments} time points, got {x.shape[2]}")

        h = self.temporal(x)          # [B, D, C, 400]
        h = self.token_reducer(h)     # [B, D, C, 100] by default
        h = self.spatial(h)           # [B, 100, E]
        h = h + self._position_encoding(h.shape[1])
        h = self.transformer(h)
        z = self.norm(h.mean(dim=1))
        return z

    def forward(self, x, return_z=False):
        z = self.encode(x)
        logits = self.classifier(z)
        return (logits, z) if return_z else logits

    def classify_z(self, z):
        return self.classifier(z)
