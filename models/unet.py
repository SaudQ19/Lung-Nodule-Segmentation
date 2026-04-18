"""
models/unet.py
--------------
2.5-D U-Net for lung nodule segmentation.

Architecture
------------
Encoder  : 3 ConvBlock stages with MaxPool2d downsampling
Bottleneck: ConvBlock with spatial dropout (p=0.3)
Decoder  : 3 transposed-conv upsampling stages with skip connections
Head     : 1×1 convolution → raw logits (1 channel)

Each ConvBlock contains two Conv2d → InstanceNorm2d → ReLU sequences.
InstanceNorm is preferred over BatchNorm because segmentation tasks
often use small per-GPU batch sizes.

Input  : (B, 2*context+1, H, W)  — default 5-channel 2.5-D stack
Output : (B, 1, H, W)            — raw logits (apply sigmoid for probabilities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double-convolution block: Conv → IN → ReLU → Conv → IN → ReLU [→ Dropout].

    Parameters
    ----------
    in_ch   : input channels
    out_ch  : output channels
    dropout : spatial dropout probability (0 = disabled)
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2p5D(nn.Module):
    """2.5-D U-Net segmentation network.

    Parameters
    ----------
    in_channels  : number of input channels (= 2*context_slices + 1)
    out_channels : number of output channels (1 for binary segmentation)
    base         : base channel width — controls model capacity
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base: int = 64,
    ) -> None:
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base,        base * 2)
        self.enc3 = ConvBlock(base * 2,    base * 4)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(base * 4, base * 8, dropout=0.3)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3  = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1  = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Conv2d(base, out_channels, kernel_size=1)

    def _match_size(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Bilinear resize *x* to match *ref*'s spatial dimensions if needed."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear",
                              align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        # Decode with skip connections
        d3 = self._match_size(self.up3(b),  e3)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self._match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self._match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.head(d1)   # raw logits — apply sigmoid externally
