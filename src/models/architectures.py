import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AnisotropicUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.enc2 = ConvBlock3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.enc3 = ConvBlock3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.bottleneck = ConvBlock3D(f * 4, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec3 = ConvBlock3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = ConvBlock3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock3D(f * 2, f)

        self.out_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)