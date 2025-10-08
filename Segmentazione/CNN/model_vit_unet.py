# CNN/model_vit_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
"""
Questo file contiene la ViT-Unet.
Funzioni Principali:
Encoder (ViT): Definisce come utilizzare il modello Vision Transformer (ViT) pre-addestrato per analizzare un'immagine
di input e capirne le caratteristiche principali.

Decoder (CNN): Definisce la parte del modello che prende le informazioni dall'encoder e, passo dopo passo, costruisce 
la maschera di segmentazione finale."""

class ConvBlock(nn.Module):
    """Standard convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling block: Transposed Convolution -> Concatenate -> ConvBlock"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size mismatch between upsampled tensor and skip connection
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ViT_Unet(nn.Module):
    def __init__(self, num_classes=3, pretrained_model='vit_small_patch16_224.augreg_in21k_ft_in1k'):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = (224, 224)
        self.patch_size = (16, 16)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])

        # --- Encoder (Vision Transformer) ---
        self.encoder = timm.create_model(
            pretrained_model,
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3]  # Requesting features from 4 stages
        )

        encoder_channels = self.encoder.feature_info.channels()
        print(f"ViT Encoder feature channels: {encoder_channels}")

        # --- Bridge ---
        bridge_in_channels = encoder_channels[-1]
        self.bridge = ConvBlock(bridge_in_channels, 512)

        # --- Decoder ---
        self.up1 = UpBlock(512, encoder_channels[2], 256)
        self.up2 = UpBlock(256, encoder_channels[1], 128)
        self.up3 = UpBlock(128, encoder_channels[0], 64)

        # This final upsampling block will bring the resolution up.
        # The scale factor might need adjustment depending on the ViT architecture.
        # For a patch size of 16, the first feature map is 224/4 = 56x56
        # So we upsample 56 -> 112 -> 224
        self.up_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(32, 16)
        )

        # --- Output Layer ---
        self.outc = nn.Conv2d(16, self.num_classes, kernel_size=1)

    def reshape_if_needed(self, x):
        # Checks if the feature map is a sequence of tokens and reshapes it
        if len(x.shape) == 3:  # Shape is (batch_size, num_tokens, channels)
            b, n, c = x.shape
            # Assumes a square feature map
            h, w = self.grid_size
            x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x

    def forward(self, x):
        # --- Encoder Path ---
        if x.shape[2:] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=True)

        encoder_features = self.encoder(x)

        # --- MODIFICATION HERE ---
        # Reshape each feature map only if it's in the token format (3D)
        s1 = self.reshape_if_needed(encoder_features[0])
        s2 = self.reshape_if_needed(encoder_features[1])
        s3 = self.reshape_if_needed(encoder_features[2])
        s4 = self.reshape_if_needed(encoder_features[3])

        # --- Bridge ---
        bridge = self.bridge(s4)

        # --- Decoder Path ---
        d1 = self.up1(bridge, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        d4 = self.up_final(d3)

        # --- Output ---
        logits = self.outc(d4)

        # Final upscale to match the original input size, ensuring compatibility
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)

        return logits


if __name__ == '__main__':
    # Test the model build
    test_tensor = torch.randn(2, 3, 224, 224)  # Batch size 2, 3 channels, 224x224
    model = ViT_Unet(num_classes=3)
    output = model(test_tensor)
    print("Model built successfully!")
    print(f"Input shape: {test_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 3, 224, 224)