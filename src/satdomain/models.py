"""Model factory for the controlled architecture/pretraining comparison."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.30),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def create_model(
    architecture: str,
    num_classes: int,
    load_pretrained: bool = True,
) -> nn.Module:
    if architecture == "small_cnn":
        return SmallCNN(num_classes)

    if architecture in {"resnet18_scratch", "resnet18_pretrained"}:
        weights = (
            ResNet18_Weights.DEFAULT
            if architecture == "resnet18_pretrained" and load_pretrained
            else None
        )
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "deit_tiny_pretrained":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError("Install timm to use DeiT") from error
        return timm.create_model(
            "deit_tiny_patch16_224.fb_in1k",
            pretrained=load_pretrained,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown architecture: {architecture}")
