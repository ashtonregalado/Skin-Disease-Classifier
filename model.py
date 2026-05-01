import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 22, dropout: float = 0.4) -> nn.Module:

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze the entire backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280 for MobileNetV2

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        # No Softmax — CrossEntropyLoss handles it internally
    )

    return model


def unfreeze_backbone(model: nn.Module, unfreeze_from_layer: int = 14) -> nn.Module:
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen from layer {unfreeze_from_layer}+. Trainable params: {trainable:,}")
    return model


if __name__ == "__main__":
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")

    dummy = torch.randn(4, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape     : {out.shape}")  # → torch.Size([4, 22])