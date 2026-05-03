import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 22, dropout: float = 0.4) -> nn.Module:
    """
    Build a MobileNetV2-based classifier for skin disease prediction.

    - Loads pretrained MobileNetV2
    - Freezes backbone feature extractor
    - Replaces classifier head with a custom fully connected network

    Args:
        num_classes (int): Number of output classes
        dropout (float): Dropout rate for regularization

    Returns:
        nn.Module: Configured model ready for training
    """
    # Load pretrained MobileNetV2 weights
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze backbone to train only the classifier initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Get input size of original classifier layer
    in_features = model.classifier[1].in_features  # typically 1280

    # Replace classifier with custom head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),  
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),  
    )

    return model


def unfreeze_backbone(model: nn.Module, unfreeze_from_layer: int = 14) -> nn.Module:
    """
    Unfreeze later layers of the backbone for fine-tuning.

    Args:
        model (nn.Module): Model with frozen backbone
        unfreeze_from_layer (int): Layer index to start unfreezing

    Returns:
        nn.Module: Model with partially unfrozen backbone
    """
    # Enable gradients for deeper layers only
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    # Report how many parameters will be updated
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen from layer {unfreeze_from_layer}+. Trainable params: {trainable:,}")

    return model


if __name__ == "__main__":
    # Build model and inspect parameter counts
    model = build_model()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")

    # Run a dummy forward pass to verify output shape
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)

    print(f"Output shape     : {out.shape}")  # expected: [batch_size, num_classes]