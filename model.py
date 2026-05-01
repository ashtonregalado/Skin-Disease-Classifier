import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=22, dropout=0.4):

    # ── Load MobileNetV2 with pretrained ImageNet weights ─────────────────
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # ── Phase A: Freeze the entire backbone ───────────────────────────────
    # We don't want to destroy the ImageNet features it already learned.
    # Only our custom head will train at first.
    for param in model.features.parameters():
        param.requires_grad = False

    # ── Replace the classifier head ───────────────────────────────────────
    # MobileNetV2's original head outputs 1000 classes (ImageNet).
    # in_features is 1280 for MobileNetV2.
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
        # No Softmax here — CrossEntropyLoss applies it internally
    )

    return model


def unfreeze_backbone(model, unfreeze_from_layer=14):
    # ── Phase B: Unfreeze later backbone layers for fine-tuning ──────────
    # Called after initial training converges.
    # MobileNetV2 has 19 feature layers (0–18).
    # We unfreeze from layer 14 onwards — the high-level feature detectors.
    for i, layer in enumerate(model.features):
        if i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen from layer {unfreeze_from_layer}. Trainable params: {trainable:,}")
    return model


if __name__ == "__main__":
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {total - trainable:,}")

    # Verify output shape
    dummy = torch.randn(4, 3, 224, 224)   # fake batch of 4 images
    out   = model(dummy)
    print(f"Output shape     : {out.shape}")   # → torch.Size([4, 22])