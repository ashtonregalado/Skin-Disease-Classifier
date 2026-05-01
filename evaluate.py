import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model
from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, CLASS_NAMES, NUM_CLASSES = get_dataloaders()

# ── Load the final best model ─────────────────────────────────────────────
model = build_model(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load("models/best_model_final.pth", map_location=DEVICE))
model.eval()

# ── Run inference on test set ─────────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# ── Classification report ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  TEST SET RESULTS")
print("="*55)
print(classification_report(
    all_labels, all_preds,
    target_names=CLASS_NAMES,
    digits=3
))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm,
    annot    = True,
    fmt      = 'd',
    cmap     = 'Blues',
    xticklabels = CLASS_NAMES,
    yticklabels = CLASS_NAMES,
    linewidths  = 0.5
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.ylabel("Actual",    fontsize=12)
plt.xlabel("Predicted", fontsize=12)
plt.title("Confusion Matrix — Test Set", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()
print("Saved: models/confusion_matrix.png")

# ── Per-class accuracy ────────────────────────────────────────────────────
print("\nPer-class accuracy:")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for name, acc in sorted(
    zip(CLASS_NAMES, per_class_acc),
    key=lambda x: x[1]    # sort lowest to highest
):
    bar   = "█" * int(acc * 20)
    flag  = "  ← needs attention" if acc < 0.5 else ""
    print(f"  {name:<30} {acc:.1%}  {bar}{flag}")