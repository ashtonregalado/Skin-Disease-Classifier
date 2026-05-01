import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model
from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir="SkinDisease",
    train_folder="train",
    test_folder="test",
    num_workers=0,
)
NUM_CLASSES = len(classes)

# ── Guard against missing test set ────────────────────────────────────────
if test_loader is None:
    print("No test folder found. Evaluation requires a test set.")
    exit()

# ── Load final model ──────────────────────────────────────────────────────
model = build_model(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(
    torch.load("models/best_model_final.pth", map_location=DEVICE)
)
model.eval()

# ── Run inference ─────────────────────────────────────────────────────────
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# ── Classification report ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  TEST SET RESULTS")
print("="*55)
print(classification_report(
    all_labels, all_preds,
    target_names=classes,   # uses the list returned by your get_dataloaders
    digits=3
))

# ── Confusion matrix ──────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(18, 15))
sns.heatmap(
    cm,
    annot=True, fmt='d', cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    linewidths=0.5
)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.ylabel("Actual", fontsize=12)
plt.xlabel("Predicted", fontsize=12)
plt.title("Confusion Matrix — Test Set", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()

# ── Per-class accuracy (sorted lowest to highest) ─────────────────────────
print("\nPer-class accuracy:")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for name, acc in sorted(zip(classes, per_class_acc), key=lambda x: x[1]):
    bar  = "█" * int(acc * 20)
    flag = "  ← needs attention" if acc < 0.5 else ""
    print(f"  {name:<30} {acc:.1%}  {bar}{flag}")