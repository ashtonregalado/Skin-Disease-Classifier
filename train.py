import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import build_model, unfreeze_backbone
from dataset import get_dataloaders   # your dataset.py

os.makedirs("models", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Matches your dataset.py signature exactly ─────────────────────────────
train_loader, val_loader, test_loader, classes = get_dataloaders(
    data_dir="SkinDisease",
    train_folder="train",
    test_folder="test",
    batch_size=32,
    val_split=0.15,
    num_workers=0,
    img_size=224,
)

NUM_CLASSES = len(classes)   # derived from the returned list, not a separate return value
print(f"Classes ({NUM_CLASSES}): {classes}")

USE_CLASS_WEIGHTS = False

# ── Class counts in the same order as classes list ────────────────────────
# Order must match ImageFolder's alphabetical class ordering
class_count_map = {
    'Acne': 593, 'Actinic_Keratosis': 748, 'Benign_tumors': 1093,
    'Bullous': 504, 'Candidiasis': 248, 'DrugEruption': 547,
    'Eczema': 1010, 'Infestations_Bites': 524, 'Lichen': 553,
    'Lupus': 311, 'Moles': 361, 'Psoriasis': 820,
    'Rosacea': 254, 'Seborrh_Keratoses': 455, 'SkinCancer': 693,
    'Sun_Sunlight_Damage': 312, 'Tinea': 923, 'Unknown_Normal': 1651,
    'Vascular_Tumors': 543, 'Vasculitis': 461, 'Vitiligo': 714,
    'Warts': 580
}

# Use the exact order that ImageFolder assigned (alphabetical = classes list)
class_counts = [class_count_map[c] for c in classes]
total_images = sum(class_counts)

if USE_CLASS_WEIGHTS:
    class_weights = torch.tensor(
        [total_images / (NUM_CLASSES * c) for c in class_counts],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Weighted loss ready.")
else:
    criterion = nn.CrossEntropyLoss()
    print("Unweighted loss ready.")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total * 100


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total * 100

def run_training(
    model, train_loader, val_loader, criterion,
    optimizer, scheduler, device,
    num_epochs, save_path, early_stop_patience=7
):
    best_val_loss    = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch [{epoch+1:02d}/{num_epochs}]  "
            f"Train — Loss: {train_loss:.4f}  Acc: {train_acc:.1f}%  |  "
            f"Val — Loss: {val_loss:.4f}  Acc: {val_acc:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    return history

if __name__ == "__main__":

    # ── Phase A: Train head only ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  PHASE A — Head only (backbone frozen)")
    print("="*55)

    model = build_model(num_classes=NUM_CLASSES, dropout=0.4).to(DEVICE)

    optimizer_A = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler_A = ReduceLROnPlateau(
        optimizer_A, mode='min', patience=3, factor=0.5
    )

    history_A = run_training(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        criterion          = criterion,
        optimizer          = optimizer_A,
        scheduler          = scheduler_A,
        device             = DEVICE,
        num_epochs         = 18,
        save_path          = "models/best_model_phaseA.pth",
        early_stop_patience= 6
    )

    # ── Phase B: Fine-tune with unfrozen backbone ─────────────────────────
    print("\n" + "="*55)
    print("  PHASE B — Fine-tuning (backbone partially unfrozen)")
    print("="*55)

    # Always load Phase A best weights before unfreezing
    model.load_state_dict(
        torch.load("models/best_model_phaseA.pth", map_location=DEVICE)
    )
    model = unfreeze_backbone(model, unfreeze_from_layer=10)

    optimizer_B = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5    # gentler fine-tuning for the frozen-to-unfrozen transition
    )
    scheduler_B = ReduceLROnPlateau(
        optimizer_B, mode='min', patience=3, factor=0.5)

    history_B = run_training(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        criterion          = criterion,
        optimizer          = optimizer_B,
        scheduler          = scheduler_B,
        device             = DEVICE,
        num_epochs         = 25,
        save_path          = "models/best_model_final.pth",
        early_stop_patience= 8
    )

    # ── Save combined history and plot ────────────────────────────────────
    combined = {
        'train_loss': history_A['train_loss'] + history_B['train_loss'],
        'val_loss'  : history_A['val_loss']   + history_B['val_loss'],
        'train_acc' : history_A['train_acc']  + history_B['train_acc'],
        'val_acc'   : history_A['val_acc']    + history_B['val_acc'],
    }

    with open("models/history.json", "w") as f:
        json.dump(combined, f, indent=2)

    phase_A_len = len(history_A['train_loss'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (ax1, 'loss', 'Loss'),
        (ax2, 'acc',  'Accuracy (%)')
    ]:
        ax.plot(combined[f'train_{metric}'], label=f'Train')
        ax.plot(combined[f'val_{metric}'],   label=f'Val')
        ax.axvline(
            phase_A_len - 1, color='gray',
            linestyle='--', linewidth=1, label='Phase B starts'
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("models/training_curves.png", dpi=150)
    plt.show()
    print("\nTraining complete. Final model saved to models/best_model_final.pth")