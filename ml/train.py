"""
train.py
--------
Trains the DenseNet-121 chest X-ray classifier.

Usage:
    python ml/train.py --data-dir data --epochs 5 --batch-size 16 \
        --out models/pneumonia_densenet121.pth

If data/train contains no images, this will refuse to run silently on an
empty dataset. First run:
    python ml/generate_synthetic_data.py
(or place the real Chest X-Ray Images (Pneumonia) dataset in data/).
"""

import argparse
import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(__file__))
from model_def import build_model, save_checkpoint, IMG_SIZE  # noqa: E402


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/pneumonia_densenet121.pth")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir) or not any(os.scandir(train_dir)):
        print("[error] No training data found. Run `python ml/generate_synthetic_data.py` "
              "first, or populate data/train with the real dataset.", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")

    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))
    class_names = train_ds.classes
    print(f"[info] Classes: {class_names}")
    print(f"[info] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            # class_names is patched onto the checkpoint via save_checkpoint's extra dict
            save_checkpoint(model, args.out, extra={"class_names": class_names,
                                                      "best_val_acc": best_val_acc})

    with open(os.path.join(os.path.dirname(args.out), "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[done] Best val accuracy: {best_val_acc:.4f}. Model saved to {args.out}")


if __name__ == "__main__":
    main()
