import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Windows stdout encoding fix
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASICSR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASICSR_ROOT not in sys.path:
    sys.path.insert(0, BASICSR_ROOT)

from basicsr.archs.class_sr_classifier_arch import ClassSRClassifier


class LPQualityDataset(Dataset):
    """Dataset for training ClassSR quality classifier (Clear vs Degraded)."""
    def __init__(self, data_root, train_file, img_size=(32, 96), max_samples=None):
        self.samples = []
        self.img_size = img_size  # (H, W)

        clear_count = 0
        degraded_count = 0

        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                rel_p = parts[0]
                full_p = os.path.join(data_root, rel_p)

                # Class 1: Degraded / Blur / Low-Quality
                # Class 0: Clear / High-Quality
                if "degraded" in rel_p.lower():
                    label = 1
                    degraded_count += 1
                else:
                    label = 0
                    clear_count += 1

                self.samples.append((full_p, label))
                if max_samples and len(self.samples) >= max_samples:
                    break

        np.random.seed(42)
        np.random.shuffle(self.samples)
        print(f"[*] Loaded Quality Dataset from {os.path.basename(train_file)}: {clear_count:,} Clear (Class 0), {degraded_count:,} Degraded (Class 1). Total: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black image
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1] RGB tensor [3, H, W]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return tensor, torch.tensor(label, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Train ClassSR Classifier on License Plate Clear/Degraded data")
    parser.add_argument("--data_root", type=str, default=r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5")
    parser.add_argument("--train_file", type=str, default=r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5\train_labels.txt")
    parser.add_argument("--val_file", type=str, default=r"D:\IEEE\data\phD\data\50k_OCR\aaa_train_ne_version_5\val_labels.txt")
    parser.add_argument("--save_path", type=str, default=os.path.join(BASICSR_ROOT, "experiments", "pretrained_models", "class_sr_classifier_best.pth"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Dataset & Loader: Separate Train and Val
    train_set = LPQualityDataset(args.data_root, args.train_file)
    val_set = LPQualityDataset(args.data_root, args.val_file)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = ClassSRClassifier(in_nc=3, num_classes=2).to(args.device)
    params_k = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
    print(f"[*] ClassSR Classifier Initialized. Params: {params_k:.2f}K ({params_k/1e3:.4f}M)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    print(f"\n[*] Starting training for {args.epochs} epochs on {args.device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += imgs.size(0)
            pbar.set_postfix({'loss': f"{train_loss / total_train:.4f}", 'acc': f"{train_correct / total_train * 100:.2f}%"})

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += imgs.size(0)

        val_acc = (val_correct / total_val) * 100.0
        val_l = val_loss / total_val
        print(f"[*] Epoch {epoch} Val -> Loss: {val_l:.4f} | Accuracy: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'val_acc': val_acc
            }, args.save_path)
            print(f"[+] Saved Best Model ({val_acc:.2f}%) -> {args.save_path}")

    print(f"\n[DONE] Training complete. Best Accuracy: {best_acc:.2f}%. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
