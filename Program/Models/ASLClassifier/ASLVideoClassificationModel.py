import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset, DataLoader

# ------------- Configuration -------------

NUM_FRAMES   = 8
SAVE_DIR     = rf"Program\SavedPreLoads\SaveData_{NUM_FRAMES}"
MODEL_DIR    = r"Program\SavedModels"
OUTPUT_DIR   = r"Program\Models\ASLClassifier\SavedModel_ASLClassifier"
NUM_CLASSES  = 134
BATCH_SIZE   = 16

# ------------- Saved Video Dataset -------------

class SavedVideoDataset(Dataset):
    def __init__(self, split, augment=False):
        self.split_dir = os.path.join(SAVE_DIR, split)
        self.labels    = torch.load(os.path.join(self.split_dir, "labels.pt"), weights_only=True)
        self.augment   = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frames = torch.load(os.path.join(self.split_dir, f"{idx}.pt"), weights_only=True)  # (C, T, H, W)
        if self.augment:
            frames = self._augment(frames)
        return frames, self.labels[idx]

    def _augment(self, frames):
        # frames: (C, T, H, W) — apply same transform to every frame
        if random.random() < 0.5:
            frames = TF.hflip(frames)                          # horizontal flip
        brightness = random.uniform(0.8, 1.2)
        contrast   = random.uniform(0.8, 1.2)
        # apply per-frame so torchvision ops get (C, H, W)
        frames = torch.stack([
            TF.adjust_contrast(TF.adjust_brightness(frames[:, t], brightness), contrast)
            for t in range(frames.shape[1])
        ], dim=1)
        return frames


def load_splits():
    train_loader = DataLoader(SavedVideoDataset("train", augment=False), batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(SavedVideoDataset("val",   augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(SavedVideoDataset("test",  augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ------------- ASL Model -------------

class ASLModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ASLModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2) 
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2) 
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1)) 

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ------------- Training -------------

def train(num_epochs=30, lr=3e-3):
    import time
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.txt")
    log = open(log_path, "w")

    def log_print(msg):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    torch.set_num_threads(os.cpu_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device} | CPU threads: {os.cpu_count()}")

    train_loader, val_loader, test_loader = load_splits()

    model     = ASLModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for frames, labels in train_loader:
            frames, labels = frames.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(frames)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        train_acc = train_correct / train_total
        scheduler.step()

        # ------ Validation --------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs     = model(frames)
                loss        = criterion(outputs, labels)
                val_loss   += loss.item()
                preds       = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start
        log_print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) "
                  f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "asl_model_best.pt"))
            log_print(f"  --> New best model saved (val acc: {val_acc:.4f})")

    # ----- Test -----
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "asl_model_best.pt"), weights_only=True))
    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs       = model(frames)
            preds         = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total   += labels.size(0)

    test_acc = test_correct / test_total
    elapsed  = time.time() - start_time

    log_print(f"\nTest Accuracy:  {test_acc:.4f}")
    log_print(f"Total Runtime:  {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    log.close()
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    train()