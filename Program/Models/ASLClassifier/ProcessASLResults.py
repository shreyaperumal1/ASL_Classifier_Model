import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------- Import from training script -------------

from ASLVideoClassificationModel import (
    ASLModel, SavedVideoDataset,
    SAVE_DIR, OUTPUT_DIR, NUM_CLASSES, BATCH_SIZE
)


# ------------- Evaluation -------------

def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            preds  = model(frames).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


# ------------- Training Curve Plot -------------

def plot_training_curves(log_path, out_dir):
    if not os.path.exists(log_path):
        print(f"  [SKIP] Log not found: {log_path}")
        return

    epochs, train_acc, val_acc, train_loss, val_loss, epoch_times = [], [], [], [], [], []

    with open(log_path) as f:
        for line in f:
            if not line.startswith("Epoch"):
                continue
            try:
                parts = line.split("|")
                ep = int(line.split("[")[1].split("/")[0])
                et = float(line.split("(")[1].split("s)")[0])
                tl = float(parts[0].split("Train Loss:")[1].strip())
                ta = float(parts[1].split("Train Acc:")[1].strip())
                vl = float(parts[2].split("Val Loss:")[1].strip())
                va = float(parts[3].split("Val Acc:")[1].strip())
                epochs.append(ep); epoch_times.append(et)
                train_loss.append(tl); train_acc.append(ta)
                val_loss.append(vl); val_acc.append(va)
            except Exception:
                continue

    if not epochs:
        print("  [SKIP] Could not parse any epoch data from log.")
        return

    cumulative_time = np.cumsum(epoch_times) / 60  # convert to minutes

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs, train_acc,  label="Train Acc",  marker="o", markersize=3)
    ax1.plot(epochs, val_acc,    label="Val Acc",    marker="o", markersize=3)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy over Epochs"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax2.plot(epochs, val_loss,   label="Val Loss",   marker="o", markersize=3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss over Epochs"); ax2.legend(); ax2.grid(True)

    ax3.bar(epochs, epoch_times, color="steelblue", alpha=0.6, label="Epoch Time (s)")
    ax3b = ax3.twinx()
    ax3b.plot(epochs, cumulative_time, color="red", marker="o", markersize=3, label="Cumulative (min)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Time per Epoch (s)")
    ax3b.set_ylabel("Cumulative Time (min)")
    ax3.set_title("Runtime per Epoch")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2); ax3.grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ------------- Main -------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    model = ASLModel(num_classes=NUM_CLASSES).to(device)
    model_path = os.path.join(OUTPUT_DIR, "asl_model_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded model from: {model_path}\n")

    # Data loaders
    val_loader  = DataLoader(SavedVideoDataset("val"),  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(SavedVideoDataset("test"), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results_path = os.path.join(OUTPUT_DIR, "results_summary.txt")
    with open(results_path, "w") as out:

        for split_name, loader in [("Validation", val_loader), ("Test", test_loader)]:
            print(f"--- Evaluating {split_name} split ---")
            y_true, y_pred = evaluate(loader, model, device)

            acc       = (y_true == y_pred).mean()
            precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)

            summary = (
                f"\n{'='*50}\n"
                f"{split_name} Results\n"
                f"{'='*50}\n"
                f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)\n"
                f"  Precision : {precision:.4f}  (macro)\n"
                f"  Recall    : {recall:.4f}  (macro)\n"
                f"  F1 Score  : {f1:.4f}  (macro)\n"
                f"  Samples   : {len(y_true)}\n"
            )
            print(summary)
            out.write(summary)

    # Training curves
    print("\n--- Generating Training Curves ---")
    plot_training_curves(os.path.join(OUTPUT_DIR, "training_log.txt"), OUTPUT_DIR)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Summary text:         {results_path}")


if __name__ == "__main__":
    main()