import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# -------------- Configuration --------------

DATASET_PATH  = r"Program\BalancedDataset"                      # Dataset Path
NUM_FRAMES    = 8                                               # Modifiable Frame Count
SAVE_DIR      = rf"Program\SavedPreLoads\SaveData_{NUM_FRAMES}" # Save Path
IMG_SIZE      = (64, 64)                                        # Modifiable Frame Size
TRAIN_RATIO   = 0.70                                            # Modifiable Train Ratio
VAL_RATIO     = 0.15                                            # Modifiable Validation Ratio
TEST_RATIO    = 0.15                                            # Modifiable Test Ratio
SEED          = 42                                              # Set Random for Reproducibility

# ------------ Processing and Saving ------------

def sample_frames(video_path, num_frames=NUM_FRAMES):
    """Uniformly sample num_frames frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(frame)
        else:
            frames.append(np.zeros((*IMG_SIZE, 3), dtype=np.uint8))
    cap.release()
    return frames

# -------------- Dataset Class --------------

class VideoFolderDataset(Dataset):
    """
    Loads videos from a folder structure:
        root/
            class_a/
                video1.mp4
                video2.mp4
            class_b/
    """
    def __init__(self, root, num_frames=NUM_FRAMES):
        self.num_frames = num_frames
        self.samples = []   # list of (video_path, label_idx)
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.endswith(".mp4"):
                    self.samples.append((
                        os.path.join(cls_path, fname),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = sample_frames(video_path, self.num_frames)
        tensor = torch.tensor(np.stack(frames), dtype=torch.float32)
        tensor = tensor.permute(3, 0, 1, 2) / 255.0
        return tensor, label


def save_splits():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Loading dataset...")
    full_dataset = VideoFolderDataset(DATASET_PATH)
    total = len(full_dataset)

    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)
    n_test  = total - n_train - n_val

    print(f"Total: {total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Save class mapping
    torch.save(full_dataset.class_to_idx, os.path.join(SAVE_DIR, "class_to_idx.pt"))

    for split, name in [(train_set, "train"), (val_set, "val"), (test_set, "test")]:
        split_dir = os.path.join(SAVE_DIR, name)
        os.makedirs(split_dir, exist_ok=True)

        labels = []
        for i, (frames, label) in enumerate(split):
            # Save each video as its own file
            torch.save(frames, os.path.join(split_dir, f"{i}.pt"))
            labels.append(label)
            if (i + 1) % 100 == 0:
                print(f"  [{name}] processed {i + 1}/{len(split)}")

        # Save labels list separately
        torch.save(torch.tensor(labels), os.path.join(split_dir, "labels.pt"))
        print(f"Saved {name} — {len(labels)} samples")

    print("Done.")


# Save the video data splits to disk

save_splits()