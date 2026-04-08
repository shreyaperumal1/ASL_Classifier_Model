import os
import random
import cv2
import numpy as np

"""
This script balances the dataset by augmenting videos in underrepresented classes.
"""

# ------------- Path Configuration -------------

SRC_DIR = "Program\\Datasets\\FilteredDataset"       # Source of Reduced Filtered Dataset
DST_DIR = "Program\\Datasets\\BalancedDataset"       # Destination of Balanced Dataset


# ------------- Augmentation Settings -------------

BRIGHTNESS_RANGE = 0.3           # 30% brightness adjustment
CONTRAST_RANGE   = 0.3           # 30% contrast adjustment


def adjust_brightness_contrast(frame, brightness_factor, contrast_factor):
    """Apply brightness and contrast jitter to a single frame."""
    # Contrast: scale around mean
    mean = frame.mean()
    frame = (frame - mean) * contrast_factor + mean

    # Brightness: add offset
    frame = frame * brightness_factor
    return np.clip(frame, 0, 255).astype(np.uint8)


def augment_video(src_path, dst_path):
    """Read a video, apply random brightness/contrast jitter, and save it."""
    cap = cv2.VideoCapture(src_path)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    brightness_factor = 1.0 + random.uniform(-BRIGHTNESS_RANGE, BRIGHTNESS_RANGE)
    contrast_factor   = 1.0 + random.uniform(-CONTRAST_RANGE,   CONTRAST_RANGE)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        augmented = adjust_brightness_contrast(frame, brightness_factor, contrast_factor)
        out.write(augmented)

    cap.release()
    out.release()


def balance_dataset():
    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: Source folder not found: {SRC_DIR}")
        return

    # Count videos per class
    class_counts = {}
    for class_name in os.listdir(SRC_DIR):
        class_path = os.path.join(SRC_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        videos = [f for f in os.listdir(class_path) if f.endswith(".mp4")]
        class_counts[class_name] = videos

    if not class_counts:
        print("ERROR: No class folders found in source directory.")
        return

    target_count = max(len(v) for v in class_counts.values())
    print(f"Target videos per class : {target_count}")
    print(f"Classes to process      : {len(class_counts)}\n")

    os.makedirs(DST_DIR, exist_ok=True)

    for class_name, videos in sorted(class_counts.items()):
        src_class_dir = os.path.join(SRC_DIR, class_name)
        dst_class_dir = os.path.join(DST_DIR, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        # Copy original videos first
        for video in videos:
            src = os.path.join(src_class_dir, video)
            dst = os.path.join(dst_class_dir, video)
            import shutil
            shutil.copy2(src, dst)

        # How many augmented videos we need to add
        num_to_add = target_count - len(videos)

        if num_to_add == 0:
            print(f"  [OK]       {class_name} ({len(videos)} videos) — no augmentation needed")
            continue

        print(f"  [AUGMENT]  {class_name} ({len(videos)} videos) — adding {num_to_add} augmented videos")

        # Cycle through original videos to generate augmented ones
        aug_index = 1
        for i in range(num_to_add):
            src_video = videos[i % len(videos)]                      # cycle if needed
            src_path  = os.path.join(src_class_dir, src_video)
            dst_name  = f"aug_{aug_index:04d}_{src_video}"
            dst_path  = os.path.join(dst_class_dir, dst_name)
            augment_video(src_path, dst_path)
            aug_index += 1

    print("\n--- Summary ---")
    print(f"  All classes balanced to : {target_count} videos")
    print(f"  Output saved to         : {DST_DIR}")


if __name__ == "__main__":
    balance_dataset()