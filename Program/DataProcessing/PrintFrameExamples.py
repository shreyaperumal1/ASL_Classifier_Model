import cv2
import matplotlib.pyplot as plt
import os

"""
This script extract frames of videos in the original dataset and saves them as images. 
These images are utilized in the report to gain insight into the dataset used.
"""

def get_middle_frame(video_path):
    """Extract the middle frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB for matplotlib
        frame = cv2.resize(frame, (224, 224))
        return frame
    return None

def save_sample_frames(dataset_path, output_path, title):
    classes = sorted(os.listdir(dataset_path))
    selected_classes = classes[:5]

    frames = []
    labels = []

    for cls in selected_classes:
        cls_path = os.path.join(dataset_path, cls)
        videos = [f for f in os.listdir(cls_path) if f.endswith(".mp4")]
        if not videos:
            continue
        frame = get_middle_frame(os.path.join(cls_path, videos[0]))
        if frame is not None:
            frames.append(frame)
            labels.append(cls)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(title, fontsize=14)
    for ax, frame, label in zip(axes, frames, labels):
        ax.imshow(frame, cmap="gray")
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {output_path}")

save_sample_frames(
    dataset_path="Program\\Datasets\\GestureVideoData",
    output_path="Program\\DatasetInformation\\original_samples.png",
    title="Sample Frames — Original Dataset"
)
