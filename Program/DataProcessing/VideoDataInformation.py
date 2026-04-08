import os
import cv2
import pandas as pd

"""
This script generates a CSV file containing the number of videos in each class of the dataset.
It also generates a CSV for classes that have more than 9 videos, which can be used for a reduced dataset.
"""
# ------------- Path Configuration -------------
DATASET_DIR = r"Program\Datasets\FilteredDataset"
OUTPUT_CSV_NUM_VIDEOS = r"Program\DatasetInformation\VideoDataClasses.csv"
OUTPUT_CSV_REDUCED_CLASSES = r"Program\DatasetInformation\ReducedClasses.csv"

# ------------- Count Videos per Class -------------
def count_videos_per_class(dataset_dir):
    class_counts = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            video_count = sum(1 for f in os.listdir(class_path) if f.endswith(".mp4"))
            class_counts.append({"Class": class_name, "NumVideos": video_count})
    return pd.DataFrame(class_counts)

# ------------- Main Execution -------------
if __name__ == "__main__":
    df_counts = count_videos_per_class(DATASET_DIR)
    df_counts.to_csv(OUTPUT_CSV_NUM_VIDEOS, index=False)
    print(f"Saved video counts to {OUTPUT_CSV_NUM_VIDEOS}")
    df_reduced = df_counts[df_counts["NumVideos"] > 9].reset_index(drop=True)
    df_reduced.to_csv(OUTPUT_CSV_REDUCED_CLASSES, index=False)
    print(f"Saved reduced classes to {OUTPUT_CSV_REDUCED_CLASSES}")
