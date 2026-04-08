import os
import shutil
import pandas as pd

"""
This script filters the original WLASL dataset based on a list of valid classes provided in a CSV file.
The CSV file "ReducedClasses.csv" contains only classes with 10 or more videos.
"""

# ------------- Path Configuration -------------

SRC_DIR = "Program\\Datasets\\GestureVideoData"        # Path to the original WLASL dataset       
DST_DIR = "Program\\Datasets\\FilteredDataset\\"       # Path the filtered dataset will be saved to
CSV_PATH = "Program\\Datasets\\Atleast10Videos.csv"    # Path to the CSV file


def filter_dataset():
    # Load valid class names from CSV
    df = pd.read_csv(CSV_PATH)
    valid_classes = sorted(df["Class"].str.strip().tolist())
    print(f"Classes to copy: {len(valid_classes)}")

    # Check source folder exists
    if not os.path.isdir(SRC_DIR):
        print(f"ERROR: Source folder not found: {SRC_DIR}")
        return

    # Create destination folder
    os.makedirs(DST_DIR, exist_ok=True)

    copied = []
    missing = []

    for class_name in valid_classes:
        src = os.path.join(SRC_DIR, class_name)
        dst = os.path.join(DST_DIR, class_name)

        if not os.path.isdir(src):
            print(f"  [MISSING]  '{class_name}' not found in source, skipping.")
            missing.append(class_name)
            continue

        shutil.copytree(src, dst, dirs_exist_ok=True)
        num_files = len(os.listdir(dst))
        print(f"  [COPIED]   {class_name} ({num_files} files)")
        copied.append(class_name)

    print("\n--- Summary ---")
    print(f"  Successfully copied : {len(copied)} classes")
    print(f"  Missing from source : {len(missing)} classes")
    if missing:
        print(f"  Missing classes     : {missing}")
    print(f"  Output saved to     : {DST_DIR}")


if __name__ == "__main__":
    filter_dataset()