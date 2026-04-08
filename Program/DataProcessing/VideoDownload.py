import kagglehub
import shutil
 
# ------------- Path Configuration -------------

DESTINATION_DIR = r"Program\Datasets\GestureVideoData"       # Destination of Raw Dataset
 
# ------------- Download Dataset -------------

print("Downloading dataset...")
path = kagglehub.dataset_download("waseemnagahhenes/sign-language-dataset-wlasl-videos")
print(f"Downloaded to: {path}")
 
# ------------- Copy to Destination -------------

print(f"Copying to: {DESTINATION_DIR}...")
shutil.copytree(path, DESTINATION_DIR)
print("Done.")