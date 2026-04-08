import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------- Path Configuration -------------

SAVE_DIR  = r"Program\SavedPreLoads\SaveData_8"             # Selects 8 Frame Pre Processed Data
OUTPUT_DIR = r"Program\Models\Baseline\SavedModel_Baseline"  # Save Directory for model

# ------------- Load Saved Data -------------

def load_split(name):
    split_dir = os.path.join(SAVE_DIR, name)
    labels = torch.load(os.path.join(split_dir, "labels.pt"))
    
    data = []
    for i in range(len(labels)):
        frames = torch.load(os.path.join(split_dir, f"{i}.pt"))
        # Mean pool across frames: (C, T, H, W) -> (C, H, W) -> flatten
        frames = frames.mean(dim=1)
        data.append(frames.reshape(-1))
    
    X = torch.stack(data).numpy()
    y = labels.numpy()
    return X, y

# ------------- Baseline Model -------------

def train_baseline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    # Standardize
    print("Standardizing...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # PCA dimensionality reduction
    print("Fitting PCA...")
    pca = PCA(n_components=256, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)

    # Train SVM
    print("Training SVM...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", decision_function_shape="ovr")
    svm.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, svm.predict(X_train))
    val_acc   = accuracy_score(y_val,   svm.predict(X_val))
    test_acc  = accuracy_score(y_test,  svm.predict(X_test))

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")

    # Save models
    joblib.dump(svm,    os.path.join(OUTPUT_DIR, "svm_baseline.pkl"))
    joblib.dump(pca,    os.path.join(OUTPUT_DIR, "pca.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print("Saved SVM, PCA, and scaler.")


# ------------- Run Model -------------
if __name__ == "__main__":
    train_baseline()