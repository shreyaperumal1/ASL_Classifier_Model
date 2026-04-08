import os
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ------------- Import from training script -------------

from BaselineModel import load_split, OUTPUT_DIR


# ------------- Evaluation -------------

def evaluate(svm, pca, scaler, split_name):
    X, y_true = load_split(split_name)
    X         = scaler.transform(X)
    X         = pca.transform(X)
    y_pred    = svm.predict(X)

    return {
        "y_true":    y_true,
        "y_pred":    y_pred,
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Samples":   len(y_true),
    }


# ------------- Main -------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load saved models
    svm    = joblib.load(os.path.join(OUTPUT_DIR, "svm_baseline.pkl"))
    pca    = joblib.load(os.path.join(OUTPUT_DIR, "pca.pkl"))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.pkl"))
    print("Loaded SVM, PCA, and scaler.\n")

    results_path = os.path.join(OUTPUT_DIR, "results_summary.txt")
    with open(results_path, "w") as out:

        for split_name, folder_name in [("Validation", "val"), ("Test", "test")]:
            print(f"--- Evaluating {split_name} split ---")
            metrics = evaluate(svm, pca, scaler, folder_name)

            summary = (
                f"\n{'='*50}\n"
                f"{split_name} Results (SVM Baseline)\n"
                f"{'='*50}\n"
                f"  Accuracy  : {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)\n"
                f"  Precision : {metrics['Precision']:.4f}  (macro)\n"
                f"  Recall    : {metrics['Recall']:.4f}  (macro)\n"
                f"  F1 Score  : {metrics['F1']:.4f}  (macro)\n"
                f"  Samples   : {metrics['Samples']}\n"
            )
            print(summary)
            out.write(summary)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Summary text:         {results_path}")


if __name__ == "__main__":
    main()