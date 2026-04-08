import subprocess

# Step 1:Load Data Set from Kaggle and put into DataSets Folder
subprocess.run(["python", "Program\\DataProcessing\\VideoDownload.py"])

# Step 2: Filter Data
subprocess.run(["python", "Program\\DataProcessing\\FilterDataset.py"])

# Step 3: Get Video Data Information
subprocess.run(["python", "Program\\DataProcessing\\VideoDataInformationVideoInfo.py"])

# Step 4: Balance Dataset
subprocess.run(["python", "Program\\DataProcessing\\BalanceDataset.py"])

# Optional -- uncomment to generate html
# subprocess.run(["python", "Program\\DataProcessing\\ReferenceFileForDataset.py"])
# subprocess.run(["python", "Program\\DataProcessing\\PrintFrameExamples.py"])

# Step 5: Load and Save Preloaded Data
subprocess.run(["python", "Program\\Models\\LoadDataAndSave.py"])

# Step 6: Run ASL Model
subprocess.run(["python", "Program\\Models\\ASLClassifier\\ASLVideoClassification.py"])
subprocess.run(["python", "Program\\Models\\ASLClassifier\\ProcessASLResults.py"])

# Step 7: Run Baseline Model
subprocess.run(["python", "Program\\Models\\Baseline\\BaselineModel.py"])
subprocess.run(["python", "Program\\Models\\Baseline\\ProcessBaselineResults.py"])