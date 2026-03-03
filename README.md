IEEE ML Challenge — Fault Detection System

Overview:

This repository contains our team’s solution for the IEEE SB GEHU Machine Learning Challenge.

The objective is to detect faulty device states using telemetry data collected from an embedded monitoring system.

The problem is formulated as a binary classification task:

0 — Normal Operation

1 — Fault Condition

Our approach focuses on building a robust model that generalizes well to unseen data using ensemble learning and a carefully designed validation strategy.

📊 Dataset

The dataset consists of 47 numerical features (F01–F47) representing operational parameters recorded during device activity cycles.

Each sample corresponds to a snapshot of the device’s state.

Key Characteristics:

All features are numeric

Target variable: "Class"

Binary labels: Normal (0) or Faulty (1)

Evaluation metric: F1 Score

🧠 Methodology
🔹 Model Architecture

We implemented an ensemble of two gradient boosting algorithms:

CatBoost Classifier

LightGBM Classifier

These models are well-suited for tabular data and can capture complex nonlinear relationships between features.

Final predictions are obtained using weighted averaging:

Final Probability = 0.6 × CatBoost + 0.4 × LightGBM
🔹 Validation Strategy

To ensure reliable performance estimation and minimize overfitting, we used:

Stratified 5-Fold Cross-Validation

Out-of-Fold (OOF) Predictions

Threshold tuning based on validation performance

The decision threshold was selected to maximize the F1 Score.

⚙️ Pipeline Overview
                Training Data
                     ↓
      Stratified 5-Fold Cross-Validation
         ↓                           ↓
   ┌───────────────┐           ┌───────────────┐
   │   CatBoost    │           │   LightGBM    │
   └───────────────┘           └───────────────┘
           └────── Weighted Averaging ──────┘
                        ↓
               Threshold Optimization
                        ↓
                Final Predictions
📈 Results

Out-of-Fold F1 Score: 0.9867

Consistent performance across folds

Minimal gap between training and validation scores

These results indicate strong generalization capability.

🔁 Reproducibility
1️⃣ Clone the Repository
git clone https://github.com/username/repo
cd repo
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Training
python train.py
📂 Output

The script generates a file named:

submission.csv

Format:

ID,Class

Predictions are aligned with the original order of the test dataset.

🛠 Technologies Used

Python

CatBoost

LightGBM

Scikit-learn

Pandas

NumPy

📜 Originality Statement

All code, experiments, and methodology in this repository were developed by our team specifically for this challenge.

💻 Notes

The solution is optimized for CPU execution and can be run on a standard laptop without GPU acceleration.
