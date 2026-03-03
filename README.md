# IEEE ML Challenge — Fault Detection System
<br>
## OVERVIEW:

This repository contains our team’s solution for the **IEEE SB GEHU Machine Learning Challenge**.

The objective is to detect faulty device states using telemetry data collected from an embedded monitoring system.

The problem is formulated as a binary classification task:

- **0 — Normal Operation**

- **1 — Fault Condition**

Our approach focuses on building a robust model that generalizes well to unseen data using ensemble learning and a carefully designed validation strategy.
---

## DATASET:

The dataset consists of 47 numerical features (F01–F47) representing operational parameters recorded during device activity cycles.

Each sample corresponds to a snapshot of the device’s state.

**Key Characteristics:**

- All features are numeric

- Target variable: "Class"

- Binary labels: Normal (0) or Faulty (1)

- Evaluation metric: F1 Score


## METHODOLOGY:
### **a. Model Architecture:**

We implemented an ensemble of two gradient boosting algorithms:

- **CatBoost Classifier**

- **LightGBM Classifier**

These models are well-suited for tabular data and can capture complex nonlinear relationships between features.

Final predictions are obtained using weighted averaging of model probabilities:

```
Final Probability=0.6×CatBoost+0.4×LightGBM
```

### **b. Validation Strategy:**

To ensure reliable performance estimation and minimize overfitting, we used:

- Stratified 5-Fold Cross-Validation

- Out-of-Fold (OOF) Predictions

- Threshold tuning based on validation performance

The decision threshold was selected to maximize the F1 Score.


## PIPELINE OVERVIEW:

```
                    Training Data
                         ↓
          Stratified 5-Fold Cross-Validation
           ↓                               ↓
      ┌───────────────┐              ┌───────────────┐
      │   CatBoost    │              │   LightGBM    │
      └───────────────┘              └───────────────┘
              └──────── Weighted Averaging ────────┘
                          ↓
                 Threshold Optimization
                          ↓
                    Final Predictions
```


## RESULTS:

- Out-of-Fold F1 Score: 0.9867

- Consistent performance across folds

- Minimal gap between training and validation scores

These results indicate strong generalization capability.


## REPRODUCIBILITY:

Clone the repository:
```
git clone https://github.com/username/repo
cd repo
```
Install dependencies:
```
pip install -r requirements.txt
```
Run the training script:
```
python train.py
```

## OUTPUT:

The script generates a file named ```"submission.csv"``` containing predictions for the test dataset.

Format:
```
ID,Class
```
Predictions are aligned with the original order of the test data.

## TECHNOLOGIES USED:

- Python

- CatBoost

- LightGBM

- Scikit-learn

- Pandas

- NumPy


## ORIGINALITY STATEMENT:

All code, experiments, and methodology in this repository were developed by our team specifically for this challenge.


## Notes

The solution is optimized for CPU execution and can be run on a standard laptop without GPU acceleration.
