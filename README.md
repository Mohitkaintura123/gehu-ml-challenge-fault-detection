# IEEE ML Challenge — Fault Detection System 
<br/>

## OVERVIEW

This repository contains our team’s solution for the **IEEE SB GEHU Machine Learning Challenge**.

The objective is to detect faulty device states using telemetry data collected from an embedded monitoring system.

The problem is formulated as a binary classification task:

- **0 — Normal Operation**

- **1 — Fault Condition**

Our approach focuses on building a robust model that generalizes well to unseen data using ensemble learning and a carefully designed validation strategy.

<br/>

## DATASET:

The dataset consists of 47 numerical features (F01–F47) representing operational parameters recorded during device activity cycles.

Each sample corresponds to a snapshot of the device’s state.

**Key Characteristics:**

- All features are numeric
- Target variable: "Class"
- Binary labels: Normal (0) or Faulty (1)
- Evaluation metric: F1 Score

<br/>

## METHODOLOGY:

### **a. Model Architecture:**

We implemented an ensemble of two gradient boosting algorithms:

- **CatBoost Classifier**
- **LightGBM Classifier**

These models are well-suited for tabular data and can capture complex nonlinear relationships between features.

Final predictions are obtained using weighted averaging of model probabilities:

```
Final Probability = 0.6 × CatBoost + 0.4 × LightGBM
```

### **b. Validation Strategy:**

To ensure reliable performance estimation and minimize overfitting, we used:

- Stratified 5-Fold Cross-Validation
- Out-of-Fold (OOF) Predictions
- Threshold tuning based on validation performance

The decision threshold was selected to maximize the F1 Score.

### **c. Feature Analysis**

Basic exploratory analysis was performed to understand feature behavior and class distribution.

Since all features represent numerical sensor readings, gradient boosting models were selected because they are particularly effective for structured tabular datasets and require minimal preprocessing.

Feature importance analysis from CatBoost and LightGBM was also examined to confirm that multiple features contribute to the prediction rather than relying on a single dominant variable.

<br/>

## PIPELINE OVERVIEW:

```
                     Training Data
                          ↓
           Stratified 5-Fold Cross-Validation
           ↓                               ↓
      ┌───────────────┐              ┌───────────────┐
      │   CatBoost       │              |   LightGBM       |
      └───────────────┘              └───────────────┘
              └──────── Weighted Averaging ────────┘
                           ↓
                  Threshold Optimization
                           ↓
                      Final Predictions
```

<br/>

## RESULTS:

- Out-of-Fold F1 Score: 0.9867
- Consistent performance across folds
- Minimal gap between training and validation scores

These results indicate strong generalization capability.

<br/>

## REPRODUCIBILITY:
    
Clone the repository:

git clone https://github.com/Mohitkaintura123/gehu-ml-challenge-fault-detection

cd repo

Install dependencies:

pip install -r requirements.txt

Run the training script:

python train.py


<br/>

## OUTPUT:

The script generates a file named **FINAL.csv** containing predictions for the test dataset.

Format:

ID,CLASS

Predictions are aligned with the original order of the test data and contain exactly **10944 rows**, matching the provided TEST dataset.

<br/>

## TECHNOLOGIES USED:

- Python
- CatBoost
- LightGBM
- Scikit-learn
- Pandas
- NumPy

<br/>

## ORIGINALITY STATEMENT:

All code, experiments, and methodology in this repository were developed by our team specifically for this challenge.

<br/>

## Notes

The solution is optimized for CPU execution and can be run on a standard laptop without GPU acceleration. Random seeds were fixed during training to ensure reproducibility of results.
