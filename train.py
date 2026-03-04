
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
import lightgbm as lgb

# Load training and test datasets
train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Separate features and target
X = train.drop("Class", axis=1)
y = train["Class"]

# Save test IDs and remove them from features
test_ids = test["ID"]
X_test = test.drop("ID", axis=1)

# Stratified K-Fold to maintain class balance
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Train models across folds
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # CatBoost model
    cat = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.025,
        depth=7,
        loss_function="Logloss",
        eval_metric="F1",
        random_state=42,
        thread_count=-1,
        early_stopping_rounds=300,
        verbose=False
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    # LightGBM model
    lgbm = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgbm.fit(X_train, y_train)

    # Validation predictions
    cat_val = cat.predict_proba(X_val)[:, 1]
    lgb_val = lgbm.predict_proba(X_val)[:, 1]

    val_probs = 0.6 * cat_val + 0.4 * lgb_val
    oof_preds[val_idx] = val_probs

    # Test predictions
    cat_test = cat.predict_proba(X_test)[:, 1]
    lgb_test = lgbm.predict_proba(X_test)[:, 1]

    test_preds += (0.6 * cat_test + 0.4 * lgb_test) / kf.n_splits

# Threshold tuning using OOF predictions
best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (oof_preds > t).astype(int)
    score = f1_score(y, preds)

    if score > best_f1:
        best_f1 = score
        best_threshold = t

print("Best threshold:", best_threshold)
print("OOF F1 score:", best_f1)

# Final predictions for test set
final_preds = (test_preds > best_threshold).astype(int)

# Create FINAL.csv
submission = pd.DataFrame({
    "ID": test_ids,
    "CLASS": final_preds
})

submission.to_csv("FINAL.csv", index=False)

print("\nFINAL.csv created successfully")
