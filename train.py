import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier
import lightgbm as lgb

# =========================
# LOAD DATA
# =========================

train = pd.read_csv("TRAIN.csv")
test = pd.read_csv("TEST.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

X = train.drop("Class", axis=1)
y = train["Class"]

test_ids = test["ID"]
test_features = test.drop("ID", axis=1)

# =========================
# STRATIFIED K-FOLD
# =========================

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test_features))

# =========================
# TRAIN ACROSS FOLDS
# =========================

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n🔥 Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ===== CATBOOST (Kaggle-style params) =====
    cat = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.025,
        depth=7,
        loss_function='Logloss',
        eval_metric='F1',
        random_state=42,
        thread_count=-1,
        early_stopping_rounds=300,
        verbose=False
    )

    cat.fit(X_train, y_train, eval_set=(X_val, y_val))

    # ===== LIGHTGBM =====
    lgbm = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             eval_metric='binary_logloss')

    # ===== VALIDATION PREDICTIONS =====
    cat_val = cat.predict_proba(X_val)[:, 1]
    lgb_val = lgbm.predict_proba(X_val)[:, 1]

    # Ensemble weights
    val_probs = 0.6 * cat_val + 0.4 * lgb_val
    oof_preds[val_idx] = val_probs

    # ===== TEST PREDICTIONS =====
    cat_test = cat.predict_proba(test_features)[:, 1]
    lgb_test = lgbm.predict_proba(test_features)[:, 1]

    test_probs = 0.6 * cat_test + 0.4 * lgb_test
    test_preds += test_probs / kf.n_splits

# =========================
# THRESHOLD OPTIMIZATION
# =========================

best_f1 = 0
best_threshold = 0.5

for t in np.arange(0.1, 0.9, 0.01):
    preds = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\n🏆 Best Threshold:", best_threshold)
print("🏆 OOF F1 Score:", best_f1)

# =========================
# FINAL TEST PREDICTIONS
# =========================

final_test_preds = (test_preds > best_threshold).astype(int)

# =========================
# SAVE SUBMISSION
# =========================

submission = pd.DataFrame({
    "ID": test_ids,
    "Class": final_test_preds
})

submission.to_csv("submission.csv", index=False)

print("\n✅ FINAL ENSEMBLE submission.csv created!")