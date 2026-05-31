"""
=============================================================
 Autonomous Vehicle CAN Bus Anomaly Detection
 Dataset:  car_data.xlsx  (5 sheets: Normal, DoS, Fuzzy,
           Gear-Spoofing, RPM-Spoofing)
 Label:    0 = Normal  |  1 = Attack
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# CONFIG  –  adjust paths / settings here
# ─────────────────────────────────────────────
DATA_PATH   = r"C:\Users\kkart\OneDrive\Desktop\Ml_Antivirus\car_data.xlsx"
SHEETS      = ["Normal_dataset", "Dos_dataset",
               "Fuzzy_dataset", "gear_dataset", "RPM_dataset"]
LABEL_COL   = "Label"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
USE_SMOTE   = True          # set False if data is already balanced
MAX_ROWS_PER_SHEET = 50_000 # cap per sheet to keep memory reasonable; set None to load all


# ══════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════
def load_data(path: str, sheets: list, max_rows=None) -> pd.DataFrame:
    frames = []
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    for sheet in sheets:
        nrows = max_rows
        df = pd.read_excel(path, sheet_name=sheet, nrows=nrows)
        print(f"  {sheet:20s}  rows={len(df):>7,}  "
              f"attacks={df[LABEL_COL].sum():>6,}  "
              f"normal={(df[LABEL_COL]==0).sum():>6,}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined total  rows={len(combined):>7,}")
    return combined


# ══════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════
DATA_BYTE_COLS = ["Day 1", "Day 2", "Day 3", "Day 4",
                  "Day 5", "Day 6", "Day 7", "Day 8"]

def hex_to_int(val):
    """Convert hex string OR numeric value to integer."""
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        try:
            return int(val, 16)
        except ValueError:
            return 0
    return int(val)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # ── Convert hex byte columns to integers ──
    for col in DATA_BYTE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(hex_to_int)

    # ── Encode CAN ID  (hex string → integer) ──
    if "ID" in df.columns:
        df["ID_int"] = df["ID"].apply(hex_to_int)
    else:
        df["ID_int"] = 0

    # ── Keep useful numeric columns ──
    feature_cols = ["ID_int", "DLC"] + DATA_BYTE_COLS

    # ── Feature engineering ──
    df["byte_sum"]    = df[DATA_BYTE_COLS].sum(axis=1)
    df["byte_mean"]   = df[DATA_BYTE_COLS].mean(axis=1)
    df["byte_std"]    = df[DATA_BYTE_COLS].std(axis=1).fillna(0)
    df["byte_max"]    = df[DATA_BYTE_COLS].max(axis=1)
    df["nonzero_cnt"] = (df[DATA_BYTE_COLS] > 0).sum(axis=1)
    df["all_ff"]      = (df[DATA_BYTE_COLS] == 255).all(axis=1).astype(int)

    engineered = ["byte_sum", "byte_mean", "byte_std",
                  "byte_max", "nonzero_cnt", "all_ff"]
    all_features = feature_cols + engineered

    # ── Drop rows where label is missing ──
    df = df.dropna(subset=[LABEL_COL])

    X = df[all_features].fillna(0)
    y = df[LABEL_COL].astype(int)

    print(f"  Features used : {all_features}")
    print(f"  Total samples : {len(y):,}")
    print(f"  Normal (0)    : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
    print(f"  Attack (1)    : {(y==1).sum():,}  ({(y==1).mean()*100:.1f}%)")
    return X, y, all_features


# ══════════════════════════════════════════════
# 3.  TRAIN / TEST SPLIT + SMOTE
# ══════════════════════════════════════════════
def split_and_balance(X, y):
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

    print(f"  Train: {len(y_train):,}  |  Test: {len(y_test):,}")

    if USE_SMOTE and (y_train.value_counts().min() / y_train.value_counts().max()) < 0.5:
        print("  Applying SMOTE to balance training set …")
        sm = SMOTE(random_state=RANDOM_SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE – Train: {len(y_train):,}  "
              f"(Normal={( y_train==0).sum():,}, Attack={(y_train==1).sum():,})")

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ══════════════════════════════════════════════
# 4.  MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════
MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                    random_state=RANDOM_SEED),
    "XGBoost"             : XGBClassifier(n_estimators=100, use_label_encoder=False,
                                          eval_metric="logloss",
                                          random_state=RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100,
                                                       random_state=RANDOM_SEED),
}

def evaluate_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    results = {}
    for name, model in MODELS.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_prob)

        print(f"  Training time : {elapsed:.2f}s")
        print(f"  ROC-AUC       : {auc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Normal", "Attack"],
                                    digits=4))

        results[name] = {"model": model, "y_pred": y_pred,
                         "y_prob": y_prob, "auc": auc}

    return results


# ══════════════════════════════════════════════
# 5.  VISUALISATIONS
# ══════════════════════════════════════════════
def plot_results(results, y_test, feature_names):
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))
    fig.suptitle("Autonomous Vehicle – CAN Bus Anomaly Detection", fontsize=14, y=1.01)

    for i, (name, res) in enumerate(results.items()):
        # ── Confusion matrix ──
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
        disp.plot(ax=axes[0, i], colorbar=False)
        axes[0, i].set_title(f"{name}\n(AUC={res['auc']:.3f})")

        # ── ROC curve ──
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        axes[1, i].plot(fpr, tpr, lw=2,
                        label=f"AUC = {res['auc']:.3f}")
        axes[1, i].plot([0,1], [0,1], "k--", lw=1)
        axes[1, i].set_xlabel("False Positive Rate")
        axes[1, i].set_ylabel("True Positive Rate")
        axes[1, i].set_title(f"ROC – {name}")
        axes[1, i].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("\n  Saved: model_comparison.png")
    plt.show()

    # ── Feature importance (Random Forest) ──
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title("Feature Importance – Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    print("  Saved: feature_importance.png")
    plt.show()


# ══════════════════════════════════════════════
# 6.  SUMMARY TABLE
# ══════════════════════════════════════════════
def print_summary(results, y_test):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<22}  {'Accuracy':>9}  {'Precision':>9}  "
          f"{'Recall':>7}  {'F1':>7}  {'ROC-AUC':>8}")
    print("  " + "-" * 72)

    for name, res in results.items():
        yp = res["y_pred"]
        acc = accuracy_score(y_test, yp)
        pre = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1  = f1_score(y_test, yp, zero_division=0)
        auc = res["auc"]
        print(f"  {name:<22}  {acc:>9.4f}  {pre:>9.4f}  "
              f"{rec:>7.4f}  {f1:>7.4f}  {auc:>8.4f}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Load
    df_raw = load_data(DATA_PATH, SHEETS, max_rows=MAX_ROWS_PER_SHEET)

    # 2. Preprocess
    X, y, feature_names = preprocess(df_raw)

    # 3. Split & balance
    X_train, X_test, y_train, y_test, scaler = split_and_balance(X, y)

    # 4. Train & evaluate
    results = evaluate_models(X_train, X_test, y_train, y_test)

    # 5. Plots
    plot_results(results, y_test, feature_names)

    # 6. Summary table
    print_summary(results, y_test)

    print("\n✅  Done! Check model_comparison.png and feature_importance.png")
