# 🚗 Autonomous Vehicle CAN Bus Anomaly Detection

Detects cyberattacks on autonomous vehicles by analyzing CAN bus traffic using machine learning.

## Attack Types Detected
- DoS (Denial of Service)
- Fuzzy Attack
- Gear Spoofing
- RPM Spoofing

## Models Used
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosted trees |
| Gradient Boosting | Sklearn boosting model |

## Features Engineered
- CAN ID (hex → int), DLC, 8 data byte columns
- Byte sum, mean, std, max, non-zero count, all-FF flag

## Setup

```bash
pip install -r requirements.txt
```

Update `DATA_PATH` in the script to point to your `car_data.xlsx` file, then run:

```bash
python av_anomaly_detection.py
```

## Output
- `model_comparison.png` — Confusion matrices + ROC curves
- `feature_importance.png` — Random Forest feature importance
- Console summary table with Accuracy, Precision, Recall, F1, ROC-AUC
