import joblib
import numpy as np

# Load the saved model and scaler
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Each row = [ID_int, DLC, Day1, Day2, Day3, Day4, Day5, Day6, Day7, Day8]
# Add or change rows as you like
# --------------------------------------------------

samples = [
    [704,  8,  14,  0,  0,  0,  0,  0,  0,  0],   # Normal-like
    [304,  8,   0,  0,  0,  0,  0,  0,  0,  0],   # Normal-like
    [10,   8, 255,255,255,255,255,255,255,255],   # Attack-like (all FF = DoS)
    [5,    8, 255,254,255,200,255,255,255,210],   # Attack-like
]

print("\n#   ID     DLC  Bytes                              Result      P(Attack)")
print("-" * 75)

for i, row in enumerate(samples):
    id_int, dlc, *bytes_ = row
    b = np.array(bytes_, dtype=float)

    # Must match the exact feature order from training
    features = [id_int, dlc, *bytes_,
                b.sum(), b.mean(), b.std(), b.max(),
                (b > 0).sum(), int((b == 255).all())]

    X     = scaler.transform([features])
    pred  = model.predict(X)[0]
    prob  = model.predict_proba(X)[0][1]
    label = "ATTACK" if pred == 1 else "NORMAL"

    print(f"{i+1}   {id_int:<6} {dlc:<5}{str(bytes_):<35}{label:<12}{prob:.4f}")
