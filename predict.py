import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from dense import Dense
from activations import ReLU
import sys

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_model_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    Ws = []
    Bs = []
    i = 0
    while True:
        wkey = f"W{i}"
        bkey = f"b{i}"
        if wkey in data and bkey in data:
            Ws.append(data[wkey])
            Bs.append(data[bkey])
            i += 1
        else:
            break
    if len(Ws) == 0:
        raise RuntimeError("No W/B pairs found in model file")
    return Ws, Bs

def build_model_from_params(Ws, Bs):
    layers = []
    for idx, (W, b) in enumerate(zip(Ws, Bs)):
        out, inp = W.shape  # W saved as (out_features, in_features)
        d = Dense(inp, out)  # Dense(input_size, output_size)
        d.weights = W.copy()
        d.bias = b.copy().reshape((out, 1))
        layers.append(d)
        # if not last Dense, insert ReLU activation by convention
        if idx < len(Ws) - 1:
            layers.append(ReLU())
    return layers

def forward_pass(model, x):
    out = x
    for layer in model:
        out = layer.forward(out)
    return out

def parse_labels(series):
    # Accept 'M'/'B' or numeric 0/1
    s = series.astype(str).str.strip().str.upper()
    if set(s.unique()) <= {"M", "B"}:
        return (s == "M").astype(np.int64).to_numpy()
    # try numeric conversion
    try:
        arr = pd.to_numeric(series, errors="coerce").to_numpy()
        if np.isnan(arr).any():
            raise ValueError
        return arr.astype(np.int64)
    except Exception:
        raise RuntimeError("Unable to parse labels. Expect 'M'/'B' or numeric 0/1 in column 1.")

def binary_cross_entropy_scalar(y_true, p_pred, eps=1e-12):
    p = np.clip(p_pred, eps, 1 - eps)
    y = y_true
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def predict_dataframe(df, model, scaler, feature_start_col=2, threshold=0.5):
    X = df.iloc[:, feature_start_col:].astype(float).values
    if scaler is not None:
        X = scaler.transform(X)

    logits_or_probs = forward_pass(model, X.T)  # shape (out, N) or (1, N)
    arr = np.array(logits_or_probs)

    # Determine probability vector for the positive class (class index 1)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            # single-row output (1, N)
            arr1 = arr.ravel()
            probs = arr1 if (np.nanmin(arr1) >= 0.0 and np.nanmax(arr1) <= 1.0) else sigmoid(arr1)
        else:
            # multi-class output (n_classes, N)
            if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0:
                probs_matrix = arr
            else:
                # apply stable softmax per column
                exps = np.exp(arr - np.max(arr, axis=0, keepdims=True))
                probs_matrix = exps / np.sum(exps, axis=0, keepdims=True)
            if probs_matrix.shape[0] > 1:
                probs = probs_matrix[1, :]
            else:
                probs = probs_matrix.ravel()
    elif arr.ndim == 1:
        probs = arr if (np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0) else sigmoid(arr)
    else:
        probs = arr.ravel()

    probs = np.asarray(probs).ravel()

    # sanity check: probs length must match number of rows
    if probs.shape[0] != X.shape[0]:
        raise ValueError(f"prob length ({probs.shape[0]}) != rows ({X.shape[0]})")

    preds = (probs >= threshold).astype(int)
    out_df = df.copy().reset_index(drop=True)
    out_df["prob"] = probs
    out_df["pred"] = preds
    return out_df

def main():
    p = argparse.ArgumentParser(description="Run MLP predictions on CSV rows and evaluate with BCE.")
    p.add_argument("csv", help="CSV input (headerless expected like data.csv).")
    p.add_argument("--model", default="mlp_model.npz", help="Model .npz file with W0/b0, W1/b1 keys")
    p.add_argument("--scaler", default="scaler.pkl", help="scaler.pkl produced by training (joblib)")
    p.add_argument("--out", default="predictions.csv", help="Output CSV file")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary class")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print("Input CSV not found", file=sys.stderr); sys.exit(2)

    Ws, Bs = load_model_npz(args.model)
    model = build_model_from_params(Ws, Bs)

    scaler = None
    try:
        scaler = joblib.load(args.scaler)
    except Exception:
        print("Warning: scaler not found or failed to load. Input features will NOT be scaled.", file=sys.stderr)

    df = pd.read_csv(csv_path, header=None)
    # ensure label column exists for evaluation
    has_label = df.shape[1] >= 2
    if has_label:
        try:
            y_true = parse_labels(df.iloc[:, 1])
        except Exception as e:
            print("Label parse error:", e, file=sys.stderr)
            has_label = False

    results = predict_dataframe(df, model, scaler, feature_start_col=2, threshold=args.threshold)
    results.to_csv(args.out, index=False, header=False)
    print(f"Wrote predictions to {args.out}")

    if has_label:
        probs = results["prob"].to_numpy()
        preds = results["pred"].to_numpy()
        total = len(preds)
        correct = int((preds == y_true).sum())
        pct = 100.0 * correct / total if total > 0 else 0.0
        bce = binary_cross_entropy_scalar(y_true, probs)
        # print evaluation summary
        print(f"Correct: {correct}/{total} ({pct:.2f}%)")
        print(f"Binary cross-entropy (E): {bce:.6f}")

if __name__ == "__main__":
    main()