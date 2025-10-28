import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# local modules
from dense import Dense
from activations import Sigmoid

def load_csv_binary(path):
    df = pd.read_csv(path, header=None)
    # assume col0=id, col1=label (M/B), rest numeric features
    X = df.iloc[:, 2:].astype(float).to_numpy()  # shape (n_samples, n_features)
    y_raw = df.iloc[:, 1].astype(str).str.upper()
    y = (y_raw == "M").astype(np.float32).reshape(-1, 1)  # malignant=1, benign=0
    return X, y

def build_model(input_dim):
    # simple 2-layer MLP: input -> Dense(16) + Sigmoid -> Dense(1) + Sigmoid
    return [
        Dense(input_dim, 16),
        Sigmoid(),
        Dense(16, 1),
        Sigmoid()
    ]

def forward_pass(layers, x):
    out = x.T  # convert to column vector shape (features, batch) expected by Dense
    for layer in layers:
        out = layer.forward(out)
    return out

def compute_bce_loss(y_pred, y_true):
    # y_pred, y_true are shape (1, batch)
    eps = 1e-8
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss = - (y_true * np.log(y_pred_clipped.T) + (1 - y_true) * np.log(1 - y_pred_clipped.T))
    return loss.mean()

def bce_grad(y_pred, y_true):
    # gradient dL/dy_pred for sigmoid output
    # shapes: y_pred (1, batch), y_true (batch,1) -> make consistent
    yp = y_pred
    yt = y_true.T
    eps = 1e-8
    return -(yt / (yp + eps)) + ((1 - yt) / (1 - yp + eps))

def train(args):
    X, y = load_csv_binary(args.csv)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_pct/100.0,
                                                      random_state=args.seed, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    n_samples, n_features = X_train.shape
    model = build_model(n_features)

    for epoch in range(1, args.epochs + 1):
        # shuffle
        perm = np.random.RandomState(args.seed + epoch).permutation(n_samples)
        X_train_sh = X_train[perm]
        y_train_sh = y_train[perm]

        epoch_loss = 0.0
        for i in range(0, n_samples, args.batch):
            xb = X_train_sh[i:i+args.batch]
            yb = y_train_sh[i:i+args.batch]
            # forward
            y_pred = forward_pass(model, xb)  # shape (1, batch)
            # loss
            loss = compute_bce_loss(y_pred, yb)
            epoch_loss += loss * len(xb)
            # gradient
            grad = bce_grad(y_pred, yb)  # shape (1, batch)
            # backward through layers in reverse
            g = grad
            for layer in reversed(model):
                g = layer.backward(g, args.lr)
        epoch_loss /= n_samples

        # validation
        y_val_pred = forward_pass(model, X_val)
        val_loss = compute_bce_loss(y_val_pred, y_val)
        val_pred_labels = (y_val_pred.T >= 0.5).astype(int)
        val_acc = (val_pred_labels == y_val).mean()

        print(f"Epoch {epoch:3d} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # save dense params
    params = {}
    idx = 0
    for layer in model:
        if hasattr(layer, "weights"):
            params[f"W{idx}"] = layer.weights
            params[f"b{idx}"] = layer.bias
            idx += 1
    np.savez(args.model_out, **params)
    print("Model saved to", args.model_out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="path to data.csv")
    p.add_argument("--val-pct", type=float, default=20.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-out", default="model.npz")
    args = p.parse_args()
    train(args)

if __name__ == "__main__":
    main()