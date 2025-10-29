# ...existing code...
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# local modules
from dense import Dense
from activations import Sigmoid, ReLU, Softmax

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def binary_cross_entropy(y_pred, y_true, eps=1e-8):
    # y_pred, y_true shape: (1, batch)
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss = - (y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return np.mean(loss)

def load_and_prepare(path, seed=42, val_pct=0.2):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 2:].astype(float).values  # features (n_samples, n_features)
    labels = df.iloc[:, 1].astype(str).str.upper()
    y = (labels == "M").astype(np.float32).to_numpy().reshape(-1, 1)  # malignant=1, benign=0
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_pct, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, scaler

def forward_pass(model, x):
    out = x
    for layer in model:
        out = layer.forward(out)
    return out

def main():
    p = argparse.ArgumentParser(description="Train a simple MLP on CSV data.")
    p.add_argument("csv", help="Path to CSV file for training data")
    p.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--hidden", type=int, default=64, help="Number of hidden units")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--val-pct", type=float, default=20.0, help="Validation percentage")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--plot-out", help="Path to save loss/accuracy plot (PNG). If not provided the plot will be shown interactively.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    X_train, X_val, y_train, y_val, scaler = load_and_prepare(csv_path, seed=args.seed, val_pct=args.val_pct/100.0)

    n_samples, n_features = X_train.shape

    # Build basic 2 hidden layer MLP
    model = [
        Dense(n_features, args.hidden),
        ReLU(),
        Dense(args.hidden, args.hidden),
        ReLU(),
        Dense(args.hidden, 1),
        Sigmoid()
    ]

    rng = np.random.RandomState(args.seed)
    # storage for plotting
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(1, args.epochs + 1):
        # shuffle
        perm = rng.permutation(n_samples)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        epoch_loss = 0.0
        # mini-batch training
        for i in range(0, n_samples, args.batch):
            xb = X_sh[i:i + args.batch]
            yb = y_sh[i:i + args.batch]
            if xb.shape[0] == 0:
                continue
            # convert to column-batch layout expected by layers: (features, batch)
            xb_col = xb.T  # (n_features, batch)
            yb_row = yb.T  # (1, batch)

            # logits = forward_pass(model, xb_col)  # (1, batch)
            # y_pred = sigmoid(logits)
            # forward_pass already runs the last Sigmoid layer, so this returns probabilities
            y_pred = forward_pass(model, xb_col)  # (1, batch)

            loss = binary_cross_entropy(y_pred, yb_row)
            epoch_loss += loss * xb.shape[0]

            # gradient of loss wrt logits (dL/dz) for BCE with sigmoid = (y_pred - y)
            grad_logits = (y_pred - yb_row)  # (1, batch)
            # backpropagate through model (last layer is Dense -> will accept grad_logits)
            g = grad_logits
            for layer in reversed(model):
                g = layer.backward(g, args.lr)

        epoch_loss /= n_samples

        # validation
        val_pred = forward_pass(model, X_val.T)  # probabilities
        val_loss = binary_cross_entropy(val_pred, y_val.T)
        val_labels = (val_pred >= 0.5).astype(int).T
        val_acc = (val_labels == y_val).mean()

        print(f"Epoch {epoch:3d} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # record metrics for plotting
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    # after training and before exit
    # save scaler and simple metadata
    joblib.dump(scaler, "scaler.pkl")
    meta = {"feature_start_col": 2, "label_map": {"M": 1, "B": 0}, "threshold": 0.5}
    np.savez("mlp_meta.npz", **meta)
    print("Scaler saved to scaler.pkl and metadata to mlp_meta.npz")

    # optional: save weights (simple np.savez)
    params = {}
    idx = 0
    for layer in model:
        if hasattr(layer, "weights"):
            params[f"W{idx}"] = layer.weights
            params[f"b{idx}"] = layer.bias
            idx += 1
    np.savez("mlp_model.npz", **params)
    print("Model saved to mlp_model.npz")

    # Plot training curves (loss & accuracy)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accs, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if args.plot_out:
        plt.savefig(args.plot_out, dpi=150)
        print(f"Training plot saved to {args.plot_out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()