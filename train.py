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

def train_one_model(name, hidden, momentum, args, X_train, X_val, y_train, y_val, scaler):
    """Train a single model configuration and return history + params."""
    n_samples, n_features = X_train.shape
    model = [
        Dense(n_features, hidden),
        ReLU(),
        Dense(hidden, hidden),
        ReLU(),
        Dense(hidden, 2),
        Softmax()
    ]

    rng = np.random.RandomState(args.seed)
    train_losses, val_losses, val_accs, train_accs = [], [], [], []

    # Early stopping state
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_params = None

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n_samples)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        epoch_loss = 0.0
        for i in range(0, n_samples, args.batch):
            xb = X_sh[i:i + args.batch]
            yb = y_sh[i:i + args.batch]
            if xb.shape[0] == 0:
                continue
            xb_col = xb.T
            yb_row = yb.T
            y_pred = forward_pass(model, xb_col)  # (2, batch)
            yb_idx = yb_row.flatten().astype(int)
            yb_onehot = np.eye(2)[yb_idx].T
            eps = 1e-8
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            batch_loss = -np.sum(yb_onehot * np.log(y_pred_clipped)) / xb.shape[0]
            epoch_loss += batch_loss * xb.shape[0]
            grad_logits = (y_pred - yb_onehot)
            g = grad_logits
            for layer in reversed(model):
                # pass per-model momentum to layers
                g = layer.backward(g, 0.01, momentum=momentum)

        epoch_loss /= n_samples

        # validation
        val_pred = forward_pass(model, X_val.T)
        yval_idx = y_val.flatten().astype(int)
        yval_onehot = np.eye(2)[yval_idx].T
        val_pred_clipped = np.clip(val_pred, 1e-8, 1 - 1e-8)
        val_loss = -np.sum(yval_onehot * np.log(val_pred_clipped)) / y_val.shape[0]
        val_labels = np.argmax(val_pred, axis=0)
        val_acc = (val_labels == y_val.flatten().astype(int)).mean()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        # compute training accuracy on full training set for plotting
        train_pred = forward_pass(model, X_train.T)
        train_labels_pred = np.argmax(train_pred, axis=0)
        train_acc = (train_labels_pred == y_train.flatten().astype(int)).mean()
        train_accs.append(train_acc)

        # Early stopping check using current val_loss
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # save best model parameters (deep copy)
            best_params = {}
            idx = 0
            for layer in model:
                if hasattr(layer, "weights"):
                    best_params[f"W{idx}"] = layer.weights.copy()
                    best_params[f"b{idx}"] = layer.bias.copy()
                    idx += 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 5:
                print(f"Early stopping at epoch {epoch} after {epochs_no_improve} epochs with no improvement in validation loss.")
                break

        print(f"[{name}] Epoch {epoch:3d} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # save model params
    params = {}
    idx = 0
    for layer in model:
        if hasattr(layer, "weights"):
            params[f"W{idx}"] = layer.weights
            params[f"b{idx}"] = layer.bias
            idx += 1
    model_file = f"mlp_model_{name}.npz"
    np.savez(model_file, **params)
    print(f"[{name}] Model saved to {model_file}")

    # save history
    history = {
        "train_loss": np.array(train_losses),
        "val_loss": np.array(val_losses),
        "val_acc": np.array(val_accs),
        "train_acc": np.array(train_accs),
    }
    hist_file = f"history_{name}.npz"
    np.savez(hist_file, **history)
    print(f"[{name}] History saved to {hist_file}")

    return {
        "name": name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "train_accs": train_accs,
        "model_file": model_file,
        "hist_file": hist_file,
    }

def main():
    p = argparse.ArgumentParser(description="Train one or multiple MLP configs on CSV data.")
    p.add_argument("csv", help="Path to CSV file for training data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden", type=int, nargs="+", default=[64], help="One or more hidden sizes (one value per model)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--momentum", type=float, nargs="+", default=[0.0], help="Momentum values (one per model or single value)")
    p.add_argument("--val-pct", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--names", type=str, nargs="+", help="Optional names for models (one per model)")
    p.add_argument("--plot-out", help="Path to save comparison plot (PNG).")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    X_train, X_val, y_train, y_val, scaler = load_and_prepare(csv_path, seed=args.seed, val_pct=args.val_pct/100.0)
    # save scaler once (shared preprocessing)
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")

    hidden_list = args.hidden
    momentum_list = args.momentum if len(args.momentum) > 1 or len(args.hidden) == 1 else args.momentum * len(hidden_list)
    # broadcast single momentum if needed
    if len(momentum_list) == 1 and len(hidden_list) > 1:
        momentum_list = momentum_list * len(hidden_list)
    if len(momentum_list) != len(hidden_list):
        # zip will truncate; enforce same length
        raise ValueError("Number of momentum values must be 1 or match number of hidden sizes")

    if args.names:
        if len(args.names) != len(hidden_list):
            raise ValueError("If --names provided it must have same count as --hidden")
        names = args.names
    else:
        names = [f"model{i}" for i in range(len(hidden_list))]

    all_histories = []
    for name, hsize, mom in zip(names, hidden_list, momentum_list):
        print(f"Training {name}: hidden={hsize}, momentum={mom}")
        hist = train_one_model(name, hsize, mom, args, X_train, X_val, y_train, y_val, scaler)
        all_histories.append(hist)

    # plot comparison curves (use per-model epoch ranges to handle early stopping)
    plt.figure(figsize=(14, 10))

    # Training loss (top-left)
    plt.subplot(2, 2, 1)
    for h in all_histories:
        ne = len(h["train_losses"])
        ep = range(1, ne + 1)
        plt.plot(ep, h["train_losses"], label=f"{h['name']}_train")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)

    # Validation loss (bottom-left)
    plt.subplot(2, 2, 3)
    for h in all_histories:
        ne = len(h["val_losses"])
        ep = range(1, ne + 1)
        plt.plot(ep, h["val_losses"], "--", label=f"{h['name']}_val")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)

    # Train acc (top-right)
    plt.subplot(2, 2, 2)
    for h in all_histories:
        ne = len(h["train_accs"])
        ep = range(1, ne + 1)
        plt.plot(ep, h["train_accs"], "-", label=f"{h['name']}_train")
    plt.title("Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.0)
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)

    # Val acc (bottom-right)
    plt.subplot(2, 2, 4)
    for h in all_histories:
        ne_val = len(h["val_accs"])
        ep_val = range(1, ne_val + 1)
        plt.plot(ep_val, h["val_accs"], "--", label=f"{h['name']}_val")
    plt.title("Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.6, 1.0)
    plt.legend(loc="best", fontsize="small")
    plt.grid(True)

    plt.tight_layout()
    if args.plot_out:
        plt.savefig(args.plot_out, dpi=150)
        print(f"Comparison plot saved to {args.plot_out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()