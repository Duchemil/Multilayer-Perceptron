import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def detect_label_series(df):
    # try to pick a reasonable label column: second column if it looks categorical
    if df.shape[1] >= 2:
        s = df.iloc[:, 1]
        # consider categorical if non-numeric or low cardinality
        try:
            numeric = pd.to_numeric(s, errors="coerce").notna().all()
        except Exception:
            numeric = False
        if (not numeric) or (s.nunique() <= 50):
            return s
    return None

def main():
    p = argparse.ArgumentParser(description="Split CSV into train/evaluation parts.")
    p.add_argument("csv", help="Path to CSV file to split")
    p.add_argument("-p", "--percent", type=float, default=10.0,
                   help="Evaluation percentage (0-100). Example: -p 20 => 20%% evaluation, 80%% train")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    if not (0 <= args.percent <= 100):
        print("Percent must be between 0 and 100", file=sys.stderr)
        sys.exit(2)

    # Read CSV without header to keep original format (your data.csv appears headerless)
    df = pd.read_csv(csv_path, header=None)

    val_frac = args.percent / 100.0
    if val_frac == 0:
        print("Validation fraction is 0% — nothing to do.", file=sys.stderr)
        sys.exit(0)
    if val_frac == 1:
        print("Validation fraction is 100% — nothing to do.", file=sys.stderr)
        sys.exit(0)

    stratify_series = detect_label_series(df)
    stratify = stratify_series if stratify_series is not None else None

    train_df, val_df = train_test_split(
        df,
        test_size=val_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify
    )

    train_out = csv_path.with_name("data_train.csv")
    val_out = csv_path.with_name("data_eval.csv")

    # write without header and without index to keep same format as input
    train_df.to_csv(train_out, header=False, index=False)
    val_df.to_csv(val_out, header=False, index=False)

    print(f"Wrote {len(train_df)} rows to {train_out}")
    print(f"Wrote {len(val_df)} rows to {val_out}")

    if stratify_series is not None:
        def counts_series(df):
            return df.iloc[:, 1].value_counts(normalize=True).sort_index()
        print("\nClass proportions (train):")
        print(counts_series(train_df).to_string())
        print("\nClass proportions (val):")
        print(counts_series(val_df).to_string())

if __name__ == "__main__":
    main()