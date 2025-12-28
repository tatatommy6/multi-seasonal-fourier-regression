import pandas as pd
import numpy as np

CSV_PATH = "benchmark/test/LD2011_2014_converted.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    house_cols = [c for c in df.columns if c.startswith("MT_")]

    y = df[house_cols].values.astype("float64")

    mu = y.mean(axis=0)
    sigma = y.std(axis=0)
    max_v = y.max(axis=0)
    ratio = max_v / (mu + 1e-6)

    stats = pd.DataFrame({
        "house": house_cols,
        "mean": mu,
        "std": sigma,
        "max": max_v,
        "max_mean_ratio": ratio
    })

    print("Top mean households")
    print(stats.sort_values("mean", ascending=False).head(10))
    print()
    print("Top max/mean ratio households")
    print(stats.sort_values("max_mean_ratio", ascending=False).head(10))

    stats.to_csv("household_stats.csv", index=False)

if __name__ == "__main__":
    main()
