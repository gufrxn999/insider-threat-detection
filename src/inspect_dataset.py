from pathlib import Path
import pandas as pd

root = Path("data/MachineLearningCSV")
files = sorted(root.rglob("*.csv"))

print(f"Found {len(files)} CSV files")

for file in files:
    print(f"\n--- {file.name} ---")
    counts = {}

    try:
        for chunk in pd.read_csv(file, chunksize=100_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                raise ValueError("Label column not found")

            labels = chunk["Label"].astype(str).str.strip().value_counts()
            for label, count in labels.items():
                counts[label] = counts.get(label, 0) + int(count)

        for label, count in sorted(counts.items()):
            print(f"{label}: {count:,}")

    except Exception as exc:
        print(f"ERROR: {exc}")
