import pandas as pd
import os

DATA_DIR = "data/CSE-CICIDS-2018"

files = sorted([
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
])

for file in files:

    df = pd.read_csv(file)

    print("\n========================================")
    print("File:", os.path.basename(file))

    counts = df["Label"].value_counts()

    print("\nAttack Counts:")
    print(counts)

    print("\nAttack Percentages:")
    print((counts / len(df) * 100).round(2))