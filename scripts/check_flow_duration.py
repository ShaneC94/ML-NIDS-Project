import pandas as pd
from src.preprocess import load_and_preprocess_2018

# Sanity check for flow_duration feature in the first training day (Feb 14, 2018)
# This script is meant to be a quick diagnostic to verify that the flow_duration feature does not contain negative values, 
# which would be anomalous and could indicate issues in data preprocessing or collection. If negative values are found, 
# it will print out a summary and sample of those values for further investigation.

X, y = load_and_preprocess_2018("data/CSE-CICIDS-2018/02-14-2018.csv")

print("flow_duration summary:")
print(X["flow_duration"].describe())

neg_count = (X["flow_duration"] < 0).sum()
print(f"\nNegative flow_duration values: {neg_count}")

if neg_count > 0:
    print("Sample negative values:")
    print(X.loc[X["flow_duration"] < 0, "flow_duration"].head())
