import pandas as pd

# Load the two CSV files
EV_results = pd.read_csv("test_results_8e6145cac74b47d68d4ad6456a19d3a6_50000_mean.csv")
EMSG_results = pd.read_csv("EMGSC/test_results_EMSGC_mean.csv")

# Ensure both files have a common column for matching (e.g., "sequence_name")
common_column = "Sequence Name"  # Change this if your column has a different name

# Merge to keep only rows in df2 that match df1
df_merged = pd.merge(EMSG_results, EV_results, on=common_column, how="left")  # Use "inner" to exclude non-matching rows

# Save the output
df_merged.to_csv("stacked_output_final_50000.csv", index=False)

print(df_merged.head())

