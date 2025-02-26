import pandas as pd

# Define the split to use – using dev for testing
split_path = "data/dev-00000-of-00001.parquet"

# Load the dataset from Hugging Face
df = pd.read_parquet("hf://datasets/princeton-nlp/SWE-bench/" + split_path)

# Fix the 'version' column so it stays as a string
df['version'] = df['version'].apply(lambda x: f"version:{x}")

# Ensure there is an "instance_id" column.
# If it doesn't exist, create one using the DataFrame's index.
if 'instance_id' not in df.columns:
    df['instance_id'] = df.index.astype(str)

# Save the DataFrame as CSV
csv_filename = "SWE-bench.csv"
df.to_csv(csv_filename, index=False)
print(f"Dataset saved as {csv_filename} with {len(df)} rows.")