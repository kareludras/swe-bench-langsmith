from langsmith import Client
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables (ensure your .env file contains your API keys)
from dotenv import load_dotenv
load_dotenv()

csv_path = "SWE-bench.csv"
df = pd.read_csv(csv_path)
print("CSV rows:", len(df))

client = Client()

dataset = client.upload_csv(
    csv_file=csv_path,
    input_keys=list(df.columns),  # Use all columns from the CSV
    output_keys=[],               # No output fields needed here
    name="SWE-bench Dataset Updated",
    description="Updated CSV with instance_id for evaluation",
    data_type="kv"
)
print("Dataset uploaded with ID:", dataset.id)
