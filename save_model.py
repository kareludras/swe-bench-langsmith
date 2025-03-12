from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Replace these identifiers with the correct ones if needed.
TOKENIZER_MODEL_ID = "deepseek-ai/deepseek-model"
MODEL_ID = "deepseek-ai/DeepSeek-R1"

# Set the local path where you want to store the model.
# Update this path to a valid directory on your machine.
LOCAL_MODEL_PATH = "C:/Users/udras/Documents/deepseek-model"

# Create the local directory if it doesn't exist.
if not os.path.exists(LOCAL_MODEL_PATH):
    os.makedirs(LOCAL_MODEL_PATH)

print("Downloading tokenizer from:", TOKENIZER_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

print("Downloading model from:", MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

print("Saving model and tokenizer to local directory:", LOCAL_MODEL_PATH)
model.save_pretrained(LOCAL_MODEL_PATH)
tokenizer.save_pretrained(LOCAL_MODEL_PATH)

print(f"Model and tokenizer successfully saved to {LOCAL_MODEL_PATH}")
