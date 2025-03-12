"""Generate patch predictions using the DeepSeek API.

This module uses environment variables (from a .env file) to configure
communication with the DeepSeek API, and provides a `predict` function
for generating patches based on a problem statement or hints.
"""

import logging
import os
import time
from typing import Dict

import openai
from dotenv import load_dotenv

# Try importing Timeout from openai.error; if not available, fallback to the built-in TimeoutError.
try:
    from openai.error import Timeout as OpenAITimeout
except ImportError:
    OpenAITimeout = TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Retrieve your DeepSeek API key from environment variables
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "Please set your DEEPSEEK_API_KEY in the environment or in a .env file."
    )

# Set the base URL for the DeepSeek API.
# According to the documentation, you can use either https://api.deepseek.com or https://api.deepseek.com/v1.
# Here we use https://api.deepseek.com/v1.
openai.api_base = "https://api.deepseek.com/v1"

# Set your API key in the OpenAI client.
openai.api_key = DEEPSEEK_API_KEY


def predict(inputs: Dict[str, str]) -> Dict[str, str]:
    """
    Generate a patch prediction using the DeepSeek API.

    Expected input keys:
      - instance_id: A unique identifier for this example.
      - problem_statement: The main issue description.
      - repo: Repository name.
      - base_commit: Commit identifier.
      - hints_text: (Optional) Additional hints if problem_statement is missing.

    Returns a dictionary with the following keys:
      - instance_id: Echoes the input instance ID.
      - model_patch: The generated patch text or an error message.
      - model_name_or_path: Identifier for the model ("DeepSeek API").

    Raises:
      ValueError: If no problem statement or hints text is provided.
    """
    instance_id = inputs.get("instance_id", "unknown")
    problem_statement = inputs.get("problem_statement", "").strip()

    # Fallback to hints_text if no problem_statement provided.
    if not problem_statement:
        problem_statement = inputs.get("hints_text", "").strip()

    if not problem_statement:
        logger.error(
            "No problem statement or hints_text provided for instance_id '%s'.", instance_id
        )
        return {
            "instance_id": instance_id,
            "model_patch": (
                "Error: No problem statement or hints provided. "
                "Please supply a detailed problem statement or hints_text."
            ),
            "model_name_or_path": "DeepSeek API",
        }

    prompt = (
        f"Problem Statement:\n{problem_statement}\n\n"
        "Generate a diff patch to fix this issue."
    )
    model_choice = "deepseek-reasoner"
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in generating code patches."},
        {"role": "user", "content": prompt},
    ]

    logger.info("[%s] Sending request to DeepSeek with prompt:\n%s", instance_id, prompt)
    start_time = time.time()
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model_choice,
                messages=messages,
                stream=False,
                request_timeout=60,  # 60-second timeout
            )
            elapsed = time.time() - start_time
            logger.info("[%s] API call completed in %.2f seconds on attempt %d.", instance_id, elapsed, attempt)
            generated_text = response.choices[0].message.content.strip()
            return {
                "instance_id": instance_id,
                "model_patch": generated_text,
                "model_name_or_path": "DeepSeek API",
            }
        except OpenAITimeout as exc:
            logger.warning("[%s] Timeout on attempt %d/%d: %s", instance_id, attempt, max_retries, exc)
            # Exponential backoff before retrying.
            if attempt < max_retries:
                backoff = 2 ** attempt  # 2, 4, 8 seconds, etc.
                logger.info("[%s] Retrying in %d seconds...", instance_id, backoff)
                time.sleep(backoff)
            else:
                return {
                    "instance_id": instance_id,
                    "model_patch": f"Error during API call (timeout): {exc}",
                    "model_name_or_path": "DeepSeek API",
                }
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.exception(
                "[%s] API call failed after %.2f seconds on attempt %d: %s", instance_id, elapsed, attempt, str(exc)
            )
            return {
                "instance_id": instance_id,
                "model_patch": f"Error during API call: {str(exc)}",
                "model_name_or_path": "DeepSeek API",
            }

    return {
        "instance_id": instance_id,
        "model_patch": "Error: Unknown error occurred.",
        "model_name_or_path": "DeepSeek API",
    }


def main() -> None:
    """Demonstrates a sample call to the `predict` function."""
    sample_input = {
        "instance_id": "sample1",
        "repo": "example-repo",
        "base_commit": "abcdef123456",
        "problem_statement": (
            "In file commands.py, the '--FIX-EVEN-UNPARSABLE' option is misnamed. "
            "It should be '--fix-even-unparsable'. Generate a diff patch to correct this."
        ),
        "hints_text": (
            "Option naming should be lowercase and consistent. Please generate a patch."
        ),
    }

    prediction = predict(sample_input)
    logger.info("DeepSeek API Prediction:\n%s", prediction)


if __name__ == "__main__":
    main()
