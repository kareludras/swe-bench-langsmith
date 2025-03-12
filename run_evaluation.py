"""Run an evaluation by using LangSmith to execute predictions on a dataset.

This script demonstrates how to fetch examples from a dataset, call the
`predict` function (imported from `predict_deepseek_api`), and evaluate
the results using `langsmith`.
"""

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langsmith import Client, evaluate

# Import the predict function from predict_deepseek_api.py
from predict_deepseek_api import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def run_evaluation(dataset_id: str) -> None:
    """Retrieve examples from the given dataset and run the `predict` function.

    Args:
        dataset_id (str): The ID of the dataset in LangSmith from which
            examples will be fetched.

    Raises:
        ValueError: If the dataset has no examples.
    """
    client = Client()

    # Retrieve all examples from the dataset
    examples = list(client.list_examples(dataset_id=dataset_id))
    logger.info("Number of examples retrieved: %d", len(examples))

    if not examples:
        raise ValueError(
            "No examples found in the dataset. Check your CSV and upload process."
        )

    # Call evaluate, which will invoke the predict function on each example
    results: List[Dict[str, Any]] = evaluate(predict, data=examples)
    logger.info("Predictions generated for all examples.")

    # Optionally, print out each output for review
    for run in results:
        logger.info("Output for run:\n%s", run["run"].outputs)


def main() -> None:
    """Main entry point for running the evaluation."""
    dataset_id_env = os.environ.get("DATASET_ID")

    if not dataset_id_env:
        raise ValueError(
            "Please set the DATASET_ID environment variable in .env or your environment."
        )

    run_evaluation(dataset_id_env)


if __name__ == "__main__":
    main()
