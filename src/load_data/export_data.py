import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset

SUPPORTED_DATASETS = [
    "mmlu-pro",
]

def get_file_path(file_name: str) -> str:
    """Return the path to a data file in the module. Helper method for loading new data"""
    module_dir = Path(os.path.dirname(__file__))
    return module_dir / "data" / file_name

def get_processed_data_path(dataset_name: str = "mmlu-pro") -> str:
    """Return the path to a data file in the module."""
    module_dir = Path(os.path.dirname(__file__))
    str_path = str(module_dir / "processed_data" / f"{dataset_name}.jsonl")
    # Make sure the file exists
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"File {str_path} does not exist")
    return str_path

def get_processed_df() -> pd.DataFrame:
    """Return the processed data dataframe."""
    return pd.read_json(get_processed_data_path(), lines=True)

def get_processed_dataset(dataset_name: str = "mmlu-pro", test_proportion: float = 0.1) -> Dataset:
    """ Return the processed data as a huggingface dataset"""
    # Load full dataset
    full_dataset = load_dataset("json", data_files=str(get_processed_data_path(dataset_name)))['train']
    
    # Split into train and test
    splits = full_dataset.train_test_split(
        test_size=test_proportion,  # Exact number of test examples
        seed=42  # For reproducibility
    )
    
    # Return the train and test splits
    return splits