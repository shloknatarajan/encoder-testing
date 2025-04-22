from load_data.export_data import get_processed_dataset
from encoder.models.modernBERT import get_modernbert_embeddings
import pickle
import argparse
import os
# Note: more probably supported since we can change the name passed in
supported_embeddings = [
    "modernbert"
]

def generate_embeddings(dataset: str, embedding_type: str, save_directory: str = "encoder/saved_embeddings"):
    """
    Generate embeddings for a dataset.
    
    Args:
        dataset (str): Name of the dataset to generate embeddings for
        embedding_type (str): Type of embedding to generate
        save_directory (str, optional): Directory to save embeddings
    """
    data = get_processed_dataset(dataset)
    train_questions = data['train']['prompted_question']
    test_questions = data['test']['prompted_question']
    if embedding_type == "modernbert":
        train_embeddings, test_embeddings = get_modernbert_embeddings(train_questions), get_modernbert_embeddings(test_questions)
        # Make sure directory exists
        os.makedirs(save_directory, exist_ok=True)
        with open(f"{save_directory}/{embedding_type}_train_embeddings.pkl", "wb") as f:
            pickle.dump(train_embeddings, f)
        with open(f"{save_directory}/{embedding_type}_test_embeddings.pkl", "wb") as f:
            pickle.dump(test_embeddings, f)
        return train_embeddings, test_embeddings # NOTE: may not need to have a return type here
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for a dataset")
    parser.add_argument("--dataset", type=str, choices=["mmlu-pro"], required=True)
    parser.add_argument("--embedding-type", type=str, choices=supported_embeddings, required=True)
    parser.add_argument("--save-directory", type=str, default="encoder/saved_embeddings")
    args = parser.parse_args()
    generate_embeddings(args.dataset, args.embedding_type, args.save_directory)
