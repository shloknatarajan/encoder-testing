"""
ModernBERT Encoder
Should probably work for any encoder from huggingface / implemented in
AutoTokenizer + AutoModel
"""
import torch
from transformers import AutoModel, AutoTokenizer
import os
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

load_dotenv()

def get_modernbert_embeddings(queries, model_name="answerdotai/ModernBERT-base", max_length=512, batch_size=8, device=None):
    """
    Generate embeddings for a list of queries using a modernBERT model.
    
    Args:
        queries (List[str]): List of text queries to embed
        model_name (str): The modernBERT model to use
        max_length (int): Maximum sequence length for tokenization
        batch_size (int): Number of queries to process at once
        device (str, optional): Device to run the model on ('cuda', 'cpu'). If None, uses CUDA if available.
    
    Returns:
        List[torch.Tensor]: List of embedding tensors for each query
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
    model = AutoModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN")).to(device)
    model.eval()
    
    all_embeddings = []
    logger.info(f"Generating embeddings for {len(queries)} queries")
    # Process in batches
    for i in tqdm(range(0, len(queries), batch_size), desc="Generating embeddings"):
        batch_queries = queries[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as the sentence embedding
            # This is a common approach, but you might want to use mean pooling instead
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings (optional but recommended)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.extend(embeddings.cpu())
    
    return all_embeddings

# Example usage
if __name__ == "__main__":
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain natural language processing",
    ]
    
    embeddings = get_modernbert_embeddings(queries)
    
    for i, (query, embedding) in enumerate(zip(queries, embeddings)):
        print(f"Query {i+1}: {query}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print("-" * 50)