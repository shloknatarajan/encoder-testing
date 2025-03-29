import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline
from dotenv import load_dotenv
import os
from load_data.export_data import get_processed_dataset
load_dotenv()

data = get_processed_dataset()

"""
tokenizers is a function that takes a string and returns a tokenized string.
"""
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", token=os.getenv("HF_TOKEN"))
"""
model is a function that takes a tokenized string and returns the embedding aka the last hidden state of the tokenized string
as it goes through the model's layers. Tokenizer converts string into raw numbers. Model adds meaning to the numbers.
"""
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", token=os.getenv("HF_TOKEN"))

tokenized_text = tokenizer(data["prompted_question"], return_tensors="pt")

outputs = model(**tokenized_text)

print(outputs)

# # pickle the outputs
# import pickle

# with open("outputs.pkl", "wb") as f:
#     pickle.dump(outputs, f)

