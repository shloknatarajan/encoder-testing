import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()


def create_mc_question_prompt(question, options):
    prompt = f"""
    Question: {question}
    Choices: {options}
    Please solve this step by step, then output your answer on a new line as 'The answer is: X'
    where X is the letter corresponding to your choice
    """
    return prompt

single_question = {"question_id": 70, "question": "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.", "options": ["Safe practices, Fear, Jealousy, Trivial", "Unsafe practices, Distress, Joy, Trivial", "Safe practices, Wants, Jealousy, Trivial", "Safe practices, Distress, Fear, Trivial", "Unsafe practices, Wants, Jealousy, Serious", "Safe practices, Distress, Jealousy, Serious", "Safe practices, Wants, Fear, Serious", "Unsafe practices, Wants, Fear, Trivial", "Unsafe practices, Distress, Fear, Serious"], "answer": "I", "answer_index": 8, "cot_content": "", "category": "business", "src": "ori_mmlu-business_ethics"}

query = create_mc_question_prompt(single_question["question"], single_question["options"])

"""
tokenizers is a function that takes a string and returns a tokenized string.
"""
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", token=os.getenv("HF_TOKEN"))
"""
model is a function that takes a tokenized string and returns the embedding aka the last hidden state of the tokenized string
as it goes through the model's layers. Tokenizer converts string into raw numbers. Model adds meaning to the numbers.
"""
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", token=os.getenv("HF_TOKEN"))

tokenized_text = tokenizer(query, return_tensors="pt")

outputs = model(**tokenized_text)

print(outputs)

# # pickle the outputs
# import pickle

# with open("outputs.pkl", "wb") as f:
#     pickle.dump(outputs, f)

