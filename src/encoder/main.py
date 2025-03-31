import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Define the dataset class
class QuestionModelPerformanceDataset(Dataset):
    def __init__(self, questions, performances, tokenizer, max_length=128):
        self.questions = questions
        self.performances = performances
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        performance = self.performances[idx]
        
        # Tokenize the input
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by the tokenizer
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'performance': torch.tensor(performance, dtype=torch.float)
        }
        
        return item

# Define the model using ModernBERT
class ModelPerformancePredictor(nn.Module):
    def __init__(self, bert_model_name, num_models):
        super(ModelPerformancePredictor, self).__init__()
        # Load the pretrained ModernBERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Get BERT embedding size
        hidden_size = self.bert.config.hidden_size
        
        # Define the prediction head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_models)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for the sequence
        sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass through the regressor to get model performance predictions
        model_performance = self.regressor(sequence_output)
        
        return model_performance

# Function to train the model
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            performance = batch['performance'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(predictions, performance)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                performance = batch['performance'].to(device)
                
                # Forward pass
                predictions = model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(predictions, performance)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    return model

# Function to preprocess the data
def preprocess_data(data_df):
    # Ensure model_performance column is not null
    data_df = data_df.dropna(subset=['model_performance'])
    
    # Extract questions
    questions = data_df['prompted_question'].tolist()
    
    # Extract model performance values
    # First, get all model names
    model_names = set()
    for perf_dict in data_df['model_performance']:
        model_names.update(perf_dict.keys())
    
    model_names = sorted(list(model_names))
    print(f"Models found: {model_names}")
    
    # Create performance arrays
    performances = []
    for perf_dict in data_df['model_performance']:
        perf_array = [perf_dict.get(model, 0.0) for model in model_names]
        performances.append(perf_array)
    
    return questions, performances, model_names

# Main function to run the entire pipeline
def run_model_performance_prediction(data_df, bert_model_name='modernbert/modernbert-base'):
    # Preprocess data
    questions, performances, model_names = preprocess_data(data_df)
    
    # Split data into train and validation sets
    q_train, q_val, p_train, p_val = train_test_split(
        questions, performances, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Create datasets
    train_dataset = QuestionModelPerformanceDataset(q_train, p_train, tokenizer)
    val_dataset = QuestionModelPerformanceDataset(q_val, p_val, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = ModelPerformancePredictor(bert_model_name, len(model_names))
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    model = train_model(model, train_loader, val_loader, optimizer, criterion, device)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, tokenizer, model_names

# Function to predict model performance for a new question
def predict_performance(question, model, tokenizer, model_names, device):
    model.to(device)
    model.eval()
    
    # Tokenize the input
    encoding = tokenizer(
        question,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
    
    # Create a dictionary mapping model names to predicted performance
    performance_dict = {
        model_name: predictions[0, i].item() 
        for i, model_name in enumerate(model_names)
    }
    
    # Get the best model
    best_model_idx = torch.argmax(predictions, dim=1).item()
    best_model = model_names[best_model_idx]
    
    return performance_dict, best_model

# Example usage
if __name__ == "__main__":
    # Sample data (in practice, load from CSV or database)
    data = {
        'prompted_question': [
            "Question: If consumption for a household is $...",
            "Question: Calculate the maximum kinetic energ...",
            "Question: Which tissue of plants most resembl...",
            "Question: Two 30\" plants are crossed, resulti..."
        ],
        'category': ["economics", "chemistry", "biology", "biology"],
        'model_performance': [
            {"biology_model": 0.1, "business_model": 0.4},
            {"biology_model": 0.1, "business_model": 0.2},
            {"biology_model": 0.9, "business_model": 0.1},
            {"biology_model": 0.9, "business_model": 0.1}
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Train the model
    model, tokenizer, model_names = run_model_performance_prediction(df)
    
    # Example inference
    new_question = "Question: What is the role of fiscal policy in controlling inflation?"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    performance_dict, best_model = predict_performance(new_question, model, tokenizer, model_names, device)
    
    print(f"Predicted performance: {performance_dict}")
    print(f"Best model: {best_model}")