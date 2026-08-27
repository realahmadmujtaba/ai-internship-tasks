"""
========================================
TASK 1: NEWS TOPIC CLASSIFIER USING BERT
========================================

OBJECTIVE:
Fine-tune a BERT model to classify news headlines into topic categories.

WHAT IS BERT?
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained 
language model that understands the context of words in sentences. We'll fine-tune 
it for our news classification task.

SKILLS GAINED:
- NLP using Transformers
- Transfer learning & fine-tuning
- Evaluation metrics for text classification
- Model deployment with Streamlit

STEP 1: IMPORT REQUIRED LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For data handling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# For BERT and transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("✓ All libraries imported successfully!")


"""
STEP 2: LOAD AG NEWS DATASET
The AG News dataset contains news articles categorized into 4 classes:
- Class 0: World News
- Class 1: Sports
- Class 2: Business
- Class 3: Science/Technology
"""

print("\n" + "="*50)
print("LOADING AG NEWS DATASET")
print("="*50)

# Load the dataset from Hugging Face
dataset = load_dataset('ag_news')

# Convert to pandas for easier exploration
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Rename columns for clarity
train_df.columns = ['label', 'text']
test_df.columns = ['label', 'text']

print(f"\nDataset loaded successfully!")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Display sample data
print("\n--- Sample Data ---")
print(train_df.head(3))

# Class distribution
print("\n--- Class Distribution (Training Set) ---")
class_names = ['World', 'Sports', 'Business', 'Science/Tech']
for idx, label in enumerate(class_names):
    count = (train_df['label'] == idx).sum()
    percentage = (count / len(train_df)) * 100
    print(f"Class {idx} ({label}): {count} samples ({percentage:.2f}%)")


"""
STEP 3: DATA PREPROCESSING
For BERT, we need to:
1. Tokenize the text (convert words to numerical tokens)
2. Add special tokens [CLS] and [SEP]
3. Create attention masks
4. Pad sequences to fixed length
"""

print("\n" + "="*50)
print("DATA PREPROCESSING & TOKENIZATION")
print("="*50)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(df, tokenizer, max_length=128, sample_size=None):
    """
    Tokenize and preprocess the text data
    
    Parameters:
    - df: pandas DataFrame with 'text' and 'label' columns
    - tokenizer: BERT tokenizer
    - max_length: maximum length of tokenized sequences
    - sample_size: use subset of data for faster training (set None for full dataset)
    
    Returns:
    - Processed tensors for model input
    """
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    input_ids = []
    attention_masks = []
    labels = []
    
    print(f"\nTokenizing {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Tokenize the text
        encoded = tokenizer.encode_plus(
            row['text'],
            add_special_tokens=True,      # Add [CLS] and [SEP] tokens
            max_length=max_length,
            padding='max_length',          # Pad to max_length
            truncation=True,               # Truncate if longer than max_length
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())
        labels.append(row['label'])
    
    # Convert lists to tensors
    input_ids_tensor = torch.stack(input_ids)
    attention_masks_tensor = torch.stack(attention_masks)
    labels_tensor = torch.tensor(labels)
    
    return input_ids_tensor, attention_masks_tensor, labels_tensor


# For faster training/testing, let's use a sample (you can use full dataset)
# Using 2000 samples for training and 500 for testing
print("\nProcessing training data...")
train_inputs, train_masks, train_labels = preprocess_text(
    train_df, tokenizer, max_length=128, sample_size=2000
)

print("\nProcessing test data...")
test_inputs, test_masks, test_labels = preprocess_text(
    test_df, tokenizer, max_length=128, sample_size=500
)

print("\n✓ Preprocessing complete!")
print(f"Train input shape: {train_inputs.shape}")
print(f"Test input shape: {test_inputs.shape}")


"""
STEP 4: CREATE DATA LOADERS
DataLoaders help us efficiently batch the data for training
"""

print("\n" + "="*50)
print("CREATING DATA LOADERS")
print("="*50)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

# Create DataLoaders
batch_size = 16

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

test_dataloader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=batch_size
)

print(f"✓ DataLoaders created successfully!")
print(f"Training batches: {len(train_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")


"""
STEP 5: MODEL SETUP
Load pre-trained BERT and configure for classification
"""

print("\n" + "="*50)
print("LOADING PRE-TRAINED BERT MODEL")
print("="*50)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained BERT model for sequence classification
# num_labels=4 because we have 4 news categories
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False
)

# Move model to device
model.to(device)

print("✓ BERT model loaded successfully!")
print(f"Model parameters: {model.num_parameters():,}")


"""
STEP 6: SETUP TRAINING
Configure optimizer, learning rate schedule, and training parameters
"""

print("\n" + "="*50)
print("SETTING UP TRAINING PARAMETERS")
print("="*50)

# Number of training epochs
epochs = 3

# Optimizer with a small learning rate (typical for fine-tuning)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Total number of training steps
total_steps = len(train_dataloader) * epochs

# Learning rate scheduler (linearly decreases LR over time)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Total training steps: {total_steps}")
print(f"Learning rate: 2e-5")


"""
STEP 7: TRAINING LOOP
Fine-tune BERT on our news classification task
"""

print("\n" + "="*50)
print("TRAINING THE MODEL")
print("="*50)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Unpack batch
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation/test data"""
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks
            )
            
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


# Training history
training_history = {'loss': []}

# Train the model
print("\nStarting training...\n")
for epoch in range(epochs):
    print(f"\n{'='*40}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*40}")
    
    # Train
    avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    training_history['loss'].append(avg_loss)
    
    print(f"Average training loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    predictions, true_labels = evaluate(model, test_dataloader, device)
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1-Score (weighted): {f1:.4f}")

print("\n✓ Training complete!")


"""
STEP 8: FINAL EVALUATION & METRICS
"""

print("\n" + "="*50)
print("FINAL EVALUATION RESULTS")
print("="*50)

# Get final predictions
final_predictions, final_true_labels = evaluate(model, test_dataloader, device)

# Calculate metrics
accuracy = accuracy_score(final_true_labels, final_predictions)
f1 = f1_score(final_true_labels, final_predictions, average='weighted')
f1_macro = f1_score(final_true_labels, final_predictions, average='macro')

print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1-Score (weighted): {f1:.4f}")
print(f"F1-Score (macro): {f1_macro:.4f}")

# Detailed classification report
print("\n--- Classification Report ---")
print(classification_report(
    final_true_labels, 
    final_predictions,
    target_names=class_names
))


"""
STEP 9: VISUALIZATIONS
"""

print("\n" + "="*50)
print("GENERATING VISUALIZATIONS")
print("="*50)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training Loss
axes[0, 0].plot(range(1, epochs+1), training_history['loss'], marker='o', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Epochs')
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix
cm = confusion_matrix(final_true_labels, final_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Per-class F1 Scores
from sklearn.metrics import f1_score as f1_per_class
f1_scores = [f1_score(final_true_labels == i, final_predictions == i, average='binary') 
             for i in range(4)]
axes[1, 0].bar(class_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
axes[1, 0].set_title('Per-Class F1 Scores')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_ylim([0, 1])
for i, v in enumerate(f1_scores):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

# 4. Model Accuracy Metrics
metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
values = [accuracy, f1, f1_macro]
colors = ['#2ecc71', '#3498db', '#e74c3c']
axes[1, 1].bar(metrics, values, color=colors)
axes[1, 1].set_title('Overall Model Metrics')
axes[1, 1].set_ylim([0, 1])
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/task1_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'task1_results.png'")
plt.show()


"""
STEP 10: SAVE THE MODEL
"""

print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

model.save_pretrained('/mnt/user-data/outputs/bert_news_classifier')
tokenizer.save_pretrained('/mnt/user-data/outputs/bert_news_classifier')

print("✓ Model saved to 'bert_news_classifier' directory")


"""
STEP 11: CREATE INFERENCE FUNCTION
For easy predictions on new headlines
"""

print("\n" + "="*50)
print("INFERENCE FUNCTION")
print("="*50)

def predict_news_category(text, model, tokenizer, device, class_names):
    """
    Predict the category of a news headline
    
    Parameters:
    - text: news headline string
    - model: trained BERT model
    - tokenizer: BERT tokenizer
    - device: torch device
    - class_names: list of class names
    
    Returns:
    - Predicted class name and confidence score
    """
    
    model.eval()
    
    # Tokenize
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence


# Test with sample headlines
print("\n--- Sample Predictions ---")
sample_texts = [
    "Latest technology breakthrough in AI and machine learning",
    "England wins the World Cup final",
    "Global stock markets reach new heights",
    "Climate change impacts on global economy"
]

for text in sample_texts:
    category, confidence = predict_news_category(text, model, tokenizer, device, class_names)
    print(f"\nText: {text}")
    print(f"Predicted: {category} (Confidence: {confidence:.4f})")


print("\n" + "="*50)
print("✓ TASK 1 COMPLETE!")
print("="*50)
print("\nKey Achievements:")
print("✓ Loaded and explored AG News dataset")
print("✓ Preprocessed and tokenized text data")
print("✓ Fine-tuned BERT model")
print("✓ Evaluated with accuracy and F1-score")
print("✓ Created visualizations")
print("✓ Built inference function for new predictions")
print("\nNext Step: Deploy with Streamlit for interactive use!")
