# Task 1: News Topic Classifier Using BERT

## Objective

Fine-tune a **BERT** (Bidirectional Encoder Representations from Transformers) model to classify news headlines into 4 topic categories:
- **Class 0:** World News
- **Class 1:** Sports
- **Class 2:** Business
- **Class 3:** Science/Technology

## Dataset

**AG News Dataset** (from Hugging Face Datasets)
- **Total samples:** ~120,000 (using 2,500 for training in this implementation)
- **Classes:** 4 balanced news categories
- **Task type:** Multi-class text classification

## Methodology & Approach

### 1. **Data Preprocessing**
- Load dataset from Hugging Face
- Tokenize text using BERT tokenizer
- Add special tokens: `[CLS]` and `[SEP]`
- Create attention masks for proper sequence handling
- Pad sequences to fixed length (128 tokens)

### 2. **Model Architecture**
- **Base Model:** `bert-base-uncased` (pre-trained on large corpus)
- **Task-specific head:** Sequence classification layer for 4 classes
- **Training approach:** Fine-tuning (update all BERT parameters)

### 3. **Training Configuration**
- **Optimizer:** AdamW (with weight decay)
- **Learning rate:** 2e-5 (typical for BERT fine-tuning)
- **Batch size:** 16
- **Epochs:** 3
- **Loss function:** Cross-entropy loss
- **Learning rate schedule:** Linear warmup

### 4. **Evaluation Metrics**
- **Accuracy:** Overall correctness
- **F1-Score (weighted):** Handles class imbalance
- **Confusion Matrix:** Per-class performance
- **Classification Report:** Precision, Recall, F1 per class

## Key Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~84% |
| F1-Score (Weighted) | ~0.84 |
| F1-Score (Macro) | ~0.83 |

*Note: Results may vary based on random seed and sample selection*

## Key Learnings

1. **Transfer Learning:** Leveraging pre-trained BERT saves time and improves performance
2. **Tokenization:** BERT requires specific tokenization with special tokens
3. **Fine-tuning vs. From-scratch:** Fine-tuning produces better results with less data
4. **Attention Masks:** Critical for handling variable-length sequences
5. **Learning Rate:** Small LR (2e-5) prevents catastrophic forgetting of pre-trained knowledge

## How to Run

### Prerequisites
```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn tqdm
```

#

## Run the Script
```bash
python task1_bert_classifier.py
```

#

## Steps Executed
1. Load AG News dataset
2. Tokenize and preprocess text
3. Create data loaders
4. Load pre-trained BERT model
5. Train for 3 epochs
6. Evaluate on test set
7. Generate visualizations
8. Save trained model
9. Create inference function

### Outputs
- `bert_news_classifier/` - Saved model and tokenizer
- `task1_results.png` - Training and evaluation visualizations

## Visualizations Generated

1. **Training Loss Curve** - Shows loss decreasing over epochs
2. **Confusion Matrix** - Per-class prediction accuracy
3. **Per-Class F1 Scores** - Performance breakdown by category
4. **Overall Metrics** - Accuracy, F1-weighted, F1-macro

## Deployment Options

### Streamlit Deployment Example
```python
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification# Load saved model
model = BertForSequenceClassification.from_pretrained('bert_news_classifier')
tokenizer = BertTokenizer.from_pretrained('bert_news_classifier')

st.title("News Topic Classifier")
user_text = st.text_input("Enter news headline:")

if user_text:# Tokenize
inputs = tokenizer(user_text, return_tensors='pt')# Predict
outputs = model(**inputs)
category = outputs.logits.argmax(dim=1).item()
st.write(f"Category: {['World', 'Sports', 'Business', 'Science/Tech'][category]}")
```

#

## Gradio Deployment Example
```python
import gradio as gr
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="bert-base-uncased")

def classify_news(text):
result = classifier(text, ["World", "Sports", "Business", "Science/Tech"])
return result['labels'][0]

gr.Interface(fn=classify_news, inputs="text", outputs="text").launch()
```

## Skills Gained

- NLP with Transformers
- Transfer learning and fine-tuning
- Text tokenization and preprocessing
- Evaluation metrics for classification
- Model deployment with Streamlit/Gradio

## Code Structure

```
task1_bert_classifier.py
Step 1: Import Libraries
Step 2: Load Dataset
Step 3: Data Preprocessing & Tokenization
Step 4: Create Data Loaders
Step 5: Model Setup
Step 6: Training Parameters
Step 7: Training Loop
Step 8: Final Evaluation
Step 9: Visualizations
Step 10: Save Model
Step 11: Inference Function
Conclusion
```

## Troubleshooting

**Issue:** GPU not available
- Solution: Code runs on CPU (slower but functional)

**Issue:** Dataset download slow
- Solution: First run downloads and caches data locally

**Issue:** Out of memory
- Solution: Reduce sample size or batch size

## References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)

## Author
AI/ML Internship - DevelopersHub Corporation

## License
Open source - Feel free to modify and use
