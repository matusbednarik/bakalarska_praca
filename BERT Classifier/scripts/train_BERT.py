import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.split_data import split_data
from preprocessing.apply_preprocessing import apply_preprocessing

# ——— Paths for splitting the raw dataset ———
RAW_CSV_PATH   = 'C:/Users/HP/Documents/bakalarka/datasets/TheHackerNews_Dataset_withoutCyberAttacksAndDuplicates.csv'
TRAIN_CSV_PATH = 'C:/Users/HP/Documents/bakalarka/datasets/article_train.csv'
VAL_CSV_PATH   = 'C:/Users/HP/Documents/bakalarka/datasets/article_val.csv'
TEST_CSV_PATH  = 'C:/Users/HP/Documents/bakalarka/datasets/article_test.csv'

def load_data():
    """Generate splits (if needed) and load train/val/test DataFrames."""
    # 1) Run split logic
    split_data(
        input_csv=RAW_CSV_PATH,
        train_csv=TRAIN_CSV_PATH,
        val_csv=VAL_CSV_PATH,
        test_csv=TEST_CSV_PATH
    )

    # 1a) Clean each split in‐place
    apply_preprocessing(TRAIN_CSV_PATH, TRAIN_CSV_PATH)
    apply_preprocessing(VAL_CSV_PATH,   VAL_CSV_PATH)
    apply_preprocessing(TEST_CSV_PATH,  TEST_CSV_PATH)

    # 2) Read them back in (now they all have a 'cleaned_text' column)
    train_df      = pd.read_csv(TRAIN_CSV_PATH)
    validation_df = pd.read_csv(VAL_CSV_PATH)
    test_df       = pd.read_csv(TEST_CSV_PATH)

    # 3) Build the BERT input column
    for df in (train_df, validation_df, test_df):
        df['combined_text'] = df['Title'] + " : " + df['cleaned_text']

    return train_df, validation_df, test_df

def map_cyberthreats(train_df, validation_df):
    """Map sentiment labels to integer classes."""
    category_map = {
        'Data_Breaches': 0,
        'Malware': 1,
        'Vulnerability': 2
    }
    y_train = train_df['Label'].map(category_map)
    y_val = validation_df['Label'].map(category_map)
    return y_train, y_val, category_map

class CyberThreatDataset(Dataset):
    """Custom Dataset for loading cyber threat data."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'f1': f1,
        # Additional metrics can be added here if needed
    }

def train_bert():
    """Train and evaluate a BERT model for cyber threat classification."""
    # Load data (and regenerate splits if needed)
    train_df, validation_df, test_df = load_data()

    # Display label distributions
    print("\nTraining Set Label Distribution:")
    print(train_df['Label'].value_counts())
    
    print("\nValidation Set Label Distribution:")
    print(validation_df['Label'].value_counts())
    
    # Map sentiments to integers
    y_train, y_val, category_map = map_cyberthreats(train_df, validation_df)
    
    # Check for any missing mappings
    if y_train.isnull().any() or y_val.isnull().any():
        raise ValueError("Some labels couldn't be mapped. Please check the 'Label' column for inconsistencies.")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = CyberThreatDataset(train_df['combined_text'], y_train, tokenizer)
    val_dataset = CyberThreatDataset(validation_df['combined_text'], y_val, tokenizer)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(category_map),
        output_attentions=False,
        output_hidden_states=False
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()

    y_pred = trainer.predict(val_dataset).predictions.argmax(-1)
    
    # Classification Report
    print(f"\n--- Classification Report (BERT) ---")
    print(classification_report(y_val, y_pred, target_names=category_map.keys(), digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix (BERT):\n{cm}")
    
    # F1 Score
    f1 = f1_score(y_val, y_pred, average='macro')
    print(f"\nMacro F1 Score: {f1:.4f}")
    
    # Save the model and tokenizer
    model.save_pretrained('C:/Users/HP/Documents/bakalarka/BERT Classifier/models/bert_finetuned_model')
    tokenizer.save_pretrained('C:/Users/HP/Documents/bakalarka/BERT Classifier/models/bert_finetuned_tokenizer')
    
    # ——— Final evaluation on the held‐out test set ———
    print(f"\n--- Test Set Classification Report (BERT) ---")
    y_test = test_df['Label'].map(category_map)
    test_dataset = CyberThreatDataset(test_df['combined_text'], y_test, tokenizer)
    test_preds = trainer.predict(test_dataset).predictions.argmax(-1)
    print(classification_report(y_test, test_preds, target_names=category_map.keys(), digits=4))
    cm_test = confusion_matrix(y_test, test_preds)
    print(f"\nTest Confusion Matrix (BERT):\n{cm_test}")
    f1_test = f1_score(y_test, test_preds, average='macro')
    print(f"\nTest Macro F1 Score: {f1_test:.4f}")

    print("\nTraining completed. Fine-tuned BERT model and tokenizer have been saved.")

if __name__ == "__main__":
    train_bert()