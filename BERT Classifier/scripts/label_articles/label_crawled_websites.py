import json
import pandas as pd
import logging
import os
import sys
import torch
from transformers import BertTokenizer, BertForSequenceClassification

preprocessing_path = r'C:\Users\HP\Documents\bakalarka\BERT Classifier'
sys.path.append(preprocessing_path)
from preprocessing.preprocess_text import preprocess_text


def load_bert_model_and_tokenizer(model_path, tokenizer_path):
    """Loads the fine-tuned BERT model and tokenizer."""
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading BERT model or tokenizer: {e}")
        return None, None

def predict_label_with_bert(model, tokenizer, text, label_map):
    """Predicts the label for a given text using the fine-tuned BERT model."""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Tokenize text
        inputs = tokenizer(
            processed_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            # unpack the dictionary
            outputs = model(**inputs)
            predictions = outputs.logits
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Map prediction to label
        label = label_map.get(predicted_class, "Unknown")
        return label, processed_text
    except Exception as e:
        logging.error(f"Error predicting label with BERT: {e}")
        return "Unknown", text

def main():
    # Paths to BERT model and tokenizer
    model_path = 'C:\\Users\\HP\\Documents\\bakalarka\\BERT Classifier\\models\\bert_finetuned_model'
    tokenizer_path = 'C:\\Users\\HP\\Documents\\bakalarka\\BERT Classifier\\models\\bert_finetuned_tokenizer'

    # Load BERT model and tokenizer
    model, tokenizer = load_bert_model_and_tokenizer(model_path, tokenizer_path)
    if model is None or tokenizer is None:
        logging.error("BERT model or tokenizer not loaded. Exiting.")
        return

    # Define the label map for the three categories
    label_map = {
        0: 'Data_Breaches',
        1: 'Malware',
        2: 'Vulnerability'
    }

    # Correct path to the JSON file
    json_path = os.path.join('C:\\Users\\HP\\Documents\\bakalarka\\focused_crawler', 'crawledWebsites.json')

    # Read JSON data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            websites = json.load(f)
        logging.info(f"Loaded {len(websites)} websites from JSON.")
    except Exception as e:
        logging.error(f"Error reading JSON file: {e}")
        return

    # Prepare data for DataFrame
    data = {
        'Title': [],
        'Link': [],
        'Content': [],
        'Preprocessed_Content': [],
        'Label': []
    }

    for website in websites:
        title = website.get('title', '')
        link = website.get('link', '')
        article_contents = website.get('article_content', [])
        
        # Concatenate all text from article_content
        content = ' '.join([item.get('text', '') for item in article_contents])
        
        # Create combined text (title + content) similar to training
        combined_text = title + " : " + content
        
        # Predict label and get preprocessed text
        label, preprocessed_content = predict_label_with_bert(model, tokenizer, combined_text, label_map)
        
        # Append data
        data['Title'].append(title)
        data['Link'].append(link)
        data['Content'].append(content)
        data['Preprocessed_Content'].append(preprocessed_content)
        data['Label'].append(label)
        
        logging.info(f"Processed article: {title} | Label: {label}")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    output_path = os.path.join('C:/Users/HP/Documents/bakalarka/BERT Classifier/scripts/filter_articles_by_label', 'bert_labeled_articlesStrba.csv')
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Saved BERT-labeled articles to {output_path}")
    except Exception as e:
        logging.error(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    main() 