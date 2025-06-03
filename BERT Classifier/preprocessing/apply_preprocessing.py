import pandas as pd
from preprocessing.preprocess_text import preprocess_text

def apply_preprocessing(input_csv: str, output_csv: str):
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded {input_csv}")
    except Exception as e:
        print(f"Error loading {input_csv}: {e}")
        raise

    # Drop rows missing the raw article text
    df = df.dropna(subset=['Article'])

    df['cleaned_text']  = df['Article'].apply(preprocess_text)
    df = df[df['cleaned_text'].str.strip() != '']

    # Save cleaned file
    try:
        df.to_csv(output_csv, index=False)
        print(f"Saved cleaned data: {output_csv}")
    except Exception as e:
        print(f"Error saving {output_csv}: {e}")
        raise

if __name__ == "__main__":
    base = 'C:/Users/HP/Documents/bakalarka/datasets'
    apply_preprocessing(f"{base}/article_train.csv", f"{base}/article_train_cleaned.csv")
    apply_preprocessing(f"{base}/article_val.csv",   f"{base}/article_val_cleaned.csv")
    apply_preprocessing(f"{base}/article_test.csv",  f"{base}/article_test_cleaned.csv") 