import pandas as pd
import os

def main():
    # Define paths
    base_dir = 'C:\\Users\\HP\\Documents\\bakalarka\\BERT Classifier\\scripts\\filter_articles_by_label'
    input_csv = os.path.join(base_dir, 'bert_labeled_articlesStrba.csv')
    output_csv = os.path.join(base_dir, 'filtered_malware_articlesStrba.csv')
    
    # Load articles
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} articles")
    
    # Filter for Malware articles only
    malware_df = df[df['Label'].str.lower() == 'malware'].copy()
    
    if 'Content' in malware_df.columns:
        malware_df = malware_df.drop(columns=['Content'])
    
    # Save filtered articles
    malware_df.to_csv(output_csv, index=False)
    print(f"Saved {len(malware_df)} Malware articles to {output_csv}")

if __name__ == "__main__":
    main()