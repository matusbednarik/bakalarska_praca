import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    test_size: float = 0.10,
    val_size: float = 0.20,
    random_state: int = 42
):
    # 1) Load the full cleaned dataset
    df = pd.read_csv(input_csv)
    # Now drop any rows missing either combined_text or Label
    df = df.dropna(subset=['Label'])
    
    # 2) Split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['Label'],
        random_state=random_state
    )
    
    # 3) Split remaining into train + validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df['Label'],
        random_state=random_state
    )
    
    # 4) Save to disk
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Split into {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

if __name__ == "__main__":
    split_data(
      input_csv='C:/Users/HP/Documents/bakalarka/datasets/TheHackerNews_Dataset_withoutCyberAttacks.csv',
      train_csv='C:/Users/HP/Documents/bakalarka/datasets/article_train.csv',
      val_csv='C:/Users/HP/Documents/bakalarka/datasets/article_val.csv',
      test_csv='C:/Users/HP/Documents/bakalarka/datasets/article_test.csv'
    ) 