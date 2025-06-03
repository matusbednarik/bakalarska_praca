import pandas as pd

# Load the Excel file
df = pd.read_excel("C:/Users/HP/Documents/bakalarka/datasets/TheHackerNews_Dataset_withoutCyberAttacks.xlsx")

# Sort so that 'Malware' rows come first
df['MalwarePriority'] = df['Label'].apply(lambda x: 1 if x == 'Malware' else 0)
df = df.sort_values(by='MalwarePriority', ascending=False)

# Drop duplicates based on the 'Article' column (keep the first, which is Malware if present)
df_unique = df.drop_duplicates(subset='Article', keep='first')

# Drop helper column
df_unique = df_unique.drop(columns=['MalwarePriority'])

# Save the cleaned data
df_unique.to_excel("C:/Users/HP/Documents/bakalarka/datasets/TheHackerNews_Dataset_withoutCyberAttacksCleaned.xlsx", index=False)
