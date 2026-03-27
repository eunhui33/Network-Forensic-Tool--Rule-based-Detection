import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\3eunh\OneDrive\Desktop\NS\code\data\02_EDA_inf.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.2,           
    stratify=df['Label'],   
    random_state=42
)

train_df.to_csv(r'C:\Users\3eunh\OneDrive\Desktop\NS\code\data\03_train.csv', index=False)
test_df.to_csv(r'C:\Users\3eunh\OneDrive\Desktop\NS\code\data\03_test.csv', index=False)
