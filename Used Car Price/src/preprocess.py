import pandas as pd

def preprocess_data(df):
    # Drop unwanted column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Convert km_driven to integer
    df['km_driven'] = df['km_driven'].str.replace(',', '').astype(int)

    return df
