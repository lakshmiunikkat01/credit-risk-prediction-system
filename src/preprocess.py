import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):
    df = pd.read_csv(path)

    print("Initial Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nAfter Handling Missing Values:\n", df.isnull().sum())

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    print("\nFinal Shape:", df.shape)

    return df