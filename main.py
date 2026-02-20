from src.preprocess import preprocess_data
from src.train import train_model

if __name__ == "__main__":
    df = preprocess_data("data/raw.csv")
    model = train_model(df)