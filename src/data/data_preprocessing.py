import pandas as pd
import os

def preprocess():
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')

    df = pd.read_csv('data/raw/games.csv')

    # Por enquanto só copiamos (podes adicionar limpeza e feature engineering aqui)
    df.to_csv('data/processed/games_processed.csv', index=False)
    print("Dados processados e guardados em data/processed/games_processed.csv")

if __name__ == '__main__':
    preprocess()
