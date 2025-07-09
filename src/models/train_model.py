import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    # Exemplo simples, dados fictícios
    data = {
        'points_per_game_home': [110, 105, 112, 108],
        'rebounds_per_game_home': [45, 42, 48, 44],
        'assists_per_game_home': [25, 20, 27, 22],
        'turnovers_per_game_home': [12, 15, 10, 13],
        'efficiency_home': [110, 105, 115, 108],

        'points_per_game_away': [105, 110, 109, 111],
        'rebounds_per_game_away': [43, 44, 40, 46],
        'assists_per_game_away': [22, 25, 21, 23],
        'turnovers_per_game_away': [14, 11, 15, 12],
        'efficiency_away': [106, 108, 107, 110],

        'result': [1, 0, 1, 0]  # 1 = home win, 0 = away win
    }

    df = pd.DataFrame(data)
    X = df.drop('result', axis=1)
    y = df['result']

    model = RandomForestClassifier()
    model.fit(X, y)

    # Garante que a pasta existe
    if not os.path.exists('src/models'):
        os.makedirs('src/models')

    joblib.dump(model, 'src/models/nba_predictor.pkl')


if __name__ == '__main__':
    train()
