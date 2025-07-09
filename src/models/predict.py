import joblib
import pandas as pd
import os

def predict(home_stats, away_stats):
    model_path = os.path.join(os.path.dirname(__file__), 'nba_predictor.pkl')
    model = joblib.load(model_path)

    # Junta as features em 1 linha
    data = {}
    for key, value in home_stats.items():
        data[f"{key}_home"] = [value]
    for key, value in away_stats.items():
        data[f"{key}_away"] = [value]

    X_new = pd.DataFrame(data)
    pred = model.predict(X_new)
    return "Home team wins" if pred[0] == 1 else "Away team wins"


if __name__ == '__main__':
    # Exemplo básico
    home = float(input("Home team avg points: "))
    away = float(input("Away team avg points: "))
    resultado = predict(home, away)
    print("Prediction:", resultado)
