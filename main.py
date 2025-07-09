import sys
from src.models.train_model import train
from src.models.predict import predict
from simulate_playoffs import simulate_playoffs  # vamos criar esta função

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [train|predict|simulate] [args]")
        return

    command = sys.argv[1]

    if command == "train":
        train()
        print("Modelo treinado e guardado!")

    elif command == "predict":
        # Exemplo simples: python main.py predict ppg_home rpg_home apg_home tpg_home eff_home ppg_away rpg_away apg_away tpg_away eff_away
        if len(sys.argv) != 12:
            print("Uso: python main.py predict ppg_home rpg_home apg_home tpg_home eff_home ppg_away rpg_away apg_away tpg_away eff_away")
            return
        home_stats = {
            'points_per_game': float(sys.argv[2]),
            'rebounds_per_game': float(sys.argv[3]),
            'assists_per_game': float(sys.argv[4]),
            'turnovers_per_game': float(sys.argv[5]),
            'efficiency': float(sys.argv[6])
        }
        away_stats = {
            'points_per_game': float(sys.argv[7]),
            'rebounds_per_game': float(sys.argv[8]),
            'assists_per_game': float(sys.argv[9]),
            'turnovers_per_game': float(sys.argv[10]),
            'efficiency': float(sys.argv[11])
        }
        result = predict(home_stats, away_stats)
        print("Prediction:", result)

    elif command == "simulate":
        # Simula os playoffs e dá o campeão previsto
        champion = simulate_playoffs()
        print(f"Campeão NBA previsto: {champion}")

    else:
        print("Comando inválido. Usa train, predict ou simulate.")

if __name__ == "__main__":
    main()
