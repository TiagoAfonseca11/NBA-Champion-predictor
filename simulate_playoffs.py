from src.models.predict import predict

# Exemplo super básico: equipa e stats fictícios para 4 equipas (substitui pelos dados reais depois)
teams = {
    'Lakers': {'points_per_game': 110, 'rebounds_per_game': 45, 'assists_per_game': 25, 'turnovers_per_game': 12, 'efficiency': 110},
    'Nets': {'points_per_game': 108, 'rebounds_per_game': 44, 'assists_per_game': 23, 'turnovers_per_game': 13, 'efficiency': 108},
    'Bucks': {'points_per_game': 112, 'rebounds_per_game': 46, 'assists_per_game': 26, 'turnovers_per_game': 11, 'efficiency': 112},
    'Heat': {'points_per_game': 107, 'rebounds_per_game': 43, 'assists_per_game': 24, 'turnovers_per_game': 14, 'efficiency': 107},
}

# Exemplo de bracket inicial (semi-finais)
round1 = [('Lakers', 'Heat'), ('Nets', 'Bucks')]

def simulate_round(matchups):
    winners = []
    for home, away in matchups:
        result = predict(teams[home], teams[away])
        winner = home if "Home" in result else away
        print(f"{home} vs {away} -> {winner} wins")
        winners.append(winner)
    return winners

def simulate_playoffs():
    print("Simulando semi-finais...")
    winners_sf = simulate_round(round1)

    print("Simulando final...")
    final_round = [(winners_sf[0], winners_sf[1])]
    winners_f = simulate_round(final_round)

    champion = winners_f[0]
    return champion
