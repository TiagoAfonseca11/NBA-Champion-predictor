import pandas as pd
import numpy as np
from datetime import datetime
import random
from src.nba_data_fetcher import get_comprehensive_team_data
from src.models.predict import NBAChampionPredictor, analyze_matchup
import warnings
warnings.filterwarnings('ignore')

class NBAPlayoffSimulator:
    """
    Simulador completo dos playoffs da NBA
    """
    
    def __init__(self):
        self.predictor = NBAChampionPredictor()
        self.teams_data = None
        self.playoff_bracket = None
        self.simulation_results = {}
        
    def load_teams_data(self, season='2024-25'):
        """
        Carregar dados das equipas
        """
        print("📊 A carregar dados das equipas...")
        self.teams_data = get_comprehensive_team_data(season)
        
        if self.teams_data.empty:
            print("❌ Não foi possível carregar dados das equipas")
            return False
            
        # Treinar modelo se necessário
        if not self.predictor.load_model():
            print("🤖 A treinar novo modelo...")
            self.predictor.train_model(self.teams_data)
            
        return True
    
    def create_playoff_bracket(self, conference_teams=8):
        """
        Criar bracket dos playoffs baseado na classificação atual
        """
        print("🏀 A criar bracket dos playoffs...")
        
        if self.teams_data is None:
            print("❌ Dados das equipas não carregados")
            return None
        
        # Separar por conferência (simplificado - baseado em nomes conhecidos)
        eastern_teams = [
            'Boston Celtics', 'Miami Heat', 'Philadelphia 76ers', 'New York Knicks',
            'Cleveland Cavaliers', 'Indiana Pacers', 'Orlando Magic', 'Brooklyn Nets',
            'Atlanta Hawks', 'Chicago Bulls', 'Toronto Raptors', 'Washington Wizards',
            'Charlotte Hornets', 'Detroit Pistons'
        ]
        
        # Filtrar equipas por conferência
        east_df = self.teams_data[self.teams_data['TEAM_NAME'].isin(eastern_teams)].copy()
        west_df = self.teams_data[~self.teams_data['TEAM_NAME'].isin(eastern_teams)].copy()
        
        # Ordenar por win percentage
        east_df = east_df.sort_values('W_PCT', ascending=False).head(conference_teams)
        west_df = west_df.sort_values('W_PCT', ascending=False).head(conference_teams)
        
        # Criar matchups da primeira ronda
        east_matchups = [
            (east_df.iloc[0]['TEAM_NAME'], east_df.iloc[7]['TEAM_NAME']),
            (east_df.iloc[1]['TEAM_NAME'], east_df.iloc[6]['TEAM_NAME']),
            (east_df.iloc[2]['TEAM_NAME'], east_df.iloc[5]['TEAM_NAME']),
            (east_df.iloc[3]['TEAM_NAME'], east_df.iloc[4]['TEAM_NAME'])
        ]
        
        west_matchups = [
            (west_df.iloc[0]['TEAM_NAME'], west_df.iloc[7]['TEAM_NAME']),
            (west_df.iloc[1]['TEAM_NAME'], west_df.iloc[6]['TEAM_NAME']),
            (west_df.iloc[2]['TEAM_NAME'], west_df.iloc[5]['TEAM_NAME']),
            (west_df.iloc[3]['TEAM_NAME'], west_df.iloc[4]['TEAM_NAME'])
        ]
        
        self.playoff_bracket = {
            'first_round': {
                'eastern': east_matchups,
                'western': west_matchups
            },
            'east_teams': east_df,
            'west_teams': west_df
        }
        
        print(f"✅ Bracket criado com {len(east_df)} equipas do Este e {len(west_df)} do Oeste")
        return self.playoff_bracket
    
    def simulate_series(self, team1_name, team2_name, series_format=7, home_advantage=0.1):
        """
        Simular uma série entre duas equipas
        """
        try:
            # Obter dados das equipas
            team1_data = self.teams_data[self.teams_data['TEAM_NAME'] == team1_name].iloc[0]
            team2_data = self.teams_data[self.teams_data['TEAM_NAME'] == team2_name].iloc[0]
            
            # Calcular probabilidades base
            team1_prob = self.calculate_team_strength(team1_data)
            team2_prob = self.calculate_team_strength(team2_data)
            
            # Normalizar probabilidades
            total_prob = team1_prob + team2_prob
            team1_win_prob = team1_prob / total_prob
            team2_win_prob = team2_prob / total_prob
            
            # Simular série
            team1_wins = 0
            team2_wins = 0
            games_needed = (series_format + 1) // 2  # Jogos necessários para ganhar
            
            game_count = 0
            while team1_wins < games_needed and team2_wins < games_needed:
                game_count += 1
                
                # Aplicar home advantage (alternado)
                current_team1_prob = team1_win_prob
                if game_count <= 2 or game_count in [5, 7]:  # Team1 em casa
                    current_team1_prob += home_advantage
                elif game_count in [3, 4, 6]:  # Team2 em casa
                    current_team1_prob -= home_advantage
                
                # Simular jogo
                if random.random() < current_team1_prob:
                    team1_wins += 1
                else:
                    team2_wins += 1
            
            # Determinar vencedor
            winner = team1_name if team1_wins > team2_wins else team2_name
            series_score = f"{team1_wins}-{team2_wins}"
            
            return {
                'winner': winner,
                'loser': team2_name if winner == team1_name else team1_name,
                'series_score': series_score,
                'games_played': game_count,
                'team1_win_prob': team1_win_prob,
                'team2_win_prob': team2_win_prob
            }
            
        except Exception as e:
            print(f"❌ Erro na simulação {team1_name} vs {team2_name}: {e}")
            return None
    
    def calculate_team_strength(self, team_data):
        """
        Calcular força da equipa baseada em múltiplas métricas
        """
        strength = 0
        
        # Win percentage (40%)
        strength += team_data.get('W_PCT', 0.5) * 0.4
        
        # Net rating (30%)
        net_rating = team_data.get('NET_RATING', 0)
        strength += (net_rating / 20) * 0.3  # Normalizado
        
        # Plus/minus (20%)
        plus_minus = team_data.get('PLUS_MINUS', 0)
        strength += (plus_minus / 500) * 0.2  # Normalizado
        
        # Clutch performance (10%)
        clutch_pct = team_data.get('CLUTCH_W_PCT', team_data.get('W_PCT', 0.5))
        strength += clutch_pct * 0.1
        
        return max(0.1, strength)  # Mínimo de 0.1
    
    def simulate_round(self, matchups, round_name):
        """
        Simular uma ronda completa
        """
        print(f"\n🏀 A simular {round_name}...")
        print("-" * 50)
        
        winners = []
        round_results = []
        
        for team1, team2 in matchups:
            result = self.simulate_series(team1, team2)
            if result:
                winner = result['winner']
                winners.append(winner)
                round_results.append(result)
                
                # Mostrar resultado
                print(f"{team1:<25} vs {team2:<25} → {winner} ({result['series_score']})")
                
                # Análise rápida
                if result['team1_win_prob'] > 0.6:
                    print(f"   📊 {team1} era favorito ({result['team1_win_prob']:.1%})")
                elif result['team2_win_prob'] > 0.6:
                    print(f"   📊 {team2} era favorito ({result['team2_win_prob']:.1%})")
                else:
                    print(f"   📊 Série equilibrada")
        
        self.simulation_results[round_name] = round_results
        return winners
    
    def simulate_full_playoffs(self, num_simulations=1000):
        """
        Simular playoffs completos múltiplas vezes
        """
        print(f"🎯 A simular playoffs {num_simulations} vezes...")
        
        champions = {}
        conference_winners = {'eastern': {}, 'western': {}}
        
        for sim in range(num_simulations):
            if sim % 100 == 0:
                print(f"Simulação {sim + 1}/{num_simulations}")
            
            # Resetar dados
            self.simulation_results = {}
            
            # Primeira ronda
            east_r1_winners = self.simulate_round(
                self.playoff_bracket['first_round']['eastern'],
                'Eastern Conference First Round'
            )
            west_r1_winners = self.simulate_round(
                self.playoff_bracket['first_round']['western'],
                'Western Conference First Round'
            )
            
            # Semifinais de conferência
            east_sf_matchups = [(east_r1_winners[0], east_r1_winners[3]), 
                               (east_r1_winners[1], east_r1_winners[2])]
            west_sf_matchups = [(west_r1_winners[0], west_r1_winners[3]), 
                               (west_r1_winners[1], west_r1_winners[2])]
            
            east_sf_winners = self.simulate_round(east_sf_matchups, 'Eastern Conference Semifinals')
            west_sf_winners = self.simulate_round(west_sf_matchups, 'Western Conference Semifinals')
            
            # Finais de conferência
            east_final_result = self.simulate_series(east_sf_winners[0], east_sf_winners[1])
            west_final_result = self.simulate_series(west_sf_winners[0], west_sf_winners[1])
            
            east_champion = east_final_result['winner']
            west_champion = west_final_result['winner']
            
            # Contar vencedores de conferência
            conference_winners['eastern'][east_champion] = conference_winners['eastern'].get(east_champion, 0) + 1
            conference_winners['western'][west_champion] = conference_winners['western'].get(west_champion, 0) + 1
            
            # Finais NBA
            finals_result = self.simulate_series(east_champion, west_champion)
            champion = finals_result['winner']
            
            # Contar campeões
            champions[champion] = champions.get(champion, 0) + 1
        
        return champions, conference_winners
    
    def simulate_single_playoffs(self):
        """
        Simular playoffs uma única vez com detalhes
        """
        print("\n🏆 SIMULAÇÃO DETALHADA DOS PLAYOFFS NBA")
        print("=" * 60)
        
        if not self.playoff_bracket:
            print("❌ Bracket não criado")
            return
        
        # Mostrar bracket inicial
        self.display_bracket()
        
        # Primeira ronda
        print("\n" + "🔥" * 20 + " PRIMEIRA RONDA " + "🔥" * 20)
        east_r1_winners = self.simulate_round(
            self.playoff_bracket['first_round']['eastern'],
            'Eastern Conference First Round'
        )
        west_r1_winners = self.simulate_round(
            self.playoff_bracket['first_round']['western'],
            'Western Conference First Round'
        )
        
        # Semifinais de conferência
        print("\n" + "⚡" * 20 + " SEMIFINAIS " + "⚡" * 20)
        east_sf_matchups = [(east_r1_winners[0], east_r1_winners[3]), 
                           (east_r1_winners[1], east_r1_winners[2])]
        west_sf_matchups = [(west_r1_winners[0], west_r1_winners[3]), 
                           (west_r1_winners[1], west_r1_winners[2])]
        
        east_sf_winners = self.simulate_round(east_sf_matchups, 'Eastern Conference Semifinals')
        west_sf_winners = self.simulate_round(west_sf_matchups, 'Western Conference Semifinals')
        
        # Finais de conferência
        print("\n" + "🏆" * 20 + " FINAIS DE CONFERÊNCIA " + "🏆" * 20)
        east_final_result = self.simulate_series(east_sf_winners[0], east_sf_winners[1])
        west_final_result = self.simulate_series(west_sf_winners[0], west_sf_winners[1])
        
        east_champion = east_final_result['winner']
        west_champion = west_final_result['winner']
        
        print(f"\n🏆 CAMPEÃO DO ESTE: {east_champion}")
        print(f"🏆 CAMPEÃO DO OESTE: {west_champion}")
        
        # Análise dos finalistas
        print(f"\n⚔️  ANÁLISE DOS FINALISTAS:")
        analyze_matchup(east_champion, west_champion, self.teams_data)
        
        # Finais NBA
        print("\n" + "👑" * 25 + " FINAIS NBA " + "👑" * 25)
        finals_result = self.simulate_series(east_champion, west_champion)
        
        print(f"\n🏆 CAMPEÃO NBA: {finals_result['winner']}")
        print(f"📊 Série: {finals_result['series_score']}")
        print(f"🎮 Jogos disputados: {finals_result['games_played']}")
        
        return finals_result['winner']
    
    def display_bracket(self):
        """
        Mostrar bracket inicial
        """
        print("\n📋 BRACKET DOS PLAYOFFS:")
        print("=" * 50)
        
        print("\n🏀 CONFERÊNCIA ESTE:")
        for i, (team1, team2) in enumerate(self.playoff_bracket['first_round']['eastern'], 1):
            print(f"  {i}. {team1} vs {team2}")
        
        print("\n🏀 CONFERÊNCIA OESTE:")
        for i, (team1, team2) in enumerate(self.playoff_bracket['first_round']['western'], 1):
            print(f"  {i}. {team1} vs {team2}")
    
    def run_championship_odds(self, simulations=1000):
        """
        Executar análise completa de probabilidades
        """
        print(f"\n🎲 ANÁLISE DE PROBABILIDADES ({simulations} simulações)")
        print("=" * 60)
        
        champions, conference_winners = self.simulate_full_playoffs(simulations)
        
        # Mostrar probabilidades de campeonato
        print("\n🏆 PROBABILIDADES DE CAMPEONATO:")
        print("-" * 50)
        
        sorted_champions = sorted(champions.items(), key=lambda x: x[1], reverse=True)
        for i, (team, wins) in enumerate(sorted_champions, 1):
            percentage = (wins / simulations) * 100
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
            print(f"{emoji} {team:<25} {percentage:6.1f}% ({wins} vitórias)")
        
        # Mostrar probabilidades de conferência
        print("\n🏀 PROBABILIDADES DE VENCER A CONFERÊNCIA:")
        print("-" * 50)
        
        print("ESTE:")
        east_sorted = sorted(conference_winners['eastern'].items(), key=lambda x: x[1], reverse=True)
        for team, wins in east_sorted:
            percentage = (wins / simulations) * 100
            print(f"  {team:<25} {percentage:6.1f}%")
        
        print("\nOESTE:")
        west_sorted = sorted(conference_winners['western'].items(), key=lambda x: x[1], reverse=True)
        for team, wins in west_sorted:
            percentage = (wins / simulations) * 100
            print(f"  {team:<25} {percentage:6.1f}%")
        
        # Salvar resultados
        self.save_simulation_results(sorted_champions, east_sorted, west_sorted, simulations)
        
        return sorted_champions[0][0]  # Retornar favorito
    
    def save_simulation_results(self, champions, east_winners, west_winners, simulations):
        """
        Salvar resultados da simulação
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"playoff_simulation_{timestamp}.csv"
        
        # Preparar dados para CSV
        results = []
        for team, wins in champions:
            results.append({
                'team': team,
                'championship_probability': (wins / simulations) * 100,
                'championship_wins': wins,
                'conference': 'Eastern' if team in dict(east_winners) else 'Western',
                'conference_wins': dict(east_winners).get(team, 0) + dict(west_winners).get(team, 0)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\n💾 Resultados salvos em {filename}")

def main():
    """
    Função principal
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Playoffs Simulator")
    parser.add_argument("mode", choices=["single", "odds", "bracket"], 
                       help="Modo: single (simulação única), odds (probabilidades), bracket (mostrar bracket)")
    parser.add_argument("--simulations", type=int, default=1000,
                       help="Número de simulações para modo odds")
    
    args = parser.parse_args()
    
    # Criar simulador
    simulator = NBAPlayoffSimulator()
    
    # Carregar dados
    if not simulator.load_teams_data():
        print("❌ Erro ao carregar dados")
        return
    
    # Criar bracket
    if not simulator.create_playoff_bracket():
        print("❌ Erro ao criar bracket")
        return
    
    # Executar modo selecionado
    if args.mode == "single":
        champion = simulator.simulate_single_playoffs()
        print(f"\n🎉 Campeão da simulação: {champion}")
        
    elif args.mode == "odds":
        favorite = simulator.run_championship_odds(args.simulations)
        print(f"\n🏆 Favorito ao título: {favorite}")
        
    elif args.mode == "bracket":
        simulator.display_bracket()

if __name__ == "__main__":
    main()