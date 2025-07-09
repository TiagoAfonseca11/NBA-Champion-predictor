#!/usr/bin/env python3
"""
Script para simulações rápidas dos playoffs NBA
"""

from simulate_playoffs import NBAPlayoffSimulator
import sys
import time

def quick_championship_odds():
    """
    Simulação rápida de probabilidades de campeonato
    """
    print("⚡ SIMULAÇÃO RÁPIDA - PROBABILIDADES DE CAMPEONATO")
    print("=" * 60)
    
    start_time = time.time()
    
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
    
    # Executar simulação com menos iterações para ser mais rápido
    favorite = simulator.run_championship_odds(simulations=500)
    
    end_time = time.time()
    print(f"\n⏱️  Tempo de execução: {end_time - start_time:.1f} segundos")
    print(f"🏆 Grande favorito: {favorite}")

def quick_single_simulation():
    """
    Uma simulação única detalhada
    """
    print("🎯 SIMULAÇÃO ÚNICA DOS PLAYOFFS")
    print("=" * 50)
    
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
    
    # Executar simulação única
    champion = simulator.simulate_single_playoffs()
    
    print(f"\n🎉 RESULTADO: {champion} é o campeão!")
    print("\n📝 Esta é apenas uma simulação - os resultados reais podem variar!")

def show_current_bracket():
    """
    Mostrar bracket atual baseado na classificação
    """
    print("📋 BRACKET ATUAL DOS PLAYOFFS")
    print("=" * 40)
    
    # Criar simulador
    simulator = NBAPlayoffSimulator()
    
    # Carregar dados
    if not simulator.load_teams_data():
        print("❌ Erro ao carregar dados")
        return
    
    # Criar e mostrar bracket
    if simulator.create_playoff_bracket():
        simulator.display_bracket()
        
        # Mostrar top teams
        print("\n🏆 TOP 5 EQUIPAS POR CONFERÊNCIA:")
        print("-" * 40)
        
        print("ESTE:")
        east_teams = simulator.playoff_bracket['east_teams'].head(5)
        for i, (_, team) in enumerate(east_teams.iterrows(), 1):
            print(f"  {i}. {team['TEAM_NAME']:<25} {team['W']}-{team['L']} ({team['W_PCT']:.1%})")
        
        print("\nOESTE:")
        west_teams = simulator.playoff_bracket['west_teams'].head(5)
        for i, (_, team) in enumerate(west_teams.iterrows(), 1):
            print(f"  {i}. {team['TEAM_NAME']:<25} {team['W']}-{team['L']} ({team['W_PCT']:.1%})")

def compare_teams():
    """
    Comparar duas equipas específicas
    """
    print("⚔️  COMPARAÇÃO ENTRE EQUIPAS")
    print("=" * 35)
    
    # Criar simulador
    simulator = NBAPlayoffSimulator()
    
    # Carregar dados
    if not simulator.load_teams_data():
        print("❌ Erro ao carregar dados")
        return
    
    # Mostrar equipas disponíveis
    print("\nEquipas disponíveis:")
    for i, team in enumerate(simulator.teams_data['TEAM_NAME'].sort_values(), 1):
        print(f"  {team}")
    
    # Pedir input do usuário
    print("\n")
    team1 = input("Digite o nome da primeira equipa: ").strip()
    team2 = input("Digite o nome da segunda equipa: ").strip()
    
    # Verificar se as equipas existem
    available_teams = simulator.teams_data['TEAM_NAME'].tolist()
    
    if team1 not in available_teams:
        print(f"❌ Equipa '{team1}' não encontrada")
        return
    
    if team2 not in available_teams:
        print(f"❌ Equipa '{team2}' não encontrada")
        return
    
    # Simular série entre as equipas
    print(f"\n🏀 Simulando série entre {team1} e {team2}...")
    
    # Executar múltiplas simulações
    team1_wins = 0
    team2_wins = 0
    
    for _ in range(100):
        result = simulator.simulate_series(team1, team2)
        if result['winner'] == team1:
            team1_wins += 1
        else:
            team2_wins += 1
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS (100 simulações):")
    print(f"{team1}: {team1_wins} vitórias ({team1_wins}%)")
    print(f"{team2}: {team2_wins} vitórias ({team2_wins}%)")
    
    if team1_wins > team2_wins:
        print(f"\n🏆 {team1} tem vantagem!")
    elif team2_wins > team1_wins:
        print(f"\n🏆 {team2} tem vantagem!")
    else:
        print(f"\n🤝 Equipas muito equilibradas!")

def main():
    """
    Menu principal
    """
    print("🏀 NBA PLAYOFFS SIMULATOR - MENU RÁPIDO")
    print("=" * 45)
    print("1. Simulação única dos playoffs")
    print("2. Probabilidades de campeonato (500 simulações)")
    print("3. Mostrar bracket atual")
    print("4. Comparar duas equipas")
    print("5. Sair")
    
    while True:
        try:
            choice = input("\nEscolha uma opção (1-5): ").strip()
            
            if choice == "1":
                quick_single_simulation()
            elif choice == "2":
                quick_championship_odds()
            elif choice == "3":
                show_current_bracket()
            elif choice == "4":
                compare_teams()
            elif choice == "5":
                print("👋 Até logo!")
                break
            else:
                print("❌ Opção inválida! Escolha entre 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    # Verificar se foi passado argumento de linha de comando
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "odds":
            quick_championship_odds()
        elif mode == "single":
            quick_single_simulation()
        elif mode == "bracket":
            show_current_bracket()
        elif mode == "compare":
            compare_teams()
        else:
            print("❌ Modo inválido! Use: odds, single, bracket, ou compare")
    else:
        main()