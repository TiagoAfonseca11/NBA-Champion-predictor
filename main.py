import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from src.nba_data_fetcher import get_comprehensive_team_data
from src.models.predict import train_and_predict_champion
import warnings
warnings.filterwarnings('ignore')

def calculate_championship_probability(df):
    """
    Calculate championship probability based on multiple factors
    """
    print("🧮 A calcular probabilidades de campeonato...")
    
    # Normalizar dados (0-1 scale)
    def normalize_column(col):
        if col.std() == 0:
            return col
        return (col - col.min()) / (col.max() - col.min())
    
    # Features positivas (quanto maior, melhor)
    positive_features = ['W_PCT', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'STL', 'BLK', 
                        'PLUS_MINUS', 'OFF_RATING', 'NET_RATING', 'AST_TO_RATIO', 
                        'DEFENSIVE_EFFICIENCY', 'SCORING_EFFICIENCY']
    
    # Features negativas (quanto menor, melhor)
    negative_features = ['TOV', 'PF', 'DEF_RATING', 'OPP_FG_PCT']
    
    # Inicializar score
    df['CHAMPIONSHIP_SCORE'] = 0
    
    # Calcular score baseado em features positivas
    for feature in positive_features:
        if feature in df.columns:
            normalized = normalize_column(df[feature])
            weight = get_feature_weight(feature)
            df['CHAMPIONSHIP_SCORE'] += normalized * weight
    
    # Subtrair score baseado em features negativas
    for feature in negative_features:
        if feature in df.columns:
            normalized = normalize_column(df[feature])
            weight = get_feature_weight(feature)
            df['CHAMPIONSHIP_SCORE'] -= normalized * weight
    
    # Adicionar bônus histórico
    if 'HIST_W_PCT' in df.columns:
        hist_bonus = normalize_column(df['HIST_W_PCT']) * 0.3
        df['CHAMPIONSHIP_SCORE'] += hist_bonus
    
    # Adicionar bônus clutch
    if 'CLUTCH_W_PCT' in df.columns:
        clutch_bonus = normalize_column(df['CLUTCH_W_PCT']) * 0.4
        df['CHAMPIONSHIP_SCORE'] += clutch_bonus
    
    # Converter para probabilidade (0-100%)
    df['CHAMPIONSHIP_PROBABILITY'] = (
        (df['CHAMPIONSHIP_SCORE'] - df['CHAMPIONSHIP_SCORE'].min()) /
        (df['CHAMPIONSHIP_SCORE'].max() - df['CHAMPIONSHIP_SCORE'].min()) * 100
    )
    
    return df

def get_feature_weight(feature):
    """
    Return weight for each feature based on importance for championship prediction
    """
    weights = {
        'W_PCT': 5.0,           # Win percentage - most important
        'NET_RATING': 4.0,      # Net rating - very important
        'PLUS_MINUS': 3.5,      # Plus/minus - very important
        'DEF_RATING': 3.0,      # Defense wins championships
        'OFF_RATING': 2.5,      # Offensive efficiency
        'FG_PCT': 2.0,          # Field goal percentage
        'FG3_PCT': 1.8,         # Three-point shooting
        'AST_TO_RATIO': 1.5,    # Ball handling
        'CLUTCH_W_PCT': 4.0,    # Clutch performance
        'HIST_W_PCT': 1.0,      # Historical performance
        'DEFENSIVE_EFFICIENCY': 2.0,
        'SCORING_EFFICIENCY': 1.8,
        'TOV': 2.0,             # Turnovers (negative)
        'OPP_FG_PCT': 1.5,      # Opponent field goal % (negative)
    }
    
    return weights.get(feature, 1.0)

def analyze_team_strengths(df):
    """
    Analyze and categorize team strengths
    """
    print("💪 A analisar pontos fortes das equipas...")
    
    results = []
    
    for _, team in df.iterrows():
        team_name = team['TEAM_NAME']
        
        # Categorizar pontos fortes
        strengths = []
        
        if team.get('OFF_RATING', 0) > df['OFF_RATING'].quantile(0.8):
            strengths.append("Ataque Explosivo")
        
        if team.get('DEF_RATING', 120) < df['DEF_RATING'].quantile(0.2):
            strengths.append("Defesa Sólida")
        
        if team.get('FG3_PCT', 0) > df['FG3_PCT'].quantile(0.8):
            strengths.append("Especialista em 3pt")
        
        if team.get('AST_TO_RATIO', 0) > df['AST_TO_RATIO'].quantile(0.8):
            strengths.append("Controlo de Bola")
        
        if team.get('CLUTCH_W_PCT', 0) > df.get('CLUTCH_W_PCT', pd.Series([0])).quantile(0.8):
            strengths.append("Clutch")
        
        if not strengths:
            strengths = ["Equilibrada"]
        
        results.append({
            'TEAM_NAME': team_name,
            'STRENGTHS': ', '.join(strengths),
            'CHAMPIONSHIP_PROBABILITY': team['CHAMPIONSHIP_PROBABILITY']
        })
    
    return pd.DataFrame(results)

def predict_champion():
    """
    Main function to predict NBA champion
    """
    print("🏀 NBA CHAMPION PREDICTOR 2025-26")
    print("=" * 50)
    
    # Obter dados completos
    df = get_comprehensive_team_data()
    
    if df.empty:
        print("❌ Não foi possível obter dados da NBA")
        return
    
    # Calcular probabilidades
    df = calculate_championship_probability(df)
    
    # Ordenar por probabilidade
    df_sorted = df.sort_values('CHAMPIONSHIP_PROBABILITY', ascending=False)
    
    # Top 10 candidatos
    print("\n🏆 TOP 10 CANDIDATOS AO TÍTULO:")
    print("-" * 50)
    
    for i, (_, team) in enumerate(df_sorted.head(10).iterrows(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}."
        print(f"{emoji} {team['TEAM_NAME']:<25} {team['CHAMPIONSHIP_PROBABILITY']:6.1f}%")
    
    # Análise detalhada do favorito
    top_team = df_sorted.iloc[0]
    print(f"\n🎯 ANÁLISE DETALHADA - {top_team['TEAM_NAME']}")
    print("-" * 50)
    print(f"📊 Probabilidade de Campeonato: {top_team['CHAMPIONSHIP_PROBABILITY']:.1f}%")
    print(f"🏆 Record: {top_team['W']}-{top_team['L']} ({top_team['W_PCT']:.1%})")
    print(f"⚡ Rating Ofensivo: {top_team.get('OFF_RATING', 'N/A')}")
    print(f"🛡️  Rating Defensivo: {top_team.get('DEF_RATING', 'N/A')}")
    print(f"📈 Plus/Minus: {top_team['PLUS_MINUS']:+.1f}")
    
    # Análise de pontos fortes
    strengths_df = analyze_team_strengths(df_sorted.head(10))
    print(f"\n💪 PONTOS FORTES DAS TOP 5:")
    print("-" * 50)
    for _, team in strengths_df.head(5).iterrows():
        print(f"{team['TEAM_NAME']:<25} → {team['STRENGTHS']}")
    
    # Alertas e insights
    print(f"\n⚠️  INSIGHTS E ALERTAS:")
    print("-" * 50)
    
    # Equipa com melhor ataque
    best_offense = df_sorted.loc[df_sorted['OFF_RATING'].idxmax()]
    print(f"🔥 Melhor Ataque: {best_offense['TEAM_NAME']} ({best_offense['OFF_RATING']:.1f})")
    
    # Equipa com melhor defesa
    best_defense = df_sorted.loc[df_sorted['DEF_RATING'].idxmin()]
    print(f"🛡️  Melhor Defesa: {best_defense['TEAM_NAME']} ({best_defense['DEF_RATING']:.1f})")
    
    # Equipa mais clutch
    if 'CLUTCH_W_PCT' in df_sorted.columns:
        most_clutch = df_sorted.loc[df_sorted['CLUTCH_W_PCT'].idxmax()]
        print(f"⏰ Mais Clutch: {most_clutch['TEAM_NAME']} ({most_clutch['CLUTCH_W_PCT']:.1%})")
    
    # Salvar resultados
    save_predictions(df_sorted)
    
    print(f"\n✅ Análise completa! Dados salvos em 'predictions_{datetime.now().strftime('%Y%m%d')}.csv'")

def save_predictions(df):
    """
    Save predictions to CSV file
    """
    filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # Selecionar colunas para salvar
    save_columns = ['TEAM_NAME', 'W', 'L', 'W_PCT', 'CHAMPIONSHIP_PROBABILITY', 'CHAMPIONSHIP_SCORE']
    available_columns = [col for col in save_columns if col in df.columns]
    
    df[available_columns].to_csv(filename, index=False)
    print(f"💾 Predições salvas em {filename}")

def show_standings():
    """
    Show current NBA standings
    """
    print("📊 CLASSIFICAÇÃO ATUAL NBA 2024-25")
    print("=" * 50)
    
    df = get_comprehensive_team_data()
    if df.empty:
        print("❌ Não foi possível obter dados")
        return
    
    # Mostrar classificação por conferência (simplificado)
    standings_cols = ['TEAM_NAME', 'W', 'L', 'W_PCT', 'PTS', 'PLUS_MINUS']
    available_cols = [col for col in standings_cols if col in df.columns]
    
    standings = df[available_cols].sort_values('W_PCT', ascending=False)
    
    print("\n🏀 TOP 15 EQUIPAS:")
    print("-" * 60)
    print(f"{'EQUIPA':<25} {'W':<3} {'L':<3} {'W%':<6} {'PTS':<5} {'+/-':<6}")
    print("-" * 60)
    
    for _, team in standings.head(15).iterrows():
        print(f"{team['TEAM_NAME']:<25} {team['W']:<3} {team['L']:<3} "
              f"{team['W_PCT']:<6.1%} {team['PTS']:<5.0f} {team['PLUS_MINUS']:+6.0f}")

def main():
    parser = argparse.ArgumentParser(description="NBA Champion Predictor 2025-26")
    parser.add_argument("mode", choices=["predict", "standings"], 
                       help="Modo de operação: predict (previsão) ou standings (classificação)")
    
    args = parser.parse_args()
    
    if args.mode == "predict":
        predict_champion()
    elif args.mode == "standings":
        show_standings()

if __name__ == "__main__":
    main()