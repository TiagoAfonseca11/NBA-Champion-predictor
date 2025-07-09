from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats
from nba_api.stats.endpoints import teamestimatedmetrics, leaguedashteamclutch
#from nba_api.stats.endpoints import leaguedashoppstats, teamdashboardbyyearoveryear
import pandas as pd
import requests
import time

def get_team_stats(season='2024-25'):
    """
    Fetch comprehensive team statistics for NBA season
    """
    print(f"📊 A buscar estatísticas da época regular {season}...")
    
    try:
        # Estatísticas básicas da equipe
        basic_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season'
        )
        df_basic = basic_stats.get_data_frames()[0]
        
        # Estatísticas avançadas
        advanced_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            measure_type_detailed_defense='Advanced'
        )
        df_advanced = advanced_stats.get_data_frames()[0]
        
        # Combinar dados básicos e avançados primeiro
        df = df_basic.merge(df_advanced[['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']], 
                           on='TEAM_ID', how='left')
        
        # Tentar obter estatísticas defensivas (pode falhar)
        try:
            defensive_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Defense'
            )
            df_defensive = defensive_stats.get_data_frames()[0]
            
            # Adicionar apenas colunas defensivas que existem
            defensive_cols = ['TEAM_ID']
            if 'OPP_FG_PCT' in df_defensive.columns:
                defensive_cols.append('OPP_FG_PCT')
            if 'OPP_FG3_PCT' in df_defensive.columns:
                defensive_cols.append('OPP_FG3_PCT')
                
            if len(defensive_cols) > 1:  # Se tivermos mais que apenas TEAM_ID
                df = df.merge(df_defensive[defensive_cols], on='TEAM_ID', how='left')
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível obter estatísticas defensivas completas: {e}")
        
        # Selecionar colunas relevantes
        columns_to_keep = [
            'TEAM_NAME', 'TEAM_ID', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
            'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING',
            'NET_RATING', 'PACE'
        ]
        
        # Adicionar colunas defensivas se existirem
        if 'OPP_FG_PCT' in df.columns:
            columns_to_keep.append('OPP_FG_PCT')
        if 'OPP_FG3_PCT' in df.columns:
            columns_to_keep.append('OPP_FG3_PCT')
        
        # Filtrar apenas as colunas que existem
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df_filtered = df[existing_columns].copy()
        
        # Adicionar métricas calculadas
        df_filtered['AST_TO_RATIO'] = df_filtered['AST'] / df_filtered['TOV'].replace(0, 1)  # Evitar divisão por zero
        df_filtered['DEFENSIVE_EFFICIENCY'] = df_filtered['STL'] + df_filtered['BLK']
        df_filtered['SCORING_EFFICIENCY'] = df_filtered['PTS'] / df_filtered['FGA'].replace(0, 1)
        
        return df_filtered
        
    except Exception as e:
        print(f"❌ Erro ao buscar estatísticas: {e}")
        return pd.DataFrame()
    
def get_injury_report():
    """
    Fetch current injury report (simplified version)
    """
    print("🏥 A buscar relatório de lesões...")
    try:
        # Placeholder para injury data - seria necessário usar uma API específica
        # como ESPN API ou scraping de websites
        injury_data = {
            'team_name': [],
            'injured_players': [],
            'severity_score': []
        }
        return pd.DataFrame(injury_data)
    except Exception as e:
        print(f"❌ Erro ao buscar lesões: {e}")
        return pd.DataFrame()

def get_historical_performance(years=5):
    """
    Fetch historical team performance data
    """
    print(f"📈 A buscar dados históricos dos últimos {years} anos...")
    
    historical_data = []
    current_year = 2024
    
    for year in range(current_year - years, current_year):
        season = f"{year}-{str(year + 1)[2:]}"
        try:
            time.sleep(1)  # Rate limiting
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season'
            )
            df = stats.get_data_frames()[0]
            df['SEASON'] = season
            historical_data.append(df[['TEAM_NAME', 'TEAM_ID', 'W_PCT', 'PTS', 'PLUS_MINUS', 'SEASON']])
        except Exception as e:
            print(f"❌ Erro ao buscar dados de {season}: {e}")
            continue
    
    if historical_data:
        return pd.concat(historical_data, ignore_index=True)
    return pd.DataFrame()

def get_clutch_performance(season='2024-25'):
    """
    Fetch clutch time performance statistics
    """
    print("⏰ A buscar estatísticas de clutch time...")
    
    try:
        clutch_stats = leaguedashteamclutch.LeagueDashTeamClutch(
            season=season,
            season_type_all_star='Regular Season'
        )
        df = clutch_stats.get_data_frames()[0]
        
        # Selecionar colunas relevantes para clutch
        clutch_columns = ['TEAM_NAME', 'TEAM_ID', 'W_PCT', 'FG_PCT', 'FG3_PCT', 'PTS', 'PLUS_MINUS']
        existing_clutch_columns = [col for col in clutch_columns if col in df.columns]
        
        df_clutch = df[existing_clutch_columns].copy()
        df_clutch.columns = [f'CLUTCH_{col}' if col != 'TEAM_NAME' and col != 'TEAM_ID' else col 
                            for col in df_clutch.columns]
        
        return df_clutch
        
    except Exception as e:
        print(f"❌ Erro ao buscar estatísticas de clutch: {e}")
        return pd.DataFrame()

def get_comprehensive_team_data(season='2024-25'):
    """
    Combine all team data sources
    """
    print("🔄 A combinar todos os dados...")
    
    # Obter todos os dados
    basic_stats = get_team_stats(season)
    clutch_stats = get_clutch_performance(season)
    historical_stats = get_historical_performance()
    
    if basic_stats.empty:
        return pd.DataFrame()
    
    # Combinar dados
    combined_df = basic_stats.copy()
    
    # Adicionar clutch stats
    if not clutch_stats.empty:
        combined_df = combined_df.merge(clutch_stats, on=['TEAM_NAME', 'TEAM_ID'], how='left')
    
    # Adicionar médias históricas
    if not historical_stats.empty:
        historical_avg = historical_stats.groupby('TEAM_NAME').agg({
            'W_PCT': 'mean',
            'PTS': 'mean',
            'PLUS_MINUS': 'mean'
        }).reset_index()
        
        historical_avg.columns = ['TEAM_NAME', 'HIST_W_PCT', 'HIST_PTS', 'HIST_PLUS_MINUS']
        combined_df = combined_df.merge(historical_avg, on='TEAM_NAME', how='left')
    
    # Preencher valores NaN
    combined_df = combined_df.fillna(0)
    
    return combined_df

if __name__ == "__main__":
    # Testar função básica
    df = get_team_stats()
    if not df.empty:
        # Mostrar top 10 equipas
        display_cols = ['TEAM_NAME', 'W', 'L', 'W_PCT', 'PTS', 'PLUS_MINUS']
        available_cols = [col for col in display_cols if col in df.columns]
        print("\n🏆 Top 10 Equipas:")
        print(df[available_cols].sort_values('W_PCT', ascending=False).head(10).to_string(index=False))
    
    # Testar função completa
    print("\n" + "="*50)
    comprehensive_df = get_comprehensive_team_data()
    if not comprehensive_df.empty:
        print(f"\n📊 Dados completos obtidos para {len(comprehensive_df)} equipas")
        print(f"🔢 Total de features: {len(comprehensive_df.columns)}")
        print("✅ Dados prontos para análise!")