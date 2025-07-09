import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

class NBAChampionPredictor:
    """
    Advanced NBA Champion Prediction Model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'nba_champion_model.pkl')
        self.scaler_path = os.path.join(os.path.dirname(__file__), 'nba_scaler.pkl')
        
    def prepare_features(self, df):
        """
        Prepare features for machine learning model
        """
        # Features selecionadas baseadas na importância para campeonatos
        feature_columns = [
            'W_PCT', 'NET_RATING', 'OFF_RATING', 'DEF_RATING', 'PACE',
            'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'TOV', 'STL', 'BLK',
            'PLUS_MINUS', 'AST_TO_RATIO', 'DEFENSIVE_EFFICIENCY', 'SCORING_EFFICIENCY'
        ]
        
        # Adicionar features históricas se disponíveis
        if 'HIST_W_PCT' in df.columns:
            feature_columns.append('HIST_W_PCT')
        if 'HIST_PLUS_MINUS' in df.columns:
            feature_columns.append('HIST_PLUS_MINUS')
        
        # Adicionar features clutch se disponíveis
        if 'CLUTCH_W_PCT' in df.columns:
            feature_columns.append('CLUTCH_W_PCT')
        if 'CLUTCH_PLUS_MINUS' in df.columns:
            feature_columns.append('CLUTCH_PLUS_MINUS')
        
        # Filtrar apenas colunas existentes
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Extrair features
        X = df[available_features].copy()
        
        # Preencher valores missing
        X = X.fillna(X.mean())
        
        # Criar features adicionais
        if 'OFF_RATING' in X.columns and 'DEF_RATING' in X.columns:
            X['RATING_DIFFERENTIAL'] = X['OFF_RATING'] - X['DEF_RATING']
        
        if 'W_PCT' in X.columns and 'HIST_W_PCT' in X.columns:
            X['CONSISTENCY'] = abs(X['W_PCT'] - X['HIST_W_PCT'])
        
        return X, available_features
    
    def create_championship_target(self, df):
        """
        Create target variable for championship prediction
        Baseado em performance histórica e métricas avançadas
        """
        # Criar target baseado em múltiplos fatores
        target = (
            df['W_PCT'] * 0.4 +
            (df['NET_RATING'] / 20) * 0.3 +  # Normalizado
            (df['PLUS_MINUS'] / 500) * 0.2 +  # Normalizado
            df.get('CLUTCH_W_PCT', df['W_PCT']) * 0.1
        )
        
        # Normalizar para 0-1
        target = (target - target.min()) / (target.max() - target.min())
        
        return target
    
    def train_model(self, df):
        """
        Train the championship prediction model
        """
        print("🤖 A treinar modelo de previsão...")
        
        # Preparar features
        X, feature_names = self.prepare_features(df)
        
        # Criar target
        y = self.create_championship_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
        # Treinar modelos
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Criar ensemble
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Weighted ensemble (RF: 60%, GB: 40%)
        ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
        
        # Avaliar modelo
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        print(f"📊 Performance do Modelo:")
        print(f"   MSE: {mse:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Salvar modelo principal (Random Forest para feature importance)
        self.model = rf_model
        self.gb_model = gb_model
        self.feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
        
        # Salvar modelos
        self.save_model()
        
        return self.model
    
    def predict_championship_probability(self, df):
        """
        Predict championship probability for all teams
        """
        if self.model is None:
            self.load_model()
        
        # Preparar features
        X, _ = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Fazer previsões com ensemble
        rf_pred = self.model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = 0.6 * rf_pred + 0.4 * gb_pred
        
        # Converter para probabilidade (0-100%)
        probabilities = (ensemble_pred / ensemble_pred.max()) * 100
        
        return probabilities
    
    def get_feature_importance(self):
        """
        Get feature importance from the model
        """
        if self.feature_importance is None:
            return {}
        
        # Ordenar por importância
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance)
    
    def save_model(self):
        """
        Save trained model and scaler
        """
        # Criar diretório se não existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Salvar modelo e GB model
        model_data = {
            'rf_model': self.model,
            'gb_model': self.gb_model,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        print(f"💾 Modelo salvo em {self.model_path}")
    
    def load_model(self):
        """
        Load saved model and scaler
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['rf_model']
                self.gb_model = model_data['gb_model']
                self.feature_importance = model_data['feature_importance']
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Modelo carregado com sucesso!")
                return True
            else:
                print("⚠️  Modelo não encontrado. Será necessário treinar novo modelo.")
                return False
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False

def predict_game_outcome(home_stats, away_stats, model_path=None):
    """
    Predict outcome of a single game
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'nba_champion_model.pkl')
    
    try:
        # Carregar modelo
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            model = model_data['rf_model']
        else:
            print("❌ Modelo não encontrado!")
            return "Modelo não disponível"
        
        # Preparar dados do jogo
        home_features = np.array([
            home_stats.get('points_per_game', 100),
            home_stats.get('rebounds_per_game', 40),
            home_stats.get('assists_per_game', 20),
            home_stats.get('turnovers_per_game', 15),
            home_stats.get('efficiency', 100),
            home_stats.get('fg_pct', 0.45),
            home_stats.get('fg3_pct', 0.35)
        ]).reshape(1, -1)
        
        away_features = np.array([
            away_stats.get('points_per_game', 100),
            away_stats.get('rebounds_per_game', 40),
            away_stats.get('assists_per_game', 20),
            away_stats.get('turnovers_per_game', 15),
            away_stats.get('efficiency', 100),
            away_stats.get('fg_pct', 0.45),
            away_stats.get('fg3_pct', 0.35)
        ]).reshape(1, -1)
        
        # Calcular diferencial
        differential = home_features - away_features
        
        # Fazer previsão (simplificada)
        home_score = np.sum(differential * [0.3, 0.2, 0.2, -0.1, 0.15, 0.1, 0.05])
        
        if home_score > 0:
            confidence = min(abs(home_score) * 10, 30)
            return f"Home team wins (confiança: {confidence:.1f}%)"
        else:
            confidence = min(abs(home_score) * 10, 30)
            return f"Away team wins (confiança: {confidence:.1f}%)"
            
    except Exception as e:
        print(f"❌ Erro na previsão: {e}")
        return "Erro na previsão"

def train_and_predict_champion(df):
    """
    Train model and predict champion
    """
    predictor = NBAChampionPredictor()
    
    # Treinar modelo
    predictor.train_model(df)
    
    # Fazer previsões
    probabilities = predictor.predict_championship_probability(df)
    
    # Adicionar probabilidades ao DataFrame
    df['ML_CHAMPIONSHIP_PROBABILITY'] = probabilities
    
    # Mostrar feature importance
    importance = predictor.get_feature_importance()
    print("\n🔍 IMPORTÂNCIA DAS FEATURES:")
    print("-" * 40)
    for feature, imp in list(importance.items())[:10]:
        print(f"{feature:<25} {imp:.3f}")
    
    return df

def analyze_matchup(team1_name, team2_name, df):
    """
    Analyze head-to-head matchup between two teams
    """
    try:
        team1_data = df[df['TEAM_NAME'] == team1_name].iloc[0]
        team2_data = df[df['TEAM_NAME'] == team2_name].iloc[0]
        
        print(f"\n⚔️  ANÁLISE: {team1_name} vs {team2_name}")
        print("=" * 50)
        
        # Comparar métricas chave
        comparisons = {
            'Record': (f"{team1_data['W']}-{team1_data['L']}", f"{team2_data['W']}-{team2_data['L']}"),
            'Win %': (f"{team1_data['W_PCT']:.1%}", f"{team2_data['W_PCT']:.1%}"),
            'Points': (f"{team1_data['PTS']:.1f}", f"{team2_data['PTS']:.1f}"),
            'Plus/Minus': (f"{team1_data['PLUS_MINUS']:+.1f}", f"{team2_data['PLUS_MINUS']:+.1f}"),
            'FG%': (f"{team1_data['FG_PCT']:.1%}", f"{team2_data['FG_PCT']:.1%}"),
            '3P%': (f"{team1_data['FG3_PCT']:.1%}", f"{team2_data['FG3_PCT']:.1%}"),
        }
        
        for metric, (val1, val2) in comparisons.items():
            print(f"{metric:<12} {val1:>8} vs {val2:<8}")
        
        # Vantagem geral
        team1_score = team1_data.get('CHAMPIONSHIP_PROBABILITY', 0)
        team2_score = team2_data.get('CHAMPIONSHIP_PROBABILITY', 0)
        
        if team1_score > team2_score:
            advantage = ((team1_score - team2_score) / team2_score) * 100
            print(f"\n🎯 Vantagem: {team1_name} ({advantage:.1f}% superior)")
        else:
            advantage = ((team2_score - team1_score) / team1_score) * 100
            print(f"\n🎯 Vantagem: {team2_name} ({advantage:.1f}% superior)")
            
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

# Função de compatibilidade com código antigo
def predict(home_stats, away_stats):
    """
    Backward compatibility function
    """
    return predict_game_outcome(home_stats, away_stats)

if __name__ == '__main__':
    # Teste do modelo
    print("🧪 Teste do Modelo de Previsão")
    print("=" * 40)
    
    # Exemplo de uso
    home_team = {
        'points_per_game': 115,
        'rebounds_per_game': 45,
        'assists_per_game': 28,
        'turnovers_per_game': 12,
        'efficiency': 115,
        'fg_pct': 0.48,
        'fg3_pct': 0.38
    }
    
    away_team = {
        'points_per_game': 108,
        'rebounds_per_game': 42,
        'assists_per_game': 24,
        'turnovers_per_game': 14,
        'efficiency': 108,
        'fg_pct': 0.45,
        'fg3_pct': 0.35
    }
    
    resultado = predict_game_outcome(home_team, away_team)
    print(f"Resultado: {resultado}")