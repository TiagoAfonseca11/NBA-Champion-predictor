import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedNBAModelTrainer:
    """
    Advanced NBA model trainer with historical data and feature engineering
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_performance = {}
        
    def load_historical_data(self):
        """
        Load historical NBA data for training
        """
        print("📚 A carregar dados históricos...")
        
        # Dados históricos simulados baseados em estatísticas reais
        # Em produção, estes dados viriam de uma base de dados ou API
        historical_seasons = [
            # 2023-24 Season (exemplo)
            {
                'season': '2023-24',
                'champion': 'Boston Celtics',
                'finalist': 'Dallas Mavericks',
                'conference_champions': ['Boston Celtics', 'Dallas Mavericks'],
                'playoff_teams': [
                    'Boston Celtics', 'New York Knicks', 'Milwaukee Bucks', 'Cleveland Cavaliers',
                    'Orlando Magic', 'Indiana Pacers', 'Philadelphia 76ers', 'Miami Heat',
                    'Oklahoma City Thunder', 'Denver Nuggets', 'Minnesota Timberwolves', 
                    'LA Clippers', 'Dallas Mavericks', 'Phoenix Suns', 'New Orleans Pelicans',
                    'Los Angeles Lakers'
                ]
            },
            # 2022-23 Season
            {
                'season': '2022-23',
                'champion': 'Denver Nuggets',
                'finalist': 'Miami Heat',
                'conference_champions': ['Denver Nuggets', 'Miami Heat'],
                'playoff_teams': [
                    'Milwaukee Bucks', 'Boston Celtics', 'Philadelphia 76ers', 'Cleveland Cavaliers',
                    'New York Knicks', 'Brooklyn Nets', 'Miami Heat', 'Atlanta Hawks',
                    'Denver Nuggets', 'Memphis Grizzlies', 'Sacramento Kings', 'Phoenix Suns',
                    'LA Clippers', 'Golden State Warriors', 'Los Angeles Lakers', 'Minnesota Timberwolves'
                ]
            },
            # 2021-22 Season
            {
                'season': '2021-22',
                'champion': 'Golden State Warriors',
                'finalist': 'Boston Celtics',
                'conference_champions': ['Golden State Warriors', 'Boston Celtics'],
                'playoff_teams': [
                    'Miami Heat', 'Boston Celtics', 'Milwaukee Bucks', 'Philadelphia 76ers',
                    'Toronto Raptors', 'Chicago Bulls', 'Brooklyn Nets', 'Atlanta Hawks',
                    'Phoenix Suns', 'Memphis Grizzlies', 'Golden State Warriors', 'Dallas Mavericks',
                    'Utah Jazz', 'Denver Nuggets', 'Minnesota Timberwolves', 'LA Clippers'
                ]
            }
        ]
        
        return historical_seasons
    
    def generate_training_data(self, current_season_data):
        """
        Generate training data combining historical and current season data
        """
        print("🔧 A gerar dados de treino...")
        
        # Simular dados históricos baseados em padrões conhecidos
        training_data = []
        
        # Características de equipas campeãs históricas
        champion_profiles = [
            {'W_PCT': 0.732, 'NET_RATING': 8.5, 'OFF_RATING': 117.2, 'DEF_RATING': 108.7, 'championship_outcome': 1.0},
            {'W_PCT': 0.695, 'NET_RATING': 6.8, 'OFF_RATING': 115.8, 'DEF_RATING': 109.0, 'championship_outcome': 1.0},
            {'W_PCT': 0.744, 'NET_RATING': 7.9, 'OFF_RATING': 116.5, 'DEF_RATING': 108.6, 'championship_outcome': 1.0},
            {'W_PCT': 0.707, 'NET_RATING': 6.2, 'OFF_RATING': 114.9, 'DEF_RATING': 108.7, 'championship_outcome': 1.0},
            {'W_PCT': 0.768, 'NET_RATING': 9.1, 'OFF_RATING': 118.3, 'DEF_RATING': 109.2, 'championship_outcome': 1.0},
        ]
        
        # Características de finalistas
        finalist_profiles = [
            {'W_PCT': 0.659, 'NET_RATING': 4.2, 'OFF_RATING': 113.8, 'DEF_RATING': 109.6, 'championship_outcome': 0.8},
            {'W_PCT': 0.671, 'NET_RATING': 5.1, 'OFF_RATING': 114.2, 'DEF_RATING': 109.1, 'championship_outcome': 0.8},
            {'W_PCT': 0.634, 'NET_RATING': 3.8, 'OFF_RATING': 112.9, 'DEF_RATING': 109.1, 'championship_outcome': 0.8},
            {'W_PCT': 0.646, 'NET_RATING': 4.5, 'OFF_RATING': 113.5, 'DEF_RATING': 109.0, 'championship_outcome': 0.8},
        ]
        
        # Características de equipas dos playoffs
        playoff_profiles = [
            {'W_PCT': 0.610, 'NET_RATING': 2.8, 'OFF_RATING': 112.5, 'DEF_RATING': 109.7, 'championship_outcome': 0.4},
            {'W_PCT': 0.585, 'NET_RATING': 1.9, 'OFF_RATING': 111.8, 'DEF_RATING': 109.9, 'championship_outcome': 0.4},
            {'W_PCT': 0.598, 'NET_RATING': 2.2, 'OFF_RATING': 112.1, 'DEF_RATING': 109.9, 'championship_outcome': 0.4},
            {'W_PCT': 0.573, 'NET_RATING': 1.5, 'OFF_RATING': 111.2, 'DEF_RATING': 109.7, 'championship_outcome': 0.4},
        ]
        
        # Características de equipas normais
        regular_profiles = [
            {'W_PCT': 0.500, 'NET_RATING': 0.2, 'OFF_RATING': 110.8, 'DEF_RATING': 110.6, 'championship_outcome': 0.1},
            {'W_PCT': 0.463, 'NET_RATING': -1.8, 'OFF_RATING': 109.5, 'DEF_RATING': 111.3, 'championship_outcome': 0.1},
            {'W_PCT': 0.439, 'NET_RATING': -2.5, 'OFF_RATING': 108.9, 'DEF_RATING': 111.4, 'championship_outcome': 0.1},
            {'W_PCT': 0.415, 'NET_RATING': -3.2, 'OFF_RATING': 108.2, 'DEF_RATING': 111.4, 'championship_outcome': 0.1},
        ]
        
        # Combinar todos os perfis
        all_profiles = champion_profiles + finalist_profiles + playoff_profiles + regular_profiles
        
        # Gerar dados com variação
        for profile in all_profiles:
            for _ in range(10):  # Múltiplas amostras por perfil
                sample = {}
                for key, value in profile.items():
                    if key != 'championship_outcome':
                        # Adicionar variação aleatória
                        noise = np.random.normal(0, 0.02)  # 2% de variação
                        sample[key] = value + noise
                    else:
                        sample[key] = value
                
                # Adicionar features adicionais
                sample['FG_PCT'] = np.random.normal(0.46, 0.02)
                sample['FG3_PCT'] = np.random.normal(0.36, 0.02)
                sample['AST'] = np.random.normal(2200, 100)
                sample['TOV'] = np.random.normal(1150, 50)
                sample['PLUS_MINUS'] = sample['NET_RATING'] * 50 + np.random.normal(0, 20)
                
                training_data.append(sample)
        
        return pd.DataFrame(training_data)
    
    def engineer_features(self, df):
        """
        Create advanced features for better prediction
        """
        print("⚙️ A criar features avançadas...")
        
        df = df.copy()
        
        # Efficiency metrics
        if 'OFF_RATING' in df.columns and 'DEF_RATING' in df.columns:
            df['NET_EFFICIENCY'] = df['OFF_RATING'] - df['DEF_RATING']
            df['TOTAL_EFFICIENCY'] = df['OFF_RATING'] + df['DEF_RATING']
        
        # Balanced team metric
        if 'AST' in df.columns and 'TOV' in df.columns:
            df['BALL_CONTROL'] = df['AST'] / (df['TOV'] + 1)
        
        # Shooting efficiency
        if 'FG_PCT' in df.columns and 'FG3_PCT' in df.columns:
            df['SHOOTING_BALANCE'] = df['FG_PCT'] + df['FG3_PCT'] * 0.5
        
        # Win momentum
        if 'W_PCT' in df.columns:
            df['WIN_MOMENTUM'] = df['W_PCT'] ** 2  # Quadratic to emphasize higher win rates
        
        # Clutch factor (if available)
        if 'CLUTCH_W_PCT' in df.columns and 'W_PCT' in df.columns:
            df['CLUTCH_FACTOR'] = df['CLUTCH_W_PCT'] / (df['W_PCT'] + 0.01)
        
        # Historical consistency (if available)
        if 'HIST_W_PCT' in df.columns and 'W_PCT' in df.columns:
            df['CONSISTENCY'] = 1 - abs(df['W_PCT'] - df['HIST_W_PCT'])
        
        # Playoff readiness score
        playoff_features = ['NET_RATING', 'W_PCT', 'BALL_CONTROL']
        available_playoff_features = [f for f in playoff_features if f in df.columns]
        
        if available_playoff_features:
            # Normalizar features
            for feature in available_playoff_features:
                df[f'{feature}_NORM'] = (df[feature] - df[feature].mean()) / df[feature].std()
            
            # Criar score combinado
            df['PLAYOFF_READINESS'] = df[[f'{f}_NORM' for f in available_playoff_features]].mean(axis=1)
        
        return df
    
    def train_championship_model(self, df):
        """
        Train the championship prediction model
        """
        print("🎯 A treinar modelo de previsão de campeão...")
        
        # Gerar dados de treino
        training_df = self.generate_training_data(df)
        
        # Engineer features
        training_df = self.engineer_features(training_df)
        
        # Preparar features
        feature_columns = [
            'W_PCT', 'NET_RATING', 'OFF_RATING', 'DEF_RATING',
            'FG_PCT', 'FG3_PCT', 'AST', 'TOV', 'PLUS_MINUS',
            'NET_EFFICIENCY', 'BALL_CONTROL', 'SHOOTING_BALANCE',
            'WIN_MOMENTUM', 'PLAYOFF_READINESS'
        ]
        
        # Filtrar colunas existentes
        available_features = [col for col in feature_columns if col in training_df.columns]
        self.feature_names = available_features
        
        X = training_df[available_features]
        y = training_df['championship_outcome']
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar múltiplos modelos
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )
        }
        
        # Treinar e avaliar modelos
        for name, model in models.items():
            print(f"🔄 A treinar {name}...")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Avaliar performance
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            
            self.model_performance[name] = {
                'MSE': mse,
                'R2': r2,
                'MAE': mae,
                'CV_R2': cv_scores.mean(),
                'CV_STD': cv_scores.std()
            }
            
            print(f"   MSE: {mse:.4f}")
            print(f"   R²: {r2:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            self.models[name] = model
        
        # Criar ensemble
        self.create_ensemble_model(X_test_scaled, y_test)
        
        return self.models
    
    def create_ensemble_model(self, X_test, y_test):
        """
        Create ensemble model combining multiple algorithms
        """
        print("🤝 A criar modelo ensemble...")
        
        # Previsões dos modelos individuais
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test)
        
        # Ensemble com pesos baseados na performance
        weights = {
            'RandomForest': 0.6,
            'GradientBoosting': 0.4
        }
        
        ensemble_pred = sum(weights[name] * pred for name, pred in predictions.items())
        
        # Avaliar ensemble
        mse = mean_squared_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        self.model_performance['Ensemble'] = {
            'MSE': mse,
            'R2': r2,
            'MAE': mae,
            'weights': weights
        }
        
        print(f"📊 Ensemble Performance:")
        print(f"   MSE: {mse:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   MAE: {mae:.4f}")
        
        # Salvar ensemble como modelo principal
        self.ensemble_weights = weights
    
    def save_models(self):
        """
        Save all trained models
        """
        print("💾 A salvar modelos...")
        
        # Criar diretório
        models_dir = 'src/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Salvar dados do modelo
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'ensemble_weights': self.ensemble_weights,
            'performance': self.model_performance,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Salvar modelo principal
        joblib.dump(model_data, f'{models_dir}/nba_champion_model.pkl')
        
        # Salvar scaler separadamente
        joblib.dump(self.scaler, f'{models_dir}/nba_scaler.pkl')
        
        print(f"✅ Modelos salvos em {models_dir}/")
    
    def print_model_summary(self):
        """
        Print summary of model performance
        """
        print("\n📈 RESUMO DA PERFORMANCE DOS MODELOS")
        print("=" * 50)
        
        for name, metrics in self.model_performance.items():
            print(f"\n{name}:")
            print(f"  R² Score: {metrics['R2']:.4f}")
            print(f"  MSE: {metrics['MSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            
            if 'CV_R2' in metrics:
                print(f"  CV R²: {metrics['CV_R2']:.4f} (±{metrics['CV_STD']:.4f})")

def train_full_model():
    """
    Train complete NBA championship prediction model
    """
    print("🚀 TREINO COMPLETO DO MODELO NBA")
    print("=" * 50)
    
    # Importar dados atuais
    try:
        from src.nba_data_fetcher import get_comprehensive_team_data
        current_data = get_comprehensive_team_data()
        
        if current_data.empty:
            print("❌ Não foi possível obter dados atuais")
            return
        
    except ImportError:
        print("⚠️  Usando dados simulados para treino")
        # Criar dados simulados se não conseguir importar
        current_data = pd.DataFrame({
            'TEAM_NAME': ['Lakers', 'Celtics', 'Warriors', 'Bucks'],
            'W_PCT': [0.650, 0.720, 0.680, 0.690],
            'NET_RATING': [4.2, 6.8, 5.1, 5.9],
            'OFF_RATING': [115.2, 117.8, 116.1, 116.9],
            'DEF_RATING': [111.0, 111.0, 111.0, 111.0],
            'FG_PCT': [0.47, 0.48, 0.46, 0.47],
            'FG3_PCT': [0.36, 0.38, 0.39, 0.35],
            'AST': [2200, 2100, 2300, 2000],
            'TOV': [1100, 1150, 1200, 1080],
            'PLUS_MINUS': [200, 340, 250, 290]
        })
    
    # Inicializar trainer
    trainer = AdvancedNBAModelTrainer()
    
    # Treinar modelo
    trainer.train_championship_model(current_data)
    
    # Salvar modelos
    trainer.save_models()
    
    # Mostrar resumo
    trainer.print_model_summary()
    
    print("\n🎉 Treino completo! Modelo pronto para previsões.")

if __name__ == '__main__':
    train_full_model()