import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

class SoccerPredictionModel:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.team_stats = {}
        self.load_model()
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model file {self.model_path} not found. Train model first.")
            self.model = None
    
    def calculate_team_stats(self, data):
        """Simplified version that works with your data"""
        stats = {}
        all_teams = set(data['home_team'].unique()) | set(data['away_team'].unique())
        
        for team in all_teams:
            home_games = data[data['home_team'] == team]
            away_games = data[data['away_team'] == team]
            
            # Use only the columns you actually have
            home_goals = home_games['home_goals'].mean() if not home_games.empty else 1.0
            away_goals = away_games['away_goals'].mean() if not away_games.empty else 1.0
            
            # If form columns don't exist, use default values
            home_form = 0.5  # Default value
            away_form = 0.5  # Default value
            
            stats[team] = {
                'attack_strength': home_goals,
                'defense_strength': away_goals,
                'form': 0.5  # Default form
            }
        return stats
    
    def train(self, data_path='sample_data.csv'):
        try:
            data = pd.read_csv(data_path)
            print(f"📊 Data loaded: {data.shape}")
            
            self.team_stats = self.calculate_team_stats(data)
            print(f"📈 Stats calculated for {len(self.team_stats)} teams")
            
            # Simple feature engineering based on available data
            features = []
            for idx, row in data.iterrows():
                home_stats = self.team_stats[row['home_team']]
                away_stats = self.team_stats[row['away_team']]
                
                feature_vector = [
                    home_stats['attack_strength'],  # Home attack
                    away_stats['defense_strength'], # Away defense
                    home_stats['form'],             # Home form (default)
                    away_stats['form']              # Away form (default)
                ]
                features.append(feature_vector)
            
            X = pd.DataFrame(features)
            y = data['result']
            
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            joblib.dump(self.model, self.model_path)
            print(f"✅ Model trained and saved")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise
    
    def predict(self, home_team, away_team):
        if self.model is None:
            raise ValueError("Model not loaded. Train model first.")
        
        try:
            home_stats = self.team_stats.get(home_team, {'attack_strength': 1.0, 'defense_strength': 1.0, 'form': 0.5})
            away_stats = self.team_stats.get(away_team, {'attack_strength': 1.0, 'defense_strength': 1.0, 'form': 0.5})
            
            features = [
                home_stats['attack_strength'],
                away_stats['defense_strength'],
                home_stats['form'],
                away_stats['form']
            ]
            
            prediction_proba = self.model.predict_proba([features])[0]
            
            return {
                'home_win': prediction_proba[0],
                'draw': prediction_proba[1],
                'away_win': prediction_proba[2],
                'confidence': max(prediction_proba)
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

def predict_match(home_team, away_team):
    model = SoccerPredictionModel()
    return model.predict(home_team, away_team)
