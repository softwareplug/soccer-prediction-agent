import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FootballPredictor:
    def __init__(self, model_path='model.pkl', features_path='features.pkl'):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.label_encoders = {}
        self.team_stats = {}
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature encoders"""
        if os.path.exists(self.model_path) and os.path.exists(self.features_path):
            self.model = joblib.load(self.model_path)
            feature_data = joblib.load(self.features_path)
            self.label_encoders = feature_data['label_encoders']
            self.team_stats = feature_data['team_stats']
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model files not found. Train model first.")
            self.model = None

    def train(self, data_path='matches.csv'):
        """Train the model with advanced feature engineering"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            print(f"📊 Loaded {len(df)} matches")
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Calculate team statistics (rolling averages)
            self._calculate_team_stats(df)
            
            # Create features
            X, y = self._create_features(df)
            
            # Encode categorical features
            X_encoded = self._encode_features(X, training=True)
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            print(f"🎯 Training Accuracy: {train_score:.3f}")
            print(f"🎯 Test Accuracy: {test_score:.3f}")
            
            # Save model and feature encoders
            self._save_model()
            print(f"✅ Model trained and saved successfully!")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise

    def predict(self, home_team, away_team, home_form=None, away_form=None):
        """Make prediction for a match with optional recent form"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            # Create feature vector
            features = self._create_prediction_features(home_team, away_team, home_form, away_form)
            
            # Encode features
            features_encoded = self._encode_features(features, training=False)
            
            # Predict
            prediction = self.model.predict_proba(features_encoded)[0]
            
            # Get predicted class
            predicted_class = self.model.predict(features_encoded)[0]
            
            return {
                'home_win_prob': prediction[0],
                'draw_prob': prediction[1], 
                'away_win_prob': prediction[2],
                'predicted_result': self._get_result_label(predicted_class),
                'confidence': max(prediction),
                'home_team': home_team,
                'away_team': away_team
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise

    def _preprocess_data(self, df):
        """Clean and prepare the data"""
        # Ensure required columns exist
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create result column
        df['result'] = df.apply(lambda row: 
            0 if row['home_score'] > row['away_score'] else  # Home win
            1 if row['home_score'] == row['away_score'] else  # Draw  
            2, axis=1  # Away win
        )
        
        # Sort by date if available
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        
        return df

    def _calculate_team_stats(self, df):
        """Calculate rolling statistics for each team"""
        self.team_stats = {}
        
        for team in set(df['home_team'].unique()) | set(df['away_team'].unique()):
            team_matches = pd.concat([
                df[df['home_team'] == team],
                df[df['away_team'] == team]
            ]).sort_values('date' if 'date' in df.columns else df.index)
            
            # Basic stats
            total_matches = len(team_matches)
            wins = len(team_matches[(
                (team_matches['home_team'] == team) & (team_matches['result'] == 0) |
                (team_matches['away_team'] == team) & (team_matches['result'] == 2)
            )])
            
            # Recent form (last 5 matches)
            recent_matches = team_matches.tail(5)
            recent_wins = len(recent_matches[(
                (recent_matches['home_team'] == team) & (recent_matches['result'] == 0) |
                (recent_matches['away_team'] == team) & (recent_matches['result'] == 2)
            )])
            
            # Goal statistics
            home_goals = team_matches[team_matches['home_team'] == team]['home_score'].sum()
            away_goals = team_matches[team_matches['away_team'] == team]['away_score'].sum()
            total_goals = home_goals + away_goals
            
            self.team_stats[team] = {
                'win_rate': wins / total_matches if total_matches > 0 else 0.33,
                'recent_form': recent_wins / 5 if len(recent_matches) > 0 else 0.33,
                'avg_goals_for': total_goals / total_matches if total_matches > 0 else 1.0,
                'total_matches': total_matches
            }

    def _create_features(self, df):
        """Create advanced features for training"""
        features = []
        targets = []
        
        for _, match in df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get team stats (up to this match)
            home_stats = self.team_stats.get(home_team, {
                'win_rate': 0.33, 'recent_form': 0.33, 'avg_goals_for': 1.0
            })
            away_stats = self.team_stats.get(away_team, {
                'win_rate': 0.33, 'recent_form': 0.33, 'avg_goals_for': 1.0
            })
            
            # Feature vector
            feature_vector = [
                home_stats['win_rate'],
                away_stats['win_rate'],
                home_stats['recent_form'], 
                away_stats['recent_form'],
                home_stats['avg_goals_for'],
                away_stats['avg_goals_for'],
                1 if home_stats['total_matches'] > 10 else 0,  # Home team experience
                1 if away_stats['total_matches'] > 10 else 0   # Away team experience
            ]
            
            features.append(feature_vector)
            targets.append(match['result'])
        
        return pd.DataFrame(features, columns=[
            'home_win_rate', 'away_win_rate', 'home_recent_form', 'away_recent_form',
            'home_avg_goals', 'away_avg_goals', 'home_experienced', 'away_experienced'
        ]), np.array(targets)

    def _create_prediction_features(self, home_team, away_team, home_form=None, away_form=None):
        """Create features for prediction"""
        home_stats = self.team_stats.get(home_team, {
            'win_rate': 0.33, 'recent_form': 0.33, 'avg_goals_for': 1.0, 'total_matches': 0
        })
        away_stats = self.team_stats.get(away_team, {
            'win_rate': 0.33, 'recent_form': 0.33, 'avg_goals_for': 1.0, 'total_matches': 0
        })
        
        # Use provided form or calculated form
        home_recent_form = home_form if home_form is not None else home_stats['recent_form']
        away_recent_form = away_form if away_form is not None else away_stats['recent_form']
        
        features = pd.DataFrame([[
            home_stats['win_rate'],
            away_stats['win_rate'],
            home_recent_form,
            away_recent_form,
            home_stats['avg_goals_for'],
            away_stats['avg_goals_for'],
            1 if home_stats['total_matches'] > 10 else 0,
            1 if away_stats['total_matches'] > 10 else 0
        ]], columns=[
            'home_win_rate', 'away_win_rate', 'home_recent_form', 'away_recent_form',
            'home_avg_goals', 'away_avg_goals', 'home_experienced', 'away_experienced'
        ])
        
        return features

    def _encode_features(self, X, training=False):
        """Encode features - placeholder for more advanced encoding"""
        # No encoding needed for numerical features in this implementation
        return X

    def _get_result_label(self, result_code):
        """Convert result code to label"""
        return {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}.get(result_code, 'Unknown')

    def _save_model(self):
        """Save model and feature encoders"""
        joblib.dump(self.model, self.model_path)
        
        feature_data = {
            'label_encoders': self.label_encoders,
            'team_stats': self.team_stats
        }
        joblib.dump(feature_data, self.features_path)

# Example usage and production-ready features
if __name__ == "__main__":
    # Create predictor
    predictor = FootballPredictor()
    
    # Sample data creation (replace with your actual data)
    def create_sample_data():
        """Create sample match data for testing"""
        matches = []
        teams = ['Team A', 'Team B', 'Team C', 'Team D']
        
        for i in range(100):
            home_team = teams[i % 4]
            away_team = teams[(i + 1) % 4]
            home_score = np.random.randint(0, 4)
            away_score = np.random.randint(0, 4)
            
            matches.append({
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'date': f'2024-{i%12+1:02d}-{i%28+1:02d}'
            })
        
        return pd.DataFrame(matches)

    # Train model
    sample_data = create_sample_data()
    sample_data.to_csv('matches.csv', index=False)
    
    predictor.train('matches.csv')
    
    # Make prediction
    prediction = predictor.predict('Team A', 'Team B')
    print(f"\n🎯 Prediction: {prediction}")
