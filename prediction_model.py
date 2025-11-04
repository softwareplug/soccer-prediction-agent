import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

class SoccerPredictionModel:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from file"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
        else:
            print(f"⚠️ Model file {self.model_path} not found. Train model first.")
            self.model = None
    
    def train(self, data_path='matches.csv'):
        """Train the model with your data"""
        # We'll add your specific training logic here
            def train(self, data_path='matches.csv'):
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X, y)
            
            # Save the model
            joblib.dump(self.model, self.model_path)
            print(f"Model trained and saved to {self.model_path}")
            
        except Exception as e:
            print(f"Training error: {e}")
            raise

    
    def predict(self, home_team, away_team):
        """Make a prediction for a match"""
        # We'll add your specific prediction logic here
            def predict(self, home_team, away_team):
            prediction = self.model.predict_proba([features])[0]
            
            return {
                'home_win': prediction[0],
                'draw': prediction[1],
                'away_win': prediction[2],
                'confidence': max(prediction)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


# Simple interface function (like your original code)
def predict_match(home_team, away_team):
    """This mimics your original predict function"""
    model = SoccerPredictionModel()
    return model.predict(home_team, away_team)
