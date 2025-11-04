#!/usr/bin/env python3

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.models.prediction_model import SoccerPredictionModel

def main():
    print("Testing Soccer Prediction Model...")
    
    # Create model instance
    model = SoccerPredictionModel()
    print("Model created")
    
    # Check if model file exists
    if os.path.exists('model.pkl'):
        print("Pre-trained model found")
    else:
        print("No pre-trained model found (normal if not trained yet)")
    
    # Try to train if data exists
    if os.path.exists('matches.csv'):
        print("Training data found - training model...")
        model.train('matches.csv')
    else:
        print("No training data found at 'matches.csv'")
    
    # Try a prediction
    try:
        result = model.predict("Team A", "Team B")
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
