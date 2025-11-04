#!/usr/bin/env python3
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_api():
    print("🧪 Testing Soccer Prediction API...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test train endpoint
    try:
        response = requests.post(f"{BASE_URL}/train")
        print(f"✅ Train endpoint: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Train endpoint failed: {e}")
    
    # Test predict endpoint
    try:
        data = {"home_team": "Arsenal", "away_team": "Chelsea"}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"✅ Predict endpoint: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Predict endpoint failed: {e}")

if __name__ == "__main__":
    test_api()
