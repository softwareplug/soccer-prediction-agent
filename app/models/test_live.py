import requests
import json

# Replace with your actual Railway URL
RAILWAY_URL = "https://web-production-bba35.up.railway.app/"  # Check your Railway dashboard
BASE_URL = f"{RAILWAY_URL}/api/v1"

def test_live_api():
    print("🚀 Testing Live API on Railway...")
    
    # Health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Train model
    try:
        response = requests.post(f"{BASE_URL}/train")
        print(f"✅ Train: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Train failed: {e}")
    
    # Make prediction
    try:
        data = {"home_team": "Arsenal", "away_team": "Chelsea"}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"✅ Predict: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Predict failed: {e}")

if __name__ == "__main__":
    test_live_api()
