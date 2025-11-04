from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.prediction_model import predict_match, SoccerPredictionModel

router = APIRouter()

# Define request model for prediction
class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

@router.post("/predict")
async def api_predict(request: PredictionRequest):
    try:
        prediction = predict_match(request.home_team, request.away_team)
        return {
            "success": True,
            "prediction": prediction,
            "match": f"{request.home_team} vs {request.away_team}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train")
async def train_model():
    try:
        model = SoccerPredictionModel()
        model.train('sample_data.csv')
        return {
            "success": True, 
            "message": "Model trained successfully",
            "teams_analyzed": len(model.team_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "soccer_predictor"}

@router.get("/teams")
async def get_teams():
    try:
        model = SoccerPredictionModel()
        if hasattr(model, 'team_stats') and model.team_stats:
            return {"success": True, "teams": list(model.team_stats.keys())}
        else:
            return {"success": True, "message": "No teams loaded. Train model first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
