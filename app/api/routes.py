from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "soccer_predictor"}

@router.post("/predict")
async def predict_match(request: PredictionRequest):
    try:
        # Simple dummy prediction - replace with your ML model later
        home_team = request.home_team.lower()
        away_team = request.away_team.lower()
        
        if "arsenal" in home_team or "chelsea" in away_team:
            home_win, draw, away_win = 0.6, 0.25, 0.15
        elif "liverpool" in home_team or "man city" in away_team:
            home_win, draw, away_win = 0.55, 0.25, 0.20
        else:
            home_win, draw, away_win = 0.5, 0.3, 0.2
        
        confidence = max(home_win, draw, away_win)
        
        return {
            "success": True,
            "prediction": {
                "home_win": home_win,
                "draw": draw,
                "away_win": away_win,
                "confidence": confidence
            },
            "match": f"{request.home_team} vs {request.away_team}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
