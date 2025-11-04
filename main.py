from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(title="Soccer Prediction API")

# Configure CORS to allow requests from your Netlify domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://soccer-prediction-agent-deployment.netlify.app",
        "http://localhost:8000",  # For local testing
        "http://localhost:3000",  # For local frontend development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Soccer Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
