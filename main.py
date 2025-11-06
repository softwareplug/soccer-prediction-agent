from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os

app = FastAPI(title="Soccer Prediction API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://surepredict.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Soccer Prediction API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "soccer_predictor"}

# Only run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
