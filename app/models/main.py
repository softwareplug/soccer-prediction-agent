from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Soccer Prediction API")

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Soccer Prediction API is running"}

if __name__ == "__main__":
    import uvicorn
    # Fixed: Pass the app as import string instead of object
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
