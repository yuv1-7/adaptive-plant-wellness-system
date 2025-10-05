from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

from services.plant_identifier import PlantIdentifierService
from models.response_models import PlantIdentificationResponse, ErrorResponse

app = FastAPI(
    title="Plant Species Identifier API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize service
plant_service = PlantIdentifierService()


@app.get("/")
async def root():
    """Root endpoint - serves the frontend"""
    from fastapi.responses import FileResponse
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "Plant Species Identifier API", "docs": "/docs"}


@app.post("/api/identify", response_model=PlantIdentificationResponse)
async def identify_plant(file: UploadFile = File(...)):
    """
    Identify plant species from uploaded image
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Plant identification results with species information
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        image_data = await file.read()
        
        # Identify plant
        result = await plant_service.identify_plant(image_data, file.filename)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "plant-identifier"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
