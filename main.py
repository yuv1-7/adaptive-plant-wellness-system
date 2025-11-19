from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
from typing import List, Optional

from services.plant_identifier import PlantIdentifierService
from services.disease_detector import PlantDiseaseDetector
from models.response_models import (
    PlantIdentificationResponse, 
    ErrorResponse,
    CombinedAnalysisResponse
)

app = FastAPI(
    title="Plant Species Identifier & Disease Detection API",
    version="2.0.0"
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

# Initialize services
plant_service = PlantIdentifierService()
disease_service = PlantDiseaseDetector()


def extract_plant_name(species_name: str) -> str:
    """Extract primary plant name from scientific name"""
    # Common patterns: "Genus species" -> "Genus"
    return species_name.split()[0].lower() if species_name else ""


def find_matching_common_name(common_names: List[str], disease_plant: str) -> Optional[str]:
    """Check if disease plant matches any common names"""
    disease_plant_lower = disease_plant.lower()
    for name in common_names:
        if disease_plant_lower in name.lower() or name.lower() in disease_plant_lower:
            return name
    return None


def filter_disease_predictions(species_results, disease_result):
    """
    Filter disease predictions to only show those matching identified species.
    If no match, create a "healthy" prediction.
    """
    if not species_results or len(species_results) == 0:
        # No species identified, return original disease results
        return disease_result
    
    if not disease_result.get('success') or not disease_result.get('predictions'):
        return disease_result
    
    # Get top identified species
    top_species = species_results[0]
    scientific_name = top_species.scientific_name.lower()
    common_names = [name.lower() for name in top_species.common_names]
    genus = top_species.genus.lower()
    family = top_species.family.lower()
    
    # Extract genus from scientific name (first word)
    identified_genus = scientific_name.split()[0] if scientific_name else ""
    
    # Build search terms from identified species
    search_terms = set()
    search_terms.add(genus)
    search_terms.add(identified_genus)
    search_terms.update(common_names)
    
    # Add partial matches for genus
    if identified_genus:
        search_terms.add(identified_genus[:4])  # First 4 chars
    
    # Enhanced plant mappings - bidirectional
    plant_mappings = {
        'azadirachta': ['neem', 'azadirachta'],
        'neem': ['azadirachta', 'neem'],
        'solanum': ['tomato', 'potato', 'solanum', 'lycopersicon'],
        'lycopersicon': ['tomato', 'solanum'],
        'tomato': ['solanum', 'lycopersicon', 'tomato'],
        'potato': ['solanum', 'potato'],
        'zea': ['corn', 'maize', 'zea'],
        'corn': ['zea', 'maize', 'corn'],
        'maize': ['zea', 'corn', 'maize'],
        'vitis': ['grape', 'vitis'],
        'grape': ['vitis', 'grape'],
        'malus': ['apple', 'malus'],
        'apple': ['malus', 'apple'],
        'prunus': ['cherry', 'peach', 'prunus'],
        'cherry': ['prunus', 'cherry'],
        'peach': ['prunus', 'peach'],
        'capsicum': ['pepper', 'chilli', 'capsicum', 'bell pepper'],
        'pepper': ['capsicum', 'pepper'],
        'chilli': ['capsicum', 'chilli'],
        'rosa': ['rose', 'rosa'],
        'rose': ['rosa', 'rose'],
        'citrus': ['orange', 'lemon', 'citrus'],
        'orange': ['citrus', 'orange'],
        'fragaria': ['strawberry', 'fragaria'],
        'strawberry': ['fragaria', 'strawberry'],
        'rubus': ['raspberry', 'blackberry', 'rubus'],
        'raspberry': ['rubus', 'raspberry'],
        'vaccinium': ['blueberry', 'vaccinium'],
        'blueberry': ['vaccinium', 'blueberry'],
        'cucurbita': ['squash', 'pumpkin', 'cucurbita'],
        'squash': ['cucurbita', 'squash'],
        'glycine': ['soybean', 'soy', 'glycine'],
        'soybean': ['glycine', 'soybean', 'soy'],
        'ocimum': ['basil', 'tulsi', 'ocimum'],
        'basil': ['ocimum', 'basil'],
        'tulsi': ['ocimum', 'tulsi', 'basil'],
    }
    
    # Add mapped terms
    for term in list(search_terms):
        if term in plant_mappings:
            search_terms.update(plant_mappings[term])
    
    # Filter predictions
    matched_predictions = []
    
    for prediction in disease_result['predictions']:
        disease_plant = prediction['plant'].lower().strip()
        
        # Check if disease plant matches identified species
        matches = False
        
        # Direct term matching
        for term in search_terms:
            if term and disease_plant and (term in disease_plant or disease_plant in term):
                matches = True
                break
        
        # Check if disease plant is in family (loose match)
        if not matches and family:
            if disease_plant in family or family in disease_plant:
                matches = True
        
        if matches:
            matched_predictions.append(prediction)
    
    # Update disease result based on matches
    if matched_predictions:
        # Found matching diseases - show them
        disease_result['predictions'] = matched_predictions
        disease_result['top_prediction'] = matched_predictions[0]
        disease_result['filtered'] = True
        disease_result['filter_message'] = f"Disease analysis for {top_species.scientific_name}"
    else:
        # No matching diseases found - plant is healthy
        display_name = top_species.common_names[0] if top_species.common_names else top_species.scientific_name
        
        healthy_prediction = {
            'plant': display_name,
            'disease': 'Healthy',
            'confidence': 95.0,  # High confidence for "no disease found"
            'is_healthy': True
        }
        
        disease_result['predictions'] = [healthy_prediction]
        disease_result['top_prediction'] = healthy_prediction
        disease_result['filtered'] = True
        disease_result['filter_message'] = f"No diseases detected - Plant appears healthy"
    
    return disease_result


@app.get("/")
async def root():
    """Root endpoint - serves the frontend"""
    html_path = static_path / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "Plant Species Identifier & Disease Detection API", "docs": "/docs"}


@app.post("/api/analyze")
async def analyze_plant(file: UploadFile = File(...)):
    """
    Complete plant analysis: species identification + disease detection
    Disease results are filtered to match identified species.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Combined results with species information and filtered disease detection
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        image_data = await file.read()
        
        # Run both analyses
        species_result = await plant_service.identify_plant(image_data, file.filename)
        disease_result = await disease_service.detect_disease(image_data)
        
        # Filter disease predictions to match identified species
        if species_result.success and species_result.results:
            disease_result = filter_disease_predictions(species_result.results, disease_result)
        
        return {
            "success": True,
            "species_identification": species_result.dict(),
            "disease_detection": disease_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


@app.post("/api/identify", response_model=PlantIdentificationResponse)
async def identify_plant(file: UploadFile = File(...)):
    """
    Identify plant species from uploaded image (species only)
    
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
        result = await plant_service.identify_plant(image_data, file.filename)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    """
    Detect plant diseases from uploaded image (disease only, no filtering)
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Disease detection results (unfiltered)
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        image_data = await file.read()
        result = await disease_service.detect_disease(image_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error detecting disease: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    model_info = disease_service.get_model_info()
    return {
        "status": "healthy", 
        "service": "plant-identifier-disease-detector",
        "disease_model": model_info
    }


@app.get("/api/supported-diseases")
async def get_supported_diseases():
    """Get list of all diseases the model can detect"""
    return {
        "classes": disease_service.get_supported_classes(),
        "count": disease_service.num_classes
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )