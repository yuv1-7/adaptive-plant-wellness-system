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
    General filtering that works for ALL cases:
    1. If species matches disease -> show filtered results
    2. If species doesn't match BUT confidence is high -> show with warning
    3. If both have low confidence -> show both unfiltered with disclaimer
    """
    if not species_results or len(species_results) == 0:
        # No species identified, return original disease results
        disease_result['filtered'] = False
        disease_result['filter_message'] = "No species identified - showing raw disease detection"
        return disease_result
    
    if not disease_result.get('success') or not disease_result.get('predictions'):
        return disease_result
    
    # Get top identified species
    top_species = species_results[0]
    species_confidence = top_species.score
    scientific_name = top_species.scientific_name.lower()
    common_names = [name.lower() for name in top_species.common_names]
    genus = top_species.genus.lower()
    
    # Extract genus from scientific name
    identified_genus = scientific_name.split()[0] if scientific_name else ""
    
    # Get top disease prediction
    top_disease = disease_result['predictions'][0]
    disease_confidence = top_disease['confidence']
    disease_plant = top_disease['plant'].lower().strip()
    
    # Build comprehensive search terms
    search_terms = set()
    search_terms.add(genus)
    search_terms.add(identified_genus)
    search_terms.update(common_names)
    
    # Expanded plant mappings (covers more cases)
    plant_mappings = {
        'solanum': ['tomato', 'potato', 'solanum', 'lycopersicon', 'eggplant', 'aubergine'],
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
        'prunus': ['cherry', 'peach', 'plum', 'prunus'],
        'cherry': ['prunus', 'cherry'],
        'peach': ['prunus', 'peach'],
        'capsicum': ['pepper', 'bell pepper', 'capsicum', 'chili', 'chilli'],
        'pepper': ['capsicum', 'pepper'],
        'bell': ['capsicum', 'pepper'],
        'rosa': ['rose', 'rosa'],
        'rose': ['rosa', 'rose'],
        'citrus': ['orange', 'lemon', 'lime', 'citrus'],
        'orange': ['citrus', 'orange'],
        'fragaria': ['strawberry', 'fragaria'],
        'strawberry': ['fragaria', 'strawberry'],
        'rubus': ['raspberry', 'blackberry', 'rubus'],
        'raspberry': ['rubus', 'raspberry'],
        'vaccinium': ['blueberry', 'vaccinium'],
        'blueberry': ['vaccinium', 'blueberry'],
        'cucurbita': ['squash', 'pumpkin', 'cucurbita', 'zucchini'],
        'squash': ['cucurbita', 'squash'],
        'glycine': ['soybean', 'soy', 'glycine', 'soya'],
        'soybean': ['glycine', 'soybean', 'soy'],
    }
    
    # Add mapped terms
    for term in list(search_terms):
        if term in plant_mappings:
            search_terms.update(plant_mappings[term])
    
    # Check if disease plant matches identified species
    def plants_match(disease_plant_name, search_terms):
        for term in search_terms:
            if term and disease_plant_name and (term in disease_plant_name or disease_plant_name in term):
                return True
        return False
    
    match_found = plants_match(disease_plant, search_terms)
    
    # CASE 1: Species and disease match - Filter and show only matching
    if match_found:
        matched_predictions = []
        for prediction in disease_result['predictions']:
            pred_plant = prediction['plant'].lower().strip()
            if plants_match(pred_plant, search_terms):
                matched_predictions.append(prediction)
        
        if matched_predictions:
            disease_result['predictions'] = matched_predictions[:5]  # Top 5
            disease_result['top_prediction'] = matched_predictions[0]
            disease_result['filtered'] = True
            disease_result['filter_message'] = f"✓ Results filtered for {top_species.scientific_name}"
            return disease_result
    
    # CASE 2: High confidence mismatch - Show disease results with strong warning
    if species_confidence > 60 and disease_confidence > 50:
        disease_result['filtered'] = False
        disease_result['filter_message'] = (
            f"⚠️ MISMATCH DETECTED: Species identified as '{top_species.scientific_name}' "
            f"({species_confidence:.1f}% confidence), but disease model detected '{top_disease['plant']}' "
            f"({disease_confidence:.1f}% confidence). This may indicate:\n"
            f"• Disease model limitation\n"
            f"• Poor image quality\n"
            f"Showing unfiltered disease results below."
        )
        return disease_result
    
    # CASE 3: Low confidence on both - Show everything with disclaimer
    if species_confidence < 60 or disease_confidence < 50:
        disease_result['filtered'] = False
        disease_result['filter_message'] = (
            f"⚠️ LOW CONFIDENCE: Species ID: {species_confidence:.1f}%, "
            f"Disease detection: {disease_confidence:.1f}%. "
            f"Results may be unreliable. Consider:\n"
            f"• Taking a clearer photo\n"
            f"• Better lighting conditions\n"
            f"• Closer view of leaves\n"
            f"Showing all results for manual review."
        )
        return disease_result
    
    # CASE 4: Mismatch with low disease confidence - Likely healthy or unsupported
    if not match_found and disease_confidence < 50:
        disease_result['filtered'] = False
        disease_result['filter_message'] = (
            f"ℹ️ Species identified as '{top_species.scientific_name}' ({species_confidence:.1f}%). "
            f"Disease detection has low confidence ({disease_confidence:.1f}%). "
            f"Plant may be healthy or disease model doesn't support this species."
        )
        return disease_result
    
    # CASE 5: Default fallback - show unfiltered with generic message
    disease_result['filtered'] = False
    disease_result['filter_message'] = (
        f"ℹ️ Showing unfiltered results. Species: {top_species.scientific_name} "
        f"({species_confidence:.1f}%), Disease detection: {disease_confidence:.1f}%"
    )
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
        print (result)
        
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