from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PlantSpecies(BaseModel):
    """Model for plant species information"""
    scientific_name: str
    common_names: List[str]
    family: str
    genus: str
    score: float
    images: List[str] = []


class PlantIdentificationResponse(BaseModel):
    """Response model for plant identification"""
    success: bool
    message: str
    results: List[PlantSpecies]


class DiseasePrediction(BaseModel):
    """Model for disease prediction"""
    plant: str
    disease: str
    confidence: float
    is_healthy: bool


class DiseaseDetectionResponse(BaseModel):
    """Response model for disease detection"""
    success: bool
    predictions: List[DiseasePrediction]
    top_prediction: Optional[DiseasePrediction] = None
    error: Optional[str] = None


class CombinedAnalysisResponse(BaseModel):
    """Combined response for species + disease analysis"""
    success: bool
    species_identification: PlantIdentificationResponse
    disease_detection: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str