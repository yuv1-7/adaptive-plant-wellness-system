from pydantic import BaseModel
from typing import List, Optional


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


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
