import httpx
import base64
from typing import Dict, Any
from models.response_models import PlantIdentificationResponse, PlantSpecies
import os
from dotenv import load_dotenv

load_dotenv()

class PlantIdentifierService:
    """Service for identifying plants using PlantNet API"""
    
    def __init__(self):
        self.api_key = os.getenv("plantid_api_key")
        self.base_url = "https://my-api.plantnet.org/v2/identify/all"
        
    async def identify_plant(self, image_data: bytes, filename: str) -> PlantIdentificationResponse:
        """
        Identify plant species from image data
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            PlantIdentificationResponse with identification results
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Prepare multipart form data
                files = {
                    'images': (filename, image_data, 'image/jpeg')
                }
                
                params = {
                    'api-key': self.api_key,
                    'include-related-images': 'false'
                }
                
                # Make request to PlantNet API
                response = await client.post(
                    self.base_url,
                    files=files,
                    params=params
                )
                
                if response.status_code == 404:
                    return PlantIdentificationResponse(
                        success=False,
                        message="No plants identified in the image",
                        results=[]
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # Parse results
                results = self._parse_results(data)
                
                return PlantIdentificationResponse(
                    success=True,
                    message="Plant identified successfully",
                    results=results
                )
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return PlantIdentificationResponse(
                    success=False,
                    message="Invalid API key. Please configure your PlantNet API key.",
                    results=[]
                )
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Identification failed: {str(e)}")
    
    def _parse_results(self, data: Dict[str, Any]) -> list[PlantSpecies]:
        """Parse PlantNet API response into PlantSpecies objects"""
        results = []
        
        if 'results' not in data:
            return results
        
        for result in data['results'][:5]:  # Top 5 results
            species = PlantSpecies(
                scientific_name=result.get('species', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                common_names=result.get('species', {}).get('commonNames', []),
                family=result.get('species', {}).get('family', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                genus=result.get('species', {}).get('genus', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                score=round(result.get('score', 0) * 100, 2),
                images=self._get_reference_images(result)
            )
            results.append(species)
        
        return results
    
    def _get_reference_images(self, result: Dict[str, Any]) -> list[str]:
        """Extract reference image URLs from result"""
        images = []
        if 'images' in result:
            for img in result['images'][:3]:  # Max 3 reference images
                if 'url' in img.get('url', {}):
                    images.append(img['url']['o'])
        return images
