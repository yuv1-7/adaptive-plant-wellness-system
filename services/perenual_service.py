import httpx
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()


class PerenualService:
    """Service for fetching plant care information from Perenual API"""
    
    def __init__(self):
        self.api_key = os.getenv("perenual_api_key")
        self.base_url = "https://perenual.com/api/v2"
        
    async def search_plant_by_name(self, plant_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a plant by name using Perenual API
        
        Args:
            plant_name: Scientific or common name of the plant
            
        Returns:
            Plant details including care information
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for plant
                search_url = f"{self.base_url}/species-list"
                params = {
                    'key': self.api_key,
                    'q': plant_name,
                    'page': 1
                }
                
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('data') or len(data['data']) == 0:
                    print(f"No results found for: {plant_name}")
                    return None
                
                # Get the first matching plant's ID
                plant_id = data['data'][0]['id']
                print(f"Found plant ID {plant_id} for: {plant_name}")
                
                # Fetch detailed information
                detail_url = f"{self.base_url}/species/details/{plant_id}"
                detail_params = {'key': self.api_key}
                
                detail_response = await client.get(detail_url, params=detail_params)
                detail_response.raise_for_status()
                plant_details = detail_response.json()
                
                return self._parse_plant_details(plant_details)
                
        except httpx.HTTPStatusError as e:
            print(f"Perenual API HTTP error: {e}")
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching plant details: {e}")
            return None
    
    def _parse_plant_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure plant details from Perenual API response"""
        
        # Helper function to safely get values
        def safe_get(obj, key, default='Unknown'):
            value = obj.get(key, default)
            # Handle list values - get first item if list
            if isinstance(value, list) and len(value) > 0:
                return value[0]
            return value if value else default
        
        return {
            'id': data.get('id'),
            'common_name': safe_get(data, 'common_name', 'Unknown'),
            'scientific_name': safe_get(data, 'scientific_name', ['Unknown']),
            'family': safe_get(data, 'family'),
            'origin': data.get('origin', []),
            'type': safe_get(data, 'type'),
            'dimension': data.get('dimension'),
            'cycle': safe_get(data, 'cycle', 'Unknown'),
            'watering': safe_get(data, 'watering', 'Average'),
            'watering_period': data.get('watering_period'),
            'watering_general_benchmark': {
                'value': data.get('watering_general_benchmark', {}).get('value'),
                'unit': data.get('watering_general_benchmark', {}).get('unit')
            },
            'sunlight': data.get('sunlight', ['Full sun']),
            'pruning_month': data.get('pruning_month', []),
            'pruning_count': data.get('pruning_count', {}),
            'maintenance': safe_get(data, 'maintenance', 'Moderate'),
            'care_level': safe_get(data, 'care_level', 'Moderate'),
            'growth_rate': safe_get(data, 'growth_rate', 'Moderate'),
            'soil': data.get('soil', []),
            'pest_susceptibility': data.get('pest_susceptibility', []),
            'flowers': data.get('flowers', False),
            'flowering_season': data.get('flowering_season'),
            'flower_color': data.get('flower_color'),
            'hardiness': {
                'min': data.get('hardiness', {}).get('min') if isinstance(data.get('hardiness'), dict) else None,
                'max': data.get('hardiness', {}).get('max') if isinstance(data.get('hardiness'), dict) else None
            },
            'propagation': data.get('propagation', []),
            'leaf': data.get('leaf', False),
            'fruits': data.get('fruits', False),
            'edible_fruit': data.get('edible_fruit', False),
            'description': data.get('description'),
            'default_image': data.get('default_image', {}).get('original_url') if data.get('default_image') else None
        }
    
    async def get_plant_care_guide(self, plant_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive care guide for a plant
        
        Args:
            plant_name: Scientific or common name
            
        Returns:
            Structured care guide information
        """
        plant_details = await self.search_plant_by_name(plant_name)
        
        if not plant_details:
            return None
        
        # Structure care guide
        care_guide = {
            'plant_info': {
                'name': plant_details['common_name'],
                'scientific_name': plant_details['scientific_name'],
                'family': plant_details['family'],
                'type': plant_details['type'],
                'care_level': plant_details['care_level'],
                'growth_rate': plant_details['growth_rate']
            },
            'watering': {
                'frequency': plant_details['watering'],
                'period': plant_details['watering_period'],
                'benchmark': plant_details['watering_general_benchmark']
            },
            'light': {
                'requirements': plant_details['sunlight']
            },
            'soil': {
                'types': plant_details['soil']
            },
            'maintenance': {
                'level': plant_details['maintenance'],
                'pruning_months': plant_details['pruning_month'],
                'pruning_count': plant_details['pruning_count']
            },
            'environment': {
                'cycle': plant_details['cycle'],
                'hardiness_zone': plant_details['hardiness']
            },
            'additional_info': {
                'flowering_season': plant_details['flowering_season'],
                'flower_color': plant_details['flower_color'],
                'propagation': plant_details['propagation'],
                'pest_susceptibility': plant_details['pest_susceptibility'],
                'edible_fruit': plant_details['edible_fruit']
            },
            'description': plant_details['description'],
            'image': plant_details['default_image']
        }
        
        return care_guide