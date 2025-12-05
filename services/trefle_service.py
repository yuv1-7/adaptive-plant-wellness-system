import httpx
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime, timedelta

load_dotenv()


class TrefleService:
    """Service for fetching plant care information from Trefle API with caching"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.api_key = os.getenv("trefle_api_key")
        self.base_url = "https://trefle.io/api/v1"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "trefle_cache.json"
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cached plant data"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _get_cache_key(self, plant_name: str) -> str:
        """Generate cache key from plant name"""
        return plant_name.lower().strip().replace(" ", "_")
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid (30 days)"""
        if 'timestamp' not in cache_entry:
            return False
        
        cached_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cached_time < timedelta(days=30)
    
    async def search_plant_by_name(self, plant_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a plant by name using Trefle API with caching
        
        Args:
            plant_name: Scientific or common name of the plant
            
        Returns:
            Plant details including care information
        """
        # Check cache first
        cache_key = self._get_cache_key(plant_name)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"✓ Using cached data for: {plant_name}")
            return self.cache[cache_key]['data']
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for plant
                search_url = f"{self.base_url}/plants/search"
                params = {
                    'token': self.api_key,
                    'q': plant_name
                }
                
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('data') or len(data['data']) == 0:
                    print(f"No results found for: {plant_name}")
                    return None
                
                plant_id = data['data'][0]['id']
                plant_slug = data['data'][0]['slug']
                print(f"Found plant ID {plant_id} ({plant_slug}) for: {plant_name}")
                
                # Fetch detailed information
                detail_url = f"{self.base_url}/plants/{plant_id}"
                detail_params = {'token': self.api_key}
                
                detail_response = await client.get(detail_url, params=detail_params)
                detail_response.raise_for_status()
                plant_details = detail_response.json()
                
                parsed_details = self._parse_plant_details(plant_details['data'])
                
                # Cache the result
                self.cache[cache_key] = {
                    'data': parsed_details,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_cache()
                print(f"✓ Cached data for: {plant_name}")
                
                return parsed_details
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"⚠️ Trefle API rate limit reached!")
                print(f"Response: {e.response.text}")
                # Check cache even if expired as fallback
                if cache_key in self.cache:
                    print(f"⚠️ Using expired cache data for: {plant_name}")
                    return self.cache[cache_key]['data']
            else:
                print(f"Trefle API HTTP error: {e}")
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching plant details: {e}")
            return None
    
    def _parse_plant_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure plant details from Trefle API response"""
        
        # Helper function to safely get values
        def safe_get(obj, key, default='Unknown'):
            value = obj.get(key, default)
            if isinstance(value, list) and len(value) > 0:
                return value[0]
            return value if value else default
        
        # Extract main specifications
        main_specs = data.get('main_species', {})
        specifications = main_specs.get('specifications', {})
        growth = main_specs.get('growth', {})
        
        return {
            'id': data.get('id'),
            'common_name': safe_get(data, 'common_name', 'Unknown'),
            'scientific_name': data.get('scientific_name', 'Unknown'),
            'family': data.get('family', 'Unknown'),
            'genus': data.get('genus', 'Unknown'),
            'origin': main_specs.get('distribution', {}).get('native', []),
            'type': self._determine_plant_type(data),
            'dimension': {
                'height': specifications.get('average_height'),
                'spread': specifications.get('spread')
            },
            'cycle': safe_get(main_specs, 'duration', 'Unknown'),
            'watering': self._map_watering(growth.get('moisture_use')),
            'watering_period': self._estimate_watering_period(growth.get('moisture_use')),
            'watering_general_benchmark': self._get_watering_benchmark(growth.get('moisture_use')),
            'sunlight': self._map_light_requirements(growth.get('light')),
            'pruning_month': self._estimate_pruning_months(main_specs.get('foliage', {})),
            'pruning_count': {},
            'maintenance': self._map_maintenance(growth.get('growth_rate')),
            'care_level': self._estimate_care_level(data),
            'growth_rate': safe_get(growth, 'growth_rate', 'Moderate'),
            'soil': self._map_soil_requirements(growth),
            'pest_susceptibility': [],  # Trefle doesn't provide this
            'flowers': main_specs.get('flower', {}).get('color') is not None,
            'flowering_season': self._get_flowering_season(main_specs.get('flower', {})),
            'flower_color': main_specs.get('flower', {}).get('color'),
            'hardiness': {
                'min': growth.get('minimum_temperature', {}).get('deg_f'),
                'max': growth.get('maximum_temperature', {}).get('deg_f')
            },
            'propagation': main_specs.get('growth', {}).get('propagation', []),
            'leaf': main_specs.get('foliage', {}) is not None,
            'fruits': main_specs.get('fruit_or_seed', {}) is not None,
            'edible_fruit': main_specs.get('edible', False),
            'description': main_specs.get('specifications', {}).get('notes'),
            'default_image': data.get('image_url')
        }
    
    def _determine_plant_type(self, data: Dict) -> str:
        """Determine plant type from Trefle data"""
        main_specs = data.get('main_species', {})
        growth = main_specs.get('growth', {})
        
        if main_specs.get('edible'):
            return 'Vegetable/Herb'
        elif growth.get('growth_habit') == 'tree':
            return 'Tree'
        elif growth.get('growth_habit') == 'shrub':
            return 'Shrub'
        elif main_specs.get('flower'):
            return 'Flowering Plant'
        else:
            return 'Plant'
    
    def _map_watering(self, moisture_use: Optional[int]) -> str:
        """Map Trefle moisture use to watering frequency"""
        if moisture_use is None:
            return 'Average'
        elif moisture_use >= 8:
            return 'Frequent'
        elif moisture_use >= 5:
            return 'Average'
        else:
            return 'Minimum'
    
    def _estimate_watering_period(self, moisture_use: Optional[int]) -> str:
        """Estimate watering period from moisture use"""
        if moisture_use is None:
            return 'Weekly'
        elif moisture_use >= 8:
            return 'Every 1-2 days'
        elif moisture_use >= 5:
            return 'Every 3-4 days'
        else:
            return 'Every 7-14 days'
    
    def _get_watering_benchmark(self, moisture_use: Optional[int]) -> Dict:
        """Get watering benchmark from moisture use"""
        if moisture_use is None:
            return {'value': 7, 'unit': 'days'}
        elif moisture_use >= 8:
            return {'value': 2, 'unit': 'days'}
        elif moisture_use >= 5:
            return {'value': 4, 'unit': 'days'}
        else:
            return {'value': 10, 'unit': 'days'}
    
    def _map_light_requirements(self, light: Optional[int]) -> List[str]:
        """Map Trefle light requirements to sunlight list"""
        if light is None:
            return ['Full sun']
        elif light >= 9:
            return ['Full sun']
        elif light >= 6:
            return ['Full sun', 'Partial shade']
        elif light >= 4:
            return ['Partial shade']
        else:
            return ['Shade']
    
    def _estimate_pruning_months(self, foliage: Dict) -> List[str]:
        """Estimate pruning months based on foliage"""
        texture = foliage.get('texture')
        if texture in ['deciduous']:
            return ['Late winter', 'Early spring']
        else:
            return ['Spring', 'Summer', 'Fall']
    
    def _map_maintenance(self, growth_rate: Optional[str]) -> str:
        """Map growth rate to maintenance level"""
        if growth_rate is None:
            return 'Moderate'
        elif growth_rate.lower() in ['rapid', 'fast']:
            return 'High'
        elif growth_rate.lower() in ['slow']:
            return 'Low'
        else:
            return 'Moderate'
    
    def _estimate_care_level(self, data: Dict) -> str:
        """Estimate care level from various factors"""
        main_specs = data.get('main_species', {})
        growth = main_specs.get('growth', {})
        
        if growth.get('atmospheric_humidity') and growth.get('atmospheric_humidity') >= 7:
            return 'Difficult'
        elif main_specs.get('edible'):
            return 'Moderate'
        else:
            return 'Easy'
    
    def _map_soil_requirements(self, growth: Dict) -> List[str]:
        """Map soil requirements from growth data"""
        soil_types = []
        
        ph = growth.get('soil_ph')
        if ph:
            if ph <= 6:
                soil_types.append('Acidic')
            elif ph >= 8:
                soil_types.append('Alkaline')
            else:
                soil_types.append('Neutral')
        
        soil_types.extend(['Well-drained', 'Loamy'])
        return soil_types
    
    def _get_flowering_season(self, flower: Dict) -> Optional[str]:
        """Get flowering season from flower data"""
        if not flower:
            return None
        
        if flower.get('color'):
            return 'Spring to Summer'
        return None
    
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
        print(care_guide)
        return care_guide
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache = {}
        self._save_cache()
        print("✓ Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(1 for entry in self.cache.values() if self._is_cache_valid(entry))
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'cache_file': str(self.cache_file)
        }