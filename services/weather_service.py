import httpx
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class WeatherService:
    """Service for fetching weather forecast data using Open-Meteo API (100% Free, No API Key)"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.geocoding_url = "https://geocoding-api.open-meteo.com/v1"
    
    async def get_weather_forecast(
        self, 
        latitude: float, 
        longitude: float, 
        days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather forecast for specified location and number of days
        Uses Open-Meteo API - completely free, no API key required
        
        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            days: Number of days (1-16, default: 7)
            
        Returns:
            Structured weather forecast data
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                forecast_url = f"{self.base_url}/forecast"
                params = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,weathercode,windspeed_10m_max',
                    'current_weather': 'true',
                    'timezone': 'auto',
                    'forecast_days': min(days, 16)
                }
                
                response = await client.get(forecast_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Get location name using reverse geocoding
                location_name = await self._get_location_name(client, latitude, longitude)
                
                return self._parse_forecast_data(data, days, location_name)
                
        except httpx.HTTPStatusError as e:
            print(f"Open-Meteo API HTTP error: {e}")
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error fetching weather forecast: {e}")
            return None
    
    async def get_weather_by_city(
        self, 
        city_name: str, 
        days: int = 7
    ) -> Optional[Dict[str, Any]]:
        """
        Get weather forecast by city name
        Uses Open-Meteo Geocoding API - completely free, no API key required
        
        Args:
            city_name: Name of the city
            days: Number of days (1-16, default: 7)
            
        Returns:
            Structured weather forecast data
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                geo_url = f"{self.geocoding_url}/search"
                geo_params = {
                    'name': city_name,
                    'count': 1,
                    'language': 'en',
                    'format': 'json'
                }
                
                geo_response = await client.get(geo_url, params=geo_params)
                geo_response.raise_for_status()
                geo_data = geo_response.json()
                
                if not geo_data.get('results'):
                    print(f"City not found: {city_name}")
                    return None
                
                result = geo_data['results'][0]
                lat = result['latitude']
                lon = result['longitude']
                
                # Get forecast using coordinates
                forecast = await self.get_weather_forecast(lat, lon, days)
                
                if forecast:
                    forecast['location'] = {
                        'city': result.get('name', city_name),
                        'country': result.get('country', 'Unknown'),
                        'admin1': result.get('admin1', ''),
                        'latitude': lat,
                        'longitude': lon
                    }
                
                return forecast
                
        except Exception as e:
            print(f"Error fetching weather by city: {e}")
            return None
    
    async def _get_location_name(self, client: httpx.AsyncClient, lat: float, lon: float) -> Dict[str, str]:
        """Get location name from coordinates using reverse geocoding"""
        try:
            geo_url = f"{self.geocoding_url}/search"
            # Use nearby search with coordinates
            geo_params = {
                'name': f"{lat},{lon}",
                'count': 1,
                'language': 'en',
                'format': 'json'
            }
            
            geo_response = await client.get(geo_url, params=geo_params)
            
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data.get('results'):
                    result = geo_data['results'][0]
                    return {
                        'city': result.get('name', 'Unknown'),
                        'country': result.get('country', 'Unknown'),
                        'admin1': result.get('admin1', '')
                    }
        except:
            pass
        
        return {'city': 'Unknown', 'country': 'Unknown', 'admin1': ''}
    
    def _parse_forecast_data(self, data: Dict[str, Any], days: int, location: Dict[str, str]) -> Dict[str, Any]:
        """Parse Open-Meteo forecast data into structured format"""
        
        daily_data = data.get('daily', {})
        
        # Weather code mapping (WMO Weather interpretation codes)
        weather_codes = {
            0: 'Clear',
            1: 'Mainly Clear',
            2: 'Partly Cloudy',
            3: 'Overcast',
            45: 'Foggy',
            48: 'Foggy',
            51: 'Light Drizzle',
            53: 'Drizzle',
            55: 'Heavy Drizzle',
            61: 'Light Rain',
            63: 'Rain',
            65: 'Heavy Rain',
            71: 'Light Snow',
            73: 'Snow',
            75: 'Heavy Snow',
            77: 'Snow Grains',
            80: 'Light Showers',
            81: 'Showers',
            82: 'Heavy Showers',
            85: 'Light Snow Showers',
            86: 'Snow Showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with Hail',
            99: 'Thunderstorm with Hail'
        }
        
        daily_forecasts = []
        
        for i in range(min(days, len(daily_data.get('time', [])))):
            date_str = daily_data['time'][i]
            temp_max = daily_data['temperature_2m_max'][i]
            temp_min = daily_data['temperature_2m_min'][i]
            temp_avg = round((temp_max + temp_min) / 2, 1)
            
            precipitation = daily_data['precipitation_sum'][i]
            rain = daily_data.get('rain_sum', [0] * len(daily_data['time']))[i]
            weathercode = daily_data['weathercode'][i]
            wind_speed = daily_data['windspeed_10m_max'][i]
            
            # Estimate humidity based on precipitation (rough approximation)
            humidity_estimate = min(40 + (precipitation * 2), 95)
            
            condition = weather_codes.get(weathercode, 'Unknown')
            
            daily_forecasts.append({
                'date': date_str,
                'day_name': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                'temperature': {
                    'min': round(temp_min, 1),
                    'max': round(temp_max, 1),
                    'avg': temp_avg
                },
                'humidity': {
                    'min': round(humidity_estimate - 10, 1) if precipitation > 0 else 30,
                    'max': round(humidity_estimate + 10, 1) if precipitation > 0 else 60,
                    'avg': round(humidity_estimate, 1)
                },
                'condition': condition,
                'rain_total': round(rain, 1),
                'precipitation_total': round(precipitation, 1),
                'wind_speed_avg': round(wind_speed, 1),
                'weathercode': weathercode
            })
        
        return {
            'location': location,
            'forecast_days': daily_forecasts,
            'summary': self._generate_summary(daily_forecasts)
        }
    
    def _generate_summary(self, daily_forecasts: List[Dict]) -> Dict[str, Any]:
        """Generate overall summary for the forecast period"""
        
        if not daily_forecasts:
            return {}
        
        all_temps = [day['temperature']['avg'] for day in daily_forecasts]
        all_rain = [day['rain_total'] for day in daily_forecasts]
        all_precipitation = [day['precipitation_total'] for day in daily_forecasts]
        all_humidity = [day['humidity']['avg'] for day in daily_forecasts]
        
        rainy_days = sum(1 for day in daily_forecasts if day['rain_total'] > 0.1)
        
        conditions = [day['condition'] for day in daily_forecasts]
        dominant_condition = max(set(conditions), key=conditions.count)
        
        return {
            'avg_temperature': round(sum(all_temps) / len(all_temps), 1),
            'temp_range': {
                'min': round(min(all_temps), 1),
                'max': round(max(all_temps), 1)
            },
            'total_rainfall': round(sum(all_rain), 1),
            'total_precipitation': round(sum(all_precipitation), 1),
            'rainy_days': rainy_days,
            'avg_humidity': round(sum(all_humidity) / len(all_humidity), 1),
            'dominant_condition': dominant_condition,
            'is_hot': sum(all_temps) / len(all_temps) > 28,
            'is_cold': sum(all_temps) / len(all_temps) < 15,
            'is_rainy_period': rainy_days >= len(daily_forecasts) / 2
        }
    
    def format_for_llm(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into natural language for LLM"""
        
        if not weather_data:
            return "Weather data unavailable."
        
        location = weather_data['location']
        summary = weather_data['summary']
        days = weather_data['forecast_days']
        
        city_name = location['city']
        if location.get('admin1'):
            city_name = f"{location['city']}, {location['admin1']}"
        
        text = f"""
## Weather Forecast for {city_name}, {location['country']}

### Weekly Overview
- Average Temperature: {summary['avg_temperature']}°C (Range: {summary['temp_range']['min']}°C - {summary['temp_range']['max']}°C)
- Total Expected Rainfall: {summary['total_rainfall']}mm over {summary['rainy_days']} days
- Total Precipitation: {summary['total_precipitation']}mm
- Average Humidity: {summary['avg_humidity']}% (estimated)
- Dominant Condition: {summary['dominant_condition']}

### Daily Breakdown
"""
        
        for day in days:
            text += f"""
**{day['day_name']} ({day['date']})**
- Temperature: {day['temperature']['min']}°C - {day['temperature']['max']}°C (avg: {day['temperature']['avg']}°C)
- Condition: {day['condition']}
- Rain: {day['rain_total']}mm
- Precipitation: {day['precipitation_total']}mm
- Estimated Humidity: {day['humidity']['avg']}%
- Wind: {day['wind_speed_avg']} km/h
"""
        
        text += "\n### Care Considerations\n"
        
        if summary['is_hot']:
            text += "- ⚠️ Hot weather expected - plants may need extra watering\n"
        if summary['is_cold']:
            text += "- ⚠️ Cool weather expected - adjust watering and protect sensitive plants\n"
        if summary['is_rainy_period']:
            text += "- 🌧️ Rainy period - reduce watering schedule, watch for overwatering\n"
        if summary['total_rainfall'] > 20:
            text += "- 💧 Heavy rainfall expected - ensure good drainage\n"
        if summary['avg_humidity'] > 70:
            text += "- 💨 High humidity - watch for fungal issues, ensure air circulation\n"
        
        return text 