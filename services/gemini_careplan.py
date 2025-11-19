import httpx
from typing import Dict, Any
import os
from dotenv import load_dotenv
import json

load_dotenv()


class GeminiCareplanGenerator:
    """Service for generating week-wise care plans using Gemini LLM"""
    
    def __init__(self):
        self.api_key = os.getenv("gemini_api_key")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
    async def generate_weekly_care_plan(self, plant_care_guide: Dict[str, Any], weeks: int = 4) -> Dict[str, Any]:
        """
        Generate a week-wise care plan using Gemini LLM
        
        Args:
            plant_care_guide: Plant care information from Perenual
            weeks: Number of weeks to generate plan for (default: 4)
            
        Returns:
            Structured weekly care plan
        """
        try:
            # Create prompt for Gemini
            prompt = self._create_care_plan_prompt(plant_care_guide, weeks)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"{self.base_url}?key={self.api_key}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 2048,
                    }
                }
                
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Extract generated text
                generated_text = data['candidates'][0]['content']['parts'][0]['text']
                
                # Parse the JSON response
                care_plan = self._parse_care_plan(generated_text, plant_care_guide)
                
                return care_plan
                
        except httpx.HTTPStatusError as e:
            print(f"Gemini API error: {e}")
            return {'success': False, 'error': 'Failed to generate care plan'}
        except Exception as e:
            print(f"Error generating care plan: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_care_plan_prompt(self, care_guide: Dict[str, Any], weeks: int) -> str:
        """Create a detailed prompt for Gemini to generate care plan"""
        
        plant_info = care_guide.get('plant_info', {})
        watering = care_guide.get('watering', {})
        light = care_guide.get('light', {})
        soil = care_guide.get('soil', {})
        maintenance = care_guide.get('maintenance', {})
        environment = care_guide.get('environment', {})
        additional = care_guide.get('additional_info', {})
        
        prompt = f"""You are a professional horticulturist. Create a detailed {weeks}-week care plan for the following plant.

Plant Information:
- Name: {plant_info.get('name', 'Unknown')}
- Scientific Name: {plant_info.get('scientific_name', 'Unknown')}
- Type: {plant_info.get('type', 'Unknown')}
- Care Level: {plant_info.get('care_level', 'Unknown')}
- Growth Rate: {plant_info.get('growth_rate', 'Unknown')}

Care Requirements:
- Watering: {watering.get('frequency', 'Unknown')} ({watering.get('period', 'as needed')})
- Watering Benchmark: Every {watering.get('benchmark', {}).get('value', 'N/A')} {watering.get('benchmark', {}).get('unit', '')}
- Sunlight: {', '.join(light.get('requirements', ['Unknown']))}
- Soil: {', '.join(soil.get('types', ['Unknown']))}
- Maintenance Level: {maintenance.get('level', 'Unknown')}
- Pruning Months: {', '.join(maintenance.get('pruning_months', ['As needed']))}
- Cycle: {environment.get('cycle', 'Unknown')}
- Flowering Season: {additional.get('flowering_season', 'N/A')}
- Pest Susceptibility: {', '.join(additional.get('pest_susceptibility', ['None noted']))}

Please generate a week-by-week care plan in the following JSON format:

{{
  "plant_summary": {{
    "name": "Plant name",
    "care_level": "Easy/Moderate/High",
    "key_points": ["Key point 1", "Key point 2", "Key point 3"]
  }},
  "weekly_schedule": [
    {{
      "week": 1,
      "title": "Week 1: Getting Started",
      "tasks": [
        {{
          "day": "Monday",
          "task": "Check soil moisture",
          "description": "Detailed instruction",
          "priority": "high/medium/low"
        }}
      ],
      "focus": "What to focus on this week",
      "tips": ["Tip 1", "Tip 2"]
    }}
  ],
  "general_care_tips": ["Overall tip 1", "Overall tip 2", "Overall tip 3"],
  "warning_signs": ["Sign 1: What to do", "Sign 2: What to do"],
  "seasonal_notes": "Any seasonal considerations"
}}

Important Guidelines:
1. Make the schedule practical and easy to follow
2. Include specific days for watering based on the plant's needs
3. Add fertilizing schedule if appropriate
4. Include pruning/deadheading tasks in appropriate weeks
5. Mention pest inspection days
6. Add light/location adjustment reminders
7. Include soil check tasks
8. Make tasks specific and actionable
9. Vary tasks throughout the week (don't make every day watering)
10. Consider the plant's growth cycle and adjust weekly tasks accordingly

Respond ONLY with the valid JSON. Do not include any markdown formatting or backticks."""

        return prompt
    
    def _parse_care_plan(self, generated_text: str, care_guide: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the generated care plan"""
        try:
            # Remove markdown formatting if present
            text = generated_text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            # Parse JSON
            care_plan = json.loads(text)
            
            # Add metadata
            care_plan['success'] = True
            care_plan['plant_info'] = care_guide.get('plant_info', {})
            care_plan['source_data'] = {
                'watering': care_guide.get('watering', {}),
                'light': care_guide.get('light', {}),
                'soil': care_guide.get('soil', {}),
                'maintenance': care_guide.get('maintenance', {})
            }
            
            return care_plan
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Generated text: {generated_text}")
            return {
                'success': False,
                'error': 'Failed to parse care plan',
                'raw_text': generated_text
            }