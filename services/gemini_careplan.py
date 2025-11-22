from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiCareplanGenerator:
    """Service for generating day-by-day care plans using Gemini LLM"""

    def __init__(self):
        self.api_key = os.getenv("gemini_api_key")
        if not self.api_key:
            raise ValueError("gemini_api_key not found in environment variables")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )

    async def generate_daily_care_plan(
        self, 
        plant_care_guide: Dict[str, Any], 
        days: int = 7,
        weather_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate natural language care plan for specified number of days with weather context"""

        try:
            prompt_text = self._create_care_plan_prompt(plant_care_guide, days, weather_data)

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a professional horticulturist and plant care expert. "
                        "Create detailed, practical, and easy-to-follow daily care plans. "
                        "Be specific about what to do each day, friendly, and educational. "
                        "Use clear formatting with headings and bullet points. "
                        "IMPORTANT: When weather data is provided, ALWAYS adjust care instructions "
                        "based on the actual forecasted conditions. Consider temperature, rainfall, "
                        "humidity, and wind when recommending watering, placement, and protection. "
                        "Complete ALL sections fully without cutting off. "
                        "Ensure the entire care plan is generated from Day 1 through the final day, "
                        "including all 5 main sections."
                    ),
                    ("user", "{input}"),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({"input": prompt_text})

            return {
                "success": True,
                "plant_name": plant_care_guide.get("plant_info", {}).get("name", "Unknown"),
                "scientific_name": plant_care_guide.get("plant_info", {}).get("scientific_name", "Unknown"),
                "care_level": plant_care_guide.get("plant_info", {}).get("care_level", "Unknown"),
                "care_plan_text": response.content,
                "days": days,
                "weather_included": weather_data is not None
            }

        except Exception as e:
            print(f"Error generating care plan: {e}")
            return {
                "success": False,
                "error": f"Failed to generate care plan: {str(e)}",
            }

    def _create_care_plan_prompt(
        self, 
        care_guide: Dict[str, Any], 
        days: int,
        weather_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the natural language prompt for Gemini with weather context"""

        plant_info = care_guide.get("plant_info", {})
        watering = care_guide.get("watering", {})
        light = care_guide.get("light", {})
        soil = care_guide.get("soil", {})
        maintenance = care_guide.get("maintenance", {})
        environment = care_guide.get("environment", {})
        additional = care_guide.get("additional_info", {})

        watering_info = f"{watering.get('frequency', 'Unknown')}"
        if watering.get("period"):
            watering_info += f" ({watering.get('period')})"
        if watering.get("benchmark", {}).get("value"):
            benchmark = watering.get("benchmark", {})
            watering_info += (
                f" - approximately every {benchmark.get('value')} {benchmark.get('unit', '')}"
            )

        sunlight_info = ", ".join(light.get("requirements", ["Unknown"]))
        soil_info = ", ".join(soil.get("types", ["Well-draining soil"]))
        pruning_months = ", ".join(maintenance.get("pruning_months", ["As needed"]))
        pest_info = ", ".join(additional.get("pest_susceptibility", ["None noted"]))

        prompt = f"""
Create a comprehensive {days}-day daily care plan for the following plant. 
Write in natural, conversational language that a plant owner can easily follow each day.

## PLANT INFORMATION
**Common Name:** {plant_info.get('name', 'Unknown')}
**Scientific Name:** {plant_info.get('scientific_name', 'Unknown')}
**Family:** {plant_info.get('family', 'Unknown')}
**Type:** {plant_info.get('type', 'Unknown')}
**Care Level:** {plant_info.get('care_level', 'Unknown')}
**Growth Rate:** {plant_info.get('growth_rate', 'Unknown')}
**Life Cycle:** {environment.get('cycle', 'Unknown')}

## CARE REQUIREMENTS
**Watering:** {watering_info}
**Sunlight:** {sunlight_info}
**Soil:** {soil_info}
**Maintenance Level:** {maintenance.get('level', 'Unknown')}
**Best Pruning Months:** {pruning_months}
**Flowering Season:** {additional.get('flowering_season', 'N/A')}
**Common Pests:** {pest_info}
**Edible Fruit:** {'Yes' if additional.get('edible_fruit') else 'No'}
"""

        # Add weather data if available
        if weather_data:
            from services.weather_service import WeatherService
            weather_service = WeatherService()
            weather_text = weather_service.format_for_llm(weather_data)
            prompt += f"\n{weather_text}\n"
            prompt += """
⚠️ CRITICAL: You MUST adapt this care plan to the actual weather forecast provided above.
- Adjust watering based on rainfall predictions
- Modify placement/protection based on temperature extremes
- Account for humidity levels in pest/disease warnings
- Consider wind conditions for fragile plants
- Be SPECIFIC about what days need adjustments based on the forecast
"""

        prompt += f"""
## TASK
Write a detailed {days}-day daily care plan. You MUST complete ALL {days} days and all 5 sections below.
{f'IMPORTANT: Adapt care instructions to match the weather forecast data provided above!' if weather_data else ''}

### 1. Plant Overview & Getting Started
Write 3-4 paragraphs covering:
- Brief introduction to this plant (2-3 sentences)
- What makes it special (2-3 sentences)
- Care difficulty level (1-2 sentences)
- What to expect in the first {days} days (2-3 sentences)
{f'- Weather impact for this week (2-3 sentences about how forecast affects care)' if weather_data else ''}

### 2. Daily Care Schedule
Provide detailed tasks for EACH of the {days} days. For each day include:

**Day X: [Catchy Title]**
- **Morning Tasks:** 2-3 specific actions {f'(adjusted for weather!)' if weather_data else ''}
- **Afternoon/Evening Tasks:** 1-2 specific actions
- **What to Observe Today:** 1-2 items
- **Key Tip for the Day:** 1 practical tip {f'(consider weather conditions!)' if weather_data else ''}

Make each day unique with different focuses:
- Day 1: Welcome & placement {f'(consider forecast)' if weather_data else ''}
- Day 2: Soil & watering assessment {f'(check rain predictions)' if weather_data else ''}
- Day 3: Leaf inspection & grooming
- Day 4: Light & temperature check {f'(adjust for weather)' if weather_data else ''}
- Day 5: Mid-week wellness check
- Day 6: Pest patrol {f'(humidity awareness)' if weather_data else ''}
- Day 7: Week review & future planning {f'(next week weather prep)' if weather_data else ''}

### 3. Quick Daily Checklist (Keep this concise - 5-6 items max)
Simple 2-5 minute daily routine:
- ✓ Item 1
- ✓ Item 2
- ✓ Item 3
{f'- ✓ Weather-based adjustment' if weather_data else ''}

### 4. Warning Signs to Watch For (Keep concise - 3-4 categories)
What to look for:
- **Normal appearance:** Brief description
- **Problem signs:** 2-3 key warnings {f'(include weather-related issues)' if weather_data else ''}
- **When to act:** Brief advice

### 5. Pro Tips for Success (Keep concise - 3-4 tips)
Most important advice:
- Tip 1
- Tip 2
- Tip 3
{f'- Weather-specific tip for this week' if weather_data else ''}

CRITICAL: You MUST complete the entire plan including ALL {days} days in section 2 and all other sections. Do not stop mid-generation.
{f'REMEMBER: Adapt ALL recommendations to the actual weather forecast provided!' if weather_data else ''}
"""

        return prompt