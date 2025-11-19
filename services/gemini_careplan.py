from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class GeminiCareplanGenerator:
    """Service for generating week-wise care plans using Gemini LLM (Natural Language Version)"""

    def __init__(self):
        self.api_key = os.getenv("gemini_api_key")
        if not self.api_key:
            raise ValueError("gemini_api_key not found in environment variables")

        # Latest Google Gemini initialization
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=3096,
        )

    async def generate_weekly_care_plan(
        self, plant_care_guide: Dict[str, Any], weeks: int = 4
    ) -> Dict[str, Any]:
        """Generate natural language care plan using LCEL (latest pipeline style)"""

        try:
            # Construct prompt text
            prompt_text = self._create_care_plan_prompt(plant_care_guide, weeks)

            # Correct LCEL prompt template syntax
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a professional horticulturist and plant care expert. "
                        "Create detailed, practical, and easy-to-follow care plans in natural language. "
                        "Be specific, friendly, and educational. Use clear formatting with headings and bullet points."
                    ),
                    ("user", "{input}"),
                ]
            )

            # LCEL pipeline using "|"
            chain = prompt | self.llm

            # Produce output
            response = await chain.ainvoke({"input": prompt_text})

            return {
                "success": True,
                "plant_name": plant_care_guide.get("plant_info", {}).get("name", "Unknown"),
                "scientific_name": plant_care_guide.get("plant_info", {}).get("scientific_name", "Unknown"),
                "care_level": plant_care_guide.get("plant_info", {}).get("care_level", "Unknown"),
                "care_plan_text": response.content,
                "weeks": weeks
            }

        except Exception as e:
            print(f"Error generating care plan: {e}")
            return {
                "success": False,
                "error": f"Failed to generate care plan: {str(e)}",
            }

    def _create_care_plan_prompt(self, care_guide: Dict[str, Any], weeks: int) -> str:
        """Build the natural language prompt for Gemini"""

        plant_info = care_guide.get("plant_info", {})
        watering = care_guide.get("watering", {})
        light = care_guide.get("light", {})
        soil = care_guide.get("soil", {})
        maintenance = care_guide.get("maintenance", {})
        environment = care_guide.get("environment", {})
        additional = care_guide.get("additional_info", {})

        # Build comprehensive plant data
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
Create a comprehensive {weeks}-week care plan for the following plant. 
Write in natural, conversational language that a plant owner can easily follow.

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

## TASK
Write a detailed {weeks}-week care plan in natural language. Structure your response as follows:

### 1. Plant Overview & Care Level
- Brief introduction
- Why people like growing it
- Care difficulty summary

### 2. Week-by-Week Care Schedule
For each of the {weeks} weeks, include:
- **Week X: [Catchy Title]**
  - Main focus
  - Daily/weekly actionable tasks
  - Monitoring checklist
  - Tips & best practices
  - Common mistakes to avoid

Include tasks such as:
- Watering with timing & method
- Light/position adjustments
- Soil checks & fertilizing timing
- Pruning, trimming, grooming
- Pest inspection
- Seasonal adjustments

### 3. General Care Guidelines
- Long-term care
- Seasonal care
- Propagation tips
- Troubleshooting issues

### 4. Warning Signs to Watch For
- Overwatering/underwatering
- Pests/diseases
- Nutrient deficiencies
- Heat/light stress

### 5. Pro Tips for Success
- Expert insights
- High-impact habits
- Bonus optional techniques

Make it practical, specific, and encouraging.
"""

        return prompt