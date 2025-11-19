from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
import json

load_dotenv()


class DailyTask(BaseModel):
    day: str = Field(description="Day of the week")
    task: str = Field(description="Brief task name")
    description: str = Field(description="Detailed task instructions")
    priority: str = Field(description="Priority level: high, medium, or low")


class WeeklySchedule(BaseModel):
    week: int = Field(description="Week number")
    title: str = Field(description="Week title")
    tasks: List[DailyTask] = Field(description="List of daily tasks")
    focus: str = Field(description="Main focus for the week")
    tips: List[str] = Field(description="Tips for this week")


class PlantSummary(BaseModel):
    name: str = Field(description="Plant name")
    care_level: str = Field(description="Care difficulty level")
    key_points: List[str] = Field(description="3-5 key care points")


class CarePlan(BaseModel):
    plant_summary: PlantSummary
    weekly_schedule: List[WeeklySchedule]
    general_care_tips: List[str]
    warning_signs: List[str]
    seasonal_notes: str


class GeminiCareplanGenerator:
    """Service for generating week-wise care plans using Gemini LLM with LangChain LCEL"""

    def __init__(self):
        self.api_key = os.getenv("gemini_api_key")
        if not self.api_key:
            raise ValueError("gemini_api_key not found in environment variables")

        # Initialize the Google Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=2048,
        )

    async def generate_weekly_care_plan(
        self, plant_care_guide: Dict[str, Any], weeks: int = 4
    ) -> Dict[str, Any]:
        """Generate care plan using LCEL pipe-chaining"""

        try:
            prompt_text = self._create_care_plan_prompt(plant_care_guide, weeks)

            # Updated LCEL prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a professional horticulturist creating detailed plant care plans. "
                        "Always respond with valid JSON only. No markdown, no backticks.",
                    ),
                    ("user", "{input}"),
                ]
            )

            # LCEL PIPE OPERATOR (new way)
            chain = prompt | self.llm

            # Async call → latest LCEL way
            response = await chain.ainvoke({"input": prompt_text})

            care_plan = self._parse_care_plan(response.content, plant_care_guide)
            return care_plan

        except Exception as e:
            print(f"Error generating care plan: {e}")
            return {
                "success": False,
                "error": f"Failed to generate care plan: {str(e)}",
            }

    def _create_care_plan_prompt(self, care_guide: Dict[str, Any], weeks: int) -> str:
        plant_info = care_guide.get("plant_info", {})
        watering = care_guide.get("watering", {})
        light = care_guide.get("light", {})
        soil = care_guide.get("soil", {})
        maintenance = care_guide.get("maintenance", {})
        environment = care_guide.get("environment", {})
        additional = care_guide.get("additional_info", {})

        prompt = f"""
Create a detailed {weeks}-week care plan for the following plant.

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

Generate JSON using this exact structure:

{{
  "plant_summary": {{
    "name": "",
    "care_level": "",
    "key_points": []
  }},
  "weekly_schedule": [
    {{
      "week": 1,
      "title": "",
      "tasks": [
        {{
          "day": "",
          "task": "",
          "description": "",
          "priority": "high"
        }}
      ],
      "focus": "",
      "tips": []
    }}
  ],
  "general_care_tips": [],
  "warning_signs": [],
  "seasonal_notes": ""
}}

Guidelines:
- Always return pure JSON
- No extra text, no markdown
- Include watering, pruning, fertilizing, pest-checking, soil checks
"""

        return prompt

    def _parse_care_plan(self, generated_text: str, care_guide: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the returned JSON"""

        try:
            text = generated_text.strip()

            # Remove markdown if model mistakenly adds any
            if text.startswith("```"):
                text = text.strip("`").strip()

            care_plan = json.loads(text)

            required = [
                "plant_summary",
                "weekly_schedule",
                "general_care_tips",
                "warning_signs",
                "seasonal_notes",
            ]
            for k in required:
                if k not in care_plan:
                    raise ValueError(f"Missing key: {k}")

            care_plan["success"] = True
            care_plan["plant_info"] = care_guide.get("plant_info", {})
            return care_plan

        except Exception as e:
            return {
                "success": False,
                "error": f"Invalid JSON returned: {str(e)}",
                "raw_text": generated_text[:1000],
            }