import os
import re
import json
import urllib.parse

from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

from models import HomeBudgetInput, PartyBudgetInput, JewelryBudgetInput

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError(
        "No GOOGLE_API_KEY found in environment variables. "
        "Please set it in your .env file."
    )

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-3.6-flash")


def extract_json_from_response(text: str) -> dict:
    cleaned = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if match:
            return json.loads(match.group(0))

        raise


def _add_links(result: dict, breakdown_key: str, platforms: dict):
    for category in result.get(breakdown_key, []):
        for item in category.get("items", []):
            terms = item.get("search_terms", "")

            if terms:
                q = urllib.parse.quote_plus(terms)

                item["shopping_links"] = {
                    name: url.format(q=q)
                    for name, url in platforms.items()
                }


HOME_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ikea": "https://www.ikea.com/in/en/search/?q={q}",
    "myntra": "https://www.myntra.com/search?q={q}",
}


PARTY_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "swiggy": "https://www.swiggy.com/search?query={q}",
    "zomato": "https://www.zomato.com/search?q={q}",
    "bookmyshow": "https://in.bookmyshow.com/search?q={q}",
}


JEWELRY_PLATFORMS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "tanishq": "https://www.tanishq.co.in/search?q={q}",
    "caratlane": "https://www.caratlane.com/search?q={q}",
}


def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:

    prompt = f"""
You are PocketSmart AI, a smart budget-aware home interior recommendation assistant.

User requirements:

Total Budget: ₹{budget_input.total_budget}
Number of Lights/Fixtures: {budget_input.num_lights}
Number of Ceiling Fans: {budget_input.num_fans}
Number of Furniture Pieces: {budget_input.num_furniture}
Number of Dining Tables: {budget_input.num_dining_tables}

Living Room: {budget_input.has_living_room}
Kitchen: {budget_input.has_kitchen}
Bedroom: {budget_input.has_bedroom}

Additional Requirements:
{budget_input.additional_requirements or "None"}

Create practical and cost-effective recommendations within the given budget.

Return ONLY valid JSON in this exact structure:

{{
  "total_budget": 0,
  "remaining_budget": 0,
  "budget_breakdown": [
    {{
      "category": "string",
      "allocation": 0,
      "items": [
        {{
          "name": "string",
          "quantity": 0,
          "estimated_price": 0,
          "description": "string",
          "search_terms": "string"
        }}
      ]
    }}
  ],
  "additional_suggestions": [
    "string"
  ]
}}

Do not include markdown or explanations outside the JSON.
"""

    response = model.generate_content(prompt)

    result = extract_json_from_response(response.text)

    _add_links(
        result,
        "budget_breakdown",
        HOME_PLATFORMS
    )

    return result


def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:

    prompt = f"""
You are PocketSmart AI, a smart budget-aware party planning recommendation assistant.

User requirements:

Total Budget: ₹{budget_input.total_budget}
Party Type: {budget_input.party_type}
Number of Guests: {budget_input.num_guests}
Venue Type: {budget_input.venue_type or "Not specified"}

Catering Required: {budget_input.needs_catering}
Decoration Required: {budget_input.needs_decoration}
Entertainment Required: {budget_input.needs_entertainment}

Additional Requirements:
{budget_input.additional_requirements or "None"}

Create practical and cost-effective party recommendations within the given budget.

Consider categories such as venue, catering, decoration and entertainment based on the user's requirements.

Return ONLY valid JSON in this exact structure:

{{
  "total_budget": 0,
  "remaining_budget": 0,
  "budget_breakdown": [
    {{
      "category": "string",
      "allocation": 0,
      "items": [
        {{
          "name": "string",
          "quantity": 0,
          "estimated_price": 0,
          "description": "string",
          "search_terms": "string"
        }}
      ]
    }}
  ],
  "additional_suggestions": [
    "string"
  ]
}}

Do not include markdown or explanations outside the JSON.
"""

    response = model.generate_content(prompt)

    result = extract_json_from_response(response.text)

    _add_links(
        result,
        "budget_breakdown",
        PARTY_PLATFORMS
    )

    return result


def get_jewelry_recommendations(
    budget_input: JewelryBudgetInput,
    image_path: str = None
) -> dict:

    base_prompt = f"""
You are PocketSmart AI, a smart budget-aware jewelry recommendation assistant.

User requirements:

Total Budget: ₹{budget_input.total_budget}
Occasion: {budget_input.occasion}
Preferences: {budget_input.preferences or "None"}

Recommend suitable jewelry options according to the occasion, style and budget.

Return ONLY valid JSON.

Consider the outfit image if one is provided.
"""

    json_format = """

Use this exact JSON structure:

{
  "total_budget": 0,
  "jewelry_recommendations": [
    {
      "name": "string",
      "type": "string",
      "estimated_price": 0,
      "description": "string",
      "search_terms": "string"
    }
  ],
  "additional_suggestions": [
    "string"
  ]
}

Do not include markdown or explanations outside the JSON.
"""

    if image_path:

        img = Image.open(image_path)

        prompt = (
            base_prompt
            + "\nAn outfit image is attached. Consider its colors and style.\n"
            + json_format
        )

        response = model.generate_content(
            [prompt, img]
        )

    else:

        prompt = base_prompt + json_format

        response = model.generate_content(prompt)

    result = extract_json_from_response(response.text)

    for item in result.get("jewelry_recommendations", []):

        terms = item.get("search_terms", "")

        if terms:

            q = urllib.parse.quote_plus(terms)

            item["shopping_links"] = {
                name: url.format(q=q)
                for name, url in JEWELRY_PLATFORMS.items()
            }

    return result