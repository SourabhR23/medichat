from typing import List, Dict, Any
import re
from euriai import EuriaiClient
from app.config import EURI_API_KEY, LLM_MODEL

TEMPERATURE = 0.7

# Initialize Euri AI client (reuse it accross calls)
client = EuriaiClient(
    api_key=EURI_API_KEY,
    model=LLM_MODEL
)

def get_chat_model(api_key: str = None) -> Dict[str, Any]:
    """Get chat model configuration for self-hosted LLM via Euri"""
    # This function is now mostly metadata; actual calls use the `client` above
    return {
        "model": LLM_MODEL,
        "temperature": TEMPERATURE,
        "api_key": api_key
    }

def ask_chat_model(chat_model, prompt: str) -> str:
    try:
        response = client.generate_completion(
            prompt=prompt,
            temperature=chat_model.get("temperature", TEMPERATURE),
            max_tokens=300,
        )

        # CASE 1: Dict response (most common with Euri)
        if isinstance(response, dict) and "choices" in response:
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "").strip()

        return str(response)

    except Exception as e:
        print(f"Unexpected error calling Euri LLM: {e}")
        return "Error: Unable to get response from the AI model."


def generate_medical_insights(text: str) -> Dict[str, Any]:
    """Generate medical insights from text analysis"""
    insights: Dict[str, Any] = {
        "medical_terms": [],
        "potential_conditions": [],
        "medications": [],
        "vital_signs": [],
        "symptoms": [],
        "recommendations": [],
    }

    # Medical terms detection
    medical_patterns: Dict[str, str] = {
        "medications": r"\b(?:medication|drug|prescription|tablet|capsule|injection|dose|mg|ml)\b",
        "symptoms": r"\b(?:pain|ache|fever|nausea|dizziness|fatigue|weakness|shortness|breath)\b",
        "conditions": r"\b(?:diabetes|hypertension|asthma|pneumonia|infection|inflammation|chronic|acute)\b",
        "vital_signs": r"\b(?:blood pressure|heart rate|temperature|pulse|respiratory rate|oxygen saturation)\b",
    }

    text_lower = text.lower()

    for category, pattern in medical_patterns.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        insights[category] = list(set(matches))

    return insights

def enhance_medical_response(response: str, insights: Dict[str, Any]) -> str:
    """Enhance medical response with additional context"""
    enhanced_response = response

    # NOTE: Your original string had mojibake; here it's cleaned to normal emoji
    disclaimer = (
        "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only "
        "and should not be considered as medical advice. Please consult with a qualified "
        "healthcare professional for proper diagnosis and treatment."
    )

    # Add relevant insights if available
    if insights.get("medications"):
        enhanced_response += (
            f"\n\n💊 **Medications mentioned**: "
            f"{', '.join(insights['medications'][:3])}"
        )

    if insights.get("symptoms"):
        enhanced_response += (
            f"\n\n🩺 **Symptoms identified**: "
            f"{', '.join(insights['symptoms'][:3])}"
        )

    if insights.get("conditions"):
        enhanced_response += (
            f"\n\n🏥 **Conditions discussed**: "
            f"{', '.join(insights['conditions'][:3])}"
        )

    enhanced_response += disclaimer

    return enhanced_response