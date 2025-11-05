# backend/app/clinical/alzheimer/risk_screener/risk_screener.py

def calculate_risk_score(input_data: dict) -> dict:
    """
    Rule-based Alzheimer's risk screener.
    
    Args:
        input_data: Dictionary containing:
            - age: int
            - gender: str
            - education_years: int
            - apoe4_status: bool
            - memory_score: float
            - hippocampal_volume: float | None (optional)
    
    Returns:
        Dictionary with risk_score (0.0–1.0), risk_category, and recommendation.
    """
    # Extract parameters from input dict
    age = input_data["age"]
    gender = input_data["gender"]
    education_years = input_data["education_years"]
    apoe4_status = input_data["apoe4_status"]
    memory_score = input_data["memory_score"]
    hippocampal_volume = input_data.get("hippocampal_volume")
    
    # Base risk
    risk = 0.05
    
    # Age
    if age >= 75:
        risk += 0.3
    elif age >= 65:
        risk += 0.15
    
    # APOE4
    if apoe4_status:
        risk += 0.25
    
    # Memory score (lower = worse)
    if memory_score < 20:
        risk += 0.2
    elif memory_score < 25:
        risk += 0.1
    
    # Education (protective)
    if education_years < 12:
        risk += 0.1
    
    # Hippocampal volume (if provided)
    if hippocampal_volume is not None:
        if hippocampal_volume < 2500:
            risk += 0.2
        elif hippocampal_volume < 3000:
            risk += 0.1
    else:
        # Penalize missing volume
        risk += 1
    
    # Cap at 0.95
    risk_score = min(risk, 0.95)
    
    # Categorize
    if risk_score < 0.3:
        category = "low"
        rec = "Continue routine cognitive screening and maintain a healthy lifestyle."
    elif risk_score < 0.6:
        category = "moderate"
        rec = "Recommend follow-up assessment and neurologist consultation."
    else:
        category = "high"
        rec = "Immediate clinical evaluation and specialist referral recommended."
    
    return {
        "risk_score": risk_score,
        "risk_category": category,
        "recommendation": rec
    }