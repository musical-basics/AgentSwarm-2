import json
import os
from typing import List, Dict, Any, Optional

def load_and_tier_models(file_path: str = "fallback_models.json") -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads models from fallback_models.json and tiers them by price.
    Tier 1: High-end (> $5/1M tokens)
    Tier 2: Mid-range (> $0.50/1M tokens)
    Tier 3: Cheap (<= $0.50/1M tokens)
    """
    if not os.path.exists(file_path):
        # Fallback if file doesn't exist (should not happen in production)
        return {"tier1": [], "tier2": [], "tier3": []}

    with open(file_path, "r") as f:
        models = json.load(f)

    tiers = {
        "tier1": [],
        "tier2": [],
        "tier3": []
    }

    for m in models:
        pricing = m.get("pricing", {})
        # Use prompt price for tiering
        p_in = float(pricing.get("prompt", 0))
        
        # High Tier: > $5 per 1M tokens ($0.000005)
        if p_in > 0.000005:
            m["tier"] = 1
            tiers["tier1"].append(m)
        # Medium Tier: > $0.50 per 1M tokens ($0.0000005)
        elif p_in > 0.0000005:
            m["tier"] = 2
            tiers["tier2"].append(m)
        # Cheap Tier: <= $0.50 per 1M tokens
        else:
            m["tier"] = 3
            tiers["tier3"].append(m)

    return tiers

def get_model_info(model_id: str, file_path: str = "fallback_models.json") -> Optional[Dict[str, Any]]:
    """Retrieves full model info for a specific ID."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        models = json.load(f)
    for m in models:
        if m["id"] == model_id:
            return m
    return None

def get_optimal_model(tier: int, min_context: int, tiers: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    Finds the best model in a tier that satisfies the context requirement.
    Picks the one with the largest context within the tier as a safety measure.
    """
    tier_key = f"tier{tier}"
    candidates = [m for m in tiers.get(tier_key, []) if m.get("context_length", 0) >= min_context]
    
    if not candidates:
        # Fallback to higher tier if empty or lower tier if higher tier requested
        if tier > 1:
            return get_optimal_model(tier - 1, min_context, tiers)
        return "openai/gpt-4o" # Ultimate fallback

    # Sort by context length descending within the tier
    candidates.sort(key=lambda x: x.get("context_length", 0), reverse=True)
    return candidates[0]["id"]
