import json
import os
from typing import List, Dict, Any, Optional

def get_fallback_path():
    """Returns the absolute path to fallback_models.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of AgentSwarm-2
    return os.path.join(base_dir, "backend", "fallback_models.json")

def load_and_tier_models(file_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads models from fallback_models.json and tiers them by price.
    Tier 1: High-end (> $5/1M tokens)
    Tier 2: Mid-range (> $0.50/1M tokens)
    Tier 3: Cheap (<= $0.50/1M tokens)
    """
    path = file_path or get_fallback_path()
    if not os.path.exists(path):
        # Fallback if file doesn't exist (should not happen in production)
        logger.error(f"Models file not found at {path}")
        return {"tier1": [], "tier2": [], "tier3": []}

    try:
        with open(path, "r") as f:
            models = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return {"tier1": [], "tier2": [], "tier3": []}

    tiers = {
        "tier1": [],
        "tier2": [],
        "tier3": []
    }

    for m in models:
        pricing = m.get("pricing", {})
        # Use prompt price for tiering
        p_in = float(pricing.get("prompt", 0))
        
        # Capability tags
        params = m.get("supported_parameters", [])
        m["has_tools"] = "tools" in params
        m["has_search"] = pricing.get("web_search") is not None or "search" in m.get("description", "").lower()
        
        # High Tier: > $5 per 1M tokens ($0.000005)
        if p_in >= 0.000005: # Changed to >= to catch $5 exactly
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

def get_model_info(model_id: str, file_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves full model info for a specific ID."""
    path = file_path or get_fallback_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            models = json.load(f)
    except: return None
    
    for m in models:
        if m["id"] == model_id:
            # Re-run capability detection for consistency
            pricing = m.get("pricing", {})
            params = m.get("supported_parameters", [])
            m["has_tools"] = "tools" in params
            m["has_search"] = pricing.get("web_search") is not None or "search" in m.get("description", "").lower()
            
            # Add tier info too
            p_in = float(pricing.get("prompt", 0))
            if p_in >= 0.000005: m["tier"] = 1
            elif p_in > 0.0000005: m["tier"] = 2
            else: m["tier"] = 3
            
            return m
    return None

def get_optimal_model(tier: int, min_context: int, tiers: Dict[str, List[Dict[str, Any]]], required_capabilities: List[str] = None) -> str:
    """
    Finds the best model in a tier that satisfies the context requirement and capabilities.
    required_capabilities: list of strings like ["has_tools", "has_search"]
    """
    tier_key = f"tier{tier}"
    tier_models = tiers.get(tier_key, [])
    
    candidates = [m for m in tier_models if m.get("context_length", 0) >= min_context]
    
    if required_capabilities:
        candidates = [
            m for m in candidates 
            if all(m.get(cap, False) for cap in required_capabilities)
        ]
    
    if not candidates:
        # Fallback to higher tier if empty or lower tier if higher tier requested
        if tier > 1:
            return get_optimal_model(tier - 1, min_context, tiers, required_capabilities)
        # If still no candidates and capabilities were requested, try ignoring tier if we can find ANY model
        if required_capabilities and tier == 1:
            # Last ditch effort: search all tiers for a capable model
            all_models = tiers["tier1"] + tiers["tier2"] + tiers["tier3"]
            cap_candidates = [
                m for m in all_models 
                if all(m.get(cap, False) for cap in required_capabilities) and m.get("context_length", 0) >= min_context
            ]
            if cap_candidates:
                cap_candidates.sort(key=lambda x: float(x.get("pricing", {}).get("prompt", 0)))
                return cap_candidates[0]["id"]
                
        return "openai/gpt-4o" # Ultimate fallback

    # Sort by context length descending within the tier
    candidates.sort(key=lambda x: x.get("context_length", 0), reverse=True)
    return candidates[0]["id"]
