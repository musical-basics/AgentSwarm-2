import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

TIER1_PROMPT_PRICE = 0.0000015  # $1.50 / 1M prompt tokens
TIER2_PROMPT_PRICE = 0.0000005  # $0.50 / 1M prompt tokens

# ---------------------------------------------------------------------------
# Model quality controls
# ---------------------------------------------------------------------------

# Models excluded from all selections — known low quality, broken pricing,
# or unreliable inference. Add model IDs here to ban them globally.
MODEL_BLACKLIST: frozenset = frozenset({
    "reka/reka-edge",           # Produces garbled/halluincated outputs
    "openrouter/free",          # Generic catch-all, unpredictable quality
    "openrouter/bodybuilder",   # Negative pricing — broken entry
    # Free-tier variants: inconsistent availability, no SLA
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "arcee-ai/trinity-mini:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
})

# Preferred models per tier, ordered by priority (first match wins).
# These are checked before falling back to cheapest/largest-context sorting.
# Criteria: known reliability + high value-to-cost ratio.
MODEL_PREFERRED: list = [
    # --- Tier 1 (>$1.50/M) ---
    "anthropic/claude-opus-4.6",        # 1M ctx, best-in-class quality
    # --- Tier 2 (>$0.50/M and <=$1.50/M) ---
    "anthropic/claude-sonnet-4.6",      # 1M ctx, excellent quality/cost
    "google/gemini-3.1-pro-preview",     # 1M ctx, latest Gemini Pro generation
    "openai/gpt-5.4-mini",              # 400k ctx, reliable mid-tier
    "qwen/qwen3-max-thinking",          # 262k ctx, strong reasoning budget
    # --- Tier 3 (≤$0.50/M) ---
    "deepseek/deepseek-v3.2",           # $0.26/M, strong coding + reasoning
    "x-ai/grok-4.1-fast",              # $0.20/M, 2M ctx, very capable
    "qwen/qwen3.5-flash-02-23",         # $0.065/M, 1M ctx, highest value score
    "google/gemini-2.0-flash-001",      # $0.10/M, 1M ctx, reliable
    "openai/gpt-4.1-nano",              # $0.10/M, 1M ctx, solid nano
    "meta-llama/llama-4-maverick",      # $0.15/M, 1M ctx, strong open model
]

# ---------------------------------------------------------------------------

def get_fallback_path():
    """Returns the absolute path to fallback_models.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root of AgentSwarm-2
    return os.path.join(base_dir, "backend", "fallback_models.json")

def load_and_tier_models(file_path: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads models from fallback_models.json and tiers them by price.
    Tier 1: High-end (> $1.50/1M tokens)
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
        # Only trust the authoritative `web_search` pricing key for search capability.
        # Description keyword matching creates too many false positives (safety models, code search, etc.)
        m["has_search"] = pricing.get("web_search") is not None
        
        # High Tier: >= $1.50 per 1M tokens
        if p_in >= TIER1_PROMPT_PRICE:
            m["tier"] = 1
            tiers["tier1"].append(m)
        # Medium Tier: > $0.50 and < $1.50 per 1M tokens
        elif p_in > TIER2_PROMPT_PRICE:
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
            m["has_search"] = pricing.get("web_search") is not None
            
            # Add tier info too
            p_in = float(pricing.get("prompt", 0))
            if p_in >= TIER1_PROMPT_PRICE: m["tier"] = 1
            elif p_in > TIER2_PROMPT_PRICE: m["tier"] = 2
            else: m["tier"] = 3
            
            return m
    return None

def get_optimal_model(
    tier: int,
    min_context: int,
    tiers: Dict[str, List[Dict[str, Any]]],
    required_capabilities: List[str] = None,
    excluded_model_ids: Optional[List[str]] = None,
    excluded_prefixes: Optional[List[str]] = None,
) -> str:
    """
    Finds the best model in a tier that satisfies the context requirement and capabilities.
    required_capabilities: list of strings like ["has_tools", "has_search"]
    
    When capabilities are required (e.g. has_search), the function will:
    1. Look in the requested tier first for a capable model
    2. If not found, search ALL tiers and return cheapest capable model
    3. This means a search-required node will never fall back to a non-searching model
    """
    # Always merge the global blacklist into the excluded set
    excluded_id_set = set(excluded_model_ids or []) | MODEL_BLACKLIST
    excluded_prefix_tuple = tuple(excluded_prefixes or [])

    def _is_allowed(model: Dict[str, Any]) -> bool:
        model_id = model.get("id", "")
        if model_id in excluded_id_set:
            return False
        if excluded_prefix_tuple and model_id.startswith(excluded_prefix_tuple):
            return False
        return True

    tier_key = f"tier{tier}"
    tier_models = tiers.get(tier_key, [])
    
    candidates = [
        m for m in tier_models
        if m.get("context_length", 0) >= min_context and _is_allowed(m)
    ]
    
    if required_capabilities:
        capable_candidates = [
            m for m in candidates 
            if all(m.get(cap, False) for cap in required_capabilities)
        ]
        
        if not capable_candidates:
            # No capable model in this tier — search ALL tiers for cheapest capable model
            all_models = tiers.get("tier1", []) + tiers.get("tier2", []) + tiers.get("tier3", [])
            all_capable = [
                m for m in all_models 
                if all(m.get(cap, False) for cap in required_capabilities)
                and m.get("context_length", 0) >= min_context
                and _is_allowed(m)
            ]
            if all_capable:
                # Check preferred list first, then fall back to cheapest
                all_capable_ids = {m["id"] for m in all_capable}
                for preferred_id in MODEL_PREFERRED:
                    if preferred_id in all_capable_ids:
                        return preferred_id
                all_capable.sort(key=lambda x: float(x.get("pricing", {}).get("prompt", 0)))
                return all_capable[0]["id"]
            # No capable model exists at all — ultimate fallback
            return "openai/gpt-4o"
        
        # Check preferred list first among capable candidates in this tier
        capable_ids = {m["id"] for m in capable_candidates}
        for preferred_id in MODEL_PREFERRED:
            if preferred_id in capable_ids:
                return preferred_id

        # Fall back to cheapest capable model within tier
        capable_candidates.sort(key=lambda x: float(x.get("pricing", {}).get("prompt", 0)))
        return capable_candidates[0]["id"]
    
    if not candidates:
        # No context-length match in this tier — escalate to tier 1 (most capable)
        if tier > 1:
            return get_optimal_model(
                tier - 1,
                min_context,
                tiers,
                required_capabilities,
                list(excluded_id_set),
                list(excluded_prefix_tuple),
            )
        return "openai/gpt-4o"  # Ultimate fallback

    # Check preferred list first among candidates in this tier
    candidate_ids = {m["id"] for m in candidates}
    for preferred_id in MODEL_PREFERRED:
        if preferred_id in candidate_ids:
            return preferred_id

    # Fall back to largest context model in tier
    candidates.sort(key=lambda x: x.get("context_length", 0), reverse=True)
    return candidates[0]["id"]
