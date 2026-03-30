import json
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional, Set
from backend.architect_types import AgentNode, SwarmTopology
from backend.models_tiering import load_and_tier_models, get_optimal_model, get_model_info
import openai

logger = logging.getLogger(__name__)
SEARCH_UNSAFE_MODEL_PREFIXES = ("reka/",)

ARCHITECT_SYSTEM_PROMPT = """
You are "The Architect", the control node for a dynamic Directed Acyclic Graph (DAG) AI Swarm.
Your job is to read a User Request, analyze a User Budget, and generate an execution workflow.

AVAILABLE MODELS & PRICING (Cost per 1,000,000 tokens — i.e. $ per million tokens):
{model_tiers_summary}

THE MATHEMATICS (Realistic Cost Bounding):
IMPORTANT: Pricing is in $/million tokens. Convert to per-token: P_per_token = P_per_million / 1,000,000.
1. T_in_est = 500 (Base Prompt) + Sum of (0.4 * max_tokens of ALL direct upstream dependencies)
2. C_s = (T_in_est * P_in / 1,000,000) + (max_tokens * P_out / 1,000,000)
3. C_total = Sum of all C_s.

For reference: A Tier 3 model at ~$0.10/M tokens running 500 tokens in + 500 tokens out ≈ $0.0001.
Typical single-node research task with 500 in + 1000 out costs $0.0001–$0.001.

CRITICAL DIRECTIVES:
1. C_total MUST BE <= ${{user_budget_usd}}.
2. CAPABILITIES: Set `requires_tools: True` if the node needs tool use. Set `requires_search: True` if the node needs real-time research (weather, news, prices, live data).
3. Route trivial tasks to Tier 3/2. Route heavy coding to Tier 1. Analyst and researcher nodes default to Tier 3. Simple scripts, calculations, and one-off data transforms should default coder nodes to Tier 3 unless the user explicitly asks for robustness, production quality, security, or advanced error handling.
4. SEARCH NODES: If `requires_search` is True, the model executor will AUTOMATICALLY provide a search-capable system prompt. However, YOU must write the `prompt` field to start with an explicit action: e.g. "Search the web for the current weather in Tokyo and report..." — do NOT write vague prompts like "find weather data".
5. If a task requires Search but fits no search-capable model within budget, prefer a cheap search-capable model over a non-searching one, OR output an 'error' node with a BUDGET_CONSTRAINT warning.
6. If the budget is mathematically impossible for the required tasks, output a single step with agent_type="error_handler" explaining the budget deficit. Keep it under 30 words.

NODE SELECTION RULES:
- Only add a "coder" node if the task explicitly requires deliverable code as output.
- Only add a "qa" node if the task requires verifying or testing code.
- "reviewer" and "qa" nodes are non-critical by default unless the user explicitly asks for validation, review, or self-correction.
- For pure research/analysis tasks (reading files, doing math, generating insights), use at most: researcher + analyst.
- The final node's output IS the user's answer — do not end with a QA test suite unless tests were explicitly requested.

NODE COUNT HEURISTIC:
- Simple (single topic, no code): 1–2 nodes max.
- Moderate (multi-topic, analysis + synthesis): 2–3 nodes max.
- Complex (requires code + verification, multi-stage research): 3–5 nodes.
- Only add nodes that directly produce a user-facing deliverable or are required dependencies.

OUTPUT SCHEMA (STRICT JSON):
{{
    "workflow_summary": "Brief summary of the execution plan",
    "total_estimated_cost": 0.00,
    "planned_nodes": [
        {{
            "node_id": "researcher_1",
            "agent_type": "researcher",
            "model_id": "provider/model-name",
            "max_tokens": 2000,
            "prompt": "Search the web for [specific query]... then summarize...",
            "dependencies": [],
            "approval_required": false,
            "critical": true,
            "requires_tools": false,
            "requires_search": false
        }}
    ]
}}
"""

async def plan_swarm_dag(
    user_prompt: str,
    budget: float,
    architect_model: str = "openai/gpt-4o",
    api_key: str = None,
    previous_results: Optional[List[Any]] = None,
    banned_model_ids: Optional[Set[str]] = None,
) -> SwarmTopology:
    """
    Architects or Repairs the swarm execution plan.
    """
    tiers = load_and_tier_models()
    
    def _fmt_model(m):
        p_in = float(m.get('pricing', {}).get('prompt', 0)) * 1_000_000
        p_out = float(m.get('pricing', {}).get('completion', 0)) * 1_000_000
        return f"{m['id']} (${p_in:.3f}/M in, ${p_out:.3f}/M out)"
    
    tier_summary = f"Tier 1 (Premium, >$5/M): {[_fmt_model(m) for m in tiers['tier1']]}\n"
    tier_summary += f"Tier 2 (Mid, >$0.50/M): {[_fmt_model(m) for m in tiers['tier2']]}\n"
    tier_summary += f"Tier 3 (Cheap, <=$0.50/M): {[_fmt_model(m) for m in tiers['tier3']]}"

    mode_context = ""
    if previous_results:
        mode_context = "REPAIR MODE ACTIVE.\nPrevious Results:\n"
        for res in previous_results:
            mode_context += f"- Node {res.node_id}: {res.status}. Output: {res.content[:200]}... Error: {res.error_log}\n"
    
    full_prompt = f"{mode_context}\nUser Request: {user_prompt}\nBudget: ${budget}\n\nDecompose this into a optimal DAG. (Keyword: json)"

    client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    
    # Architectural JSON Compliance 3.0
    sys_prompt = ARCHITECT_SYSTEM_PROMPT.format(model_tiers_summary=tier_summary, user_budget_usd=budget)
    if "json" not in sys_prompt.lower():
        sys_prompt += "\n\nIMPORTANT: Response MUST be a valid JSON object. (Keyword: json)"

    try:
        response = await client.chat.completions.create(
            model=architect_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_json = json.loads(response.choices[0].message.content)
        topology = SwarmTopology(**raw_json)
        
        topology = validate_and_correct_budget(topology, budget, tiers, banned_model_ids)
        # Always overwrite the LLM's estimate with our authoritative recalculated value
        logger.info(f"Architect estimated cost: ${topology.total_estimated_cost:.6f} (LLM self-report overwritten by validator)")
        return topology
        
    except Exception as e:
        logger.error(f"Architect planning failed: {str(e)}")
        raise

def validate_and_correct_budget(
    topology: SwarmTopology,
    budget: float,
    tiers: Dict[str, List[Dict[str, Any]]],
    banned_model_ids: Optional[Set[str]] = None,
) -> SwarmTopology:
    """
    Recalculates cost and swaps models to cheaper ones if budget is exceeded,
    while maintaining required capabilities (tools/search).
    """
    def calculate_cost(planned_nodes):
        total = 0.0
        node_map = {n.node_id: n for n in planned_nodes}
        for node in planned_nodes:
            m_info = get_model_info(node.model_id)
            if m_info:
                p_in = float(m_info.get("pricing", {}).get("prompt", 0))
                p_out = float(m_info.get("pricing", {}).get("completion", 0))
                
                # More realistic estimation: 500 base + 40% of parent outputs
                t_in_est = 500
                for dep_id in node.dependencies:
                    if dep_id in node_map:
                        t_in_est += int(node_map[dep_id].max_tokens * 0.4)
                
                node_cost = (p_in * t_in_est) + (p_out * node.max_tokens)
                total += node_cost
        return total

    banned_model_ids = banned_model_ids or set()

    # Agent-type minimum tier floors: QA must use at least Tier 2.
    # Analyst, researcher, and most coder tasks default to Tier 3.
    # Coder nodes are only promoted for explicitly premium/robust implementations.
    AGENT_MIN_TIER = {
        "qa": 2,
    }

    PREMIUM_CODER_HINTS = (
        "robust",
        "production",
        "error handling",
        "hardened",
        "security",
        "postgres",
        "postgresql",
        "database",
        "deploy",
        "critical",
    )

    for node in topology.planned_nodes:
        agent_type = (node.agent_type or "").lower()
        min_tier = AGENT_MIN_TIER.get(agent_type)
        if agent_type == "coder":
            prompt_lower = (node.prompt or "").lower()
            if any(hint in prompt_lower for hint in PREMIUM_CODER_HINTS):
                min_tier = 2
        if min_tier is not None:
            current_info = get_model_info(node.model_id)
            current_tier = current_info.get("tier", 3) if current_info else 3
            if current_tier > min_tier:
                caps = []
                if node.requires_tools: caps.append("has_tools")
                if node.requires_search: caps.append("has_search")
                upgraded = get_optimal_model(
                    min_tier,
                    8192,
                    tiers,
                    required_capabilities=caps,
                    excluded_model_ids=list(banned_model_ids),
                    excluded_prefixes=list(SEARCH_UNSAFE_MODEL_PREFIXES if node.requires_search else ()),
                )
                if upgraded != node.model_id:
                    logger.info(
                        f"Tier floor upgrade for {node.node_id} ({node.agent_type}): "
                        f"{node.model_id} (tier {current_tier}) -> {upgraded} (tier {min_tier})"
                    )
                    node.model_id = upgraded

    # Agent-type maximum tier ceilings: coder, qa, and analyst should not use Tier 1
    # (premium models) for standard tasks. Keeps cost predictable without sacrificing quality.
    AGENT_MAX_TIER = {
        "coder": 2,
        "qa": 2,
        "analyst": 2,
    }

    # Enforce max-tier ceilings: downgrade Tier 1 models to Tier 2 for standard agent types.
    # A lower tier number = more expensive; ceiling means "don't go more expensive than this".
    for node in topology.planned_nodes:
        max_tier = AGENT_MAX_TIER.get((node.agent_type or "").lower())
        if max_tier is not None:
            current_info = get_model_info(node.model_id)
            current_tier = current_info.get("tier", 3) if current_info else 3
            if current_tier < max_tier:  # tier 1 < tier 2 means it's over-budget premium
                caps = []
                if node.requires_tools: caps.append("has_tools")
                if node.requires_search: caps.append("has_search")
                downgraded = get_optimal_model(
                    max_tier,
                    8192,
                    tiers,
                    required_capabilities=caps,
                    excluded_model_ids=list(banned_model_ids),
                    excluded_prefixes=list(SEARCH_UNSAFE_MODEL_PREFIXES if node.requires_search else ()),
                )
                if downgraded != node.model_id:
                    logger.info(
                        f"Tier ceiling downgrade for {node.node_id} ({node.agent_type}): "
                        f"{node.model_id} (tier {current_tier}) -> {downgraded} (tier {max_tier})"
                    )
                    node.model_id = downgraded

    # Enforce per-run model restrictions before budget optimization.
    for node in topology.planned_nodes:
        caps = []
        if node.requires_tools:
            caps.append("has_tools")
        if node.requires_search:
            caps.append("has_search")

        current_model_id = node.model_id
        current_info = get_model_info(current_model_id)
        preferred_tier = current_info.get("tier", 2) if current_info else 2

        must_replace = (
            current_model_id in banned_model_ids
            or (node.requires_search and current_model_id.startswith(SEARCH_UNSAFE_MODEL_PREFIXES))
        )

        if must_replace:
            replacement = get_optimal_model(
                preferred_tier,
                8192,
                tiers,
                required_capabilities=caps,
                excluded_model_ids=list(banned_model_ids),
                excluded_prefixes=list(SEARCH_UNSAFE_MODEL_PREFIXES if node.requires_search else ()),
            )
            if replacement != current_model_id:
                logger.warning(
                    f"Architect replaced model for node {node.node_id}: {current_model_id} -> {replacement}"
                )
                node.model_id = replacement

    total_cost = calculate_cost(topology.planned_nodes)
    
    if total_cost > budget:
        logger.warning(f"Architect plan cost ({total_cost:.4f}) exceeds budget ({budget}). Downranking nodes...")
        
        # Sort nodes by cost contribution descending to tackle expensive ones first
        expensive_first = sorted(topology.planned_nodes, key=lambda n: n.max_tokens, reverse=True)
        
        for node in expensive_first:
            m_info = get_model_info(node.model_id)
            if m_info and m_info.get("tier", 3) < 3:
                # Build required capabilities list
                caps = []
                if node.requires_tools: caps.append("has_tools")
                if node.requires_search: caps.append("has_search")
                
                # Try next tier down
                current_tier = m_info.get("tier", 1)
                for next_tier in range(current_tier + 1, 4):
                    new_model = get_optimal_model(
                        next_tier,
                        8192,
                        tiers,
                        required_capabilities=caps,
                        excluded_model_ids=list(banned_model_ids),
                        excluded_prefixes=list(SEARCH_UNSAFE_MODEL_PREFIXES if node.requires_search else ()),
                    )
                    if new_model != "openai/gpt-4o" or current_tier == 3:
                        node.model_id = new_model
                        # Re-calculate to see if we're under budget
                        if calculate_cost(topology.planned_nodes) <= budget:
                            break
                
            if calculate_cost(topology.planned_nodes) <= budget:
                break
        
        # Final update of estimated cost
        topology.total_estimated_cost = calculate_cost(topology.planned_nodes)
        
    return topology
