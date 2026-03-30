import json
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from backend.architect_types import AgentNode, SwarmTopology
from backend.models_tiering import load_and_tier_models, get_optimal_model, get_model_info
import openai

logger = logging.getLogger(__name__)

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
3. Route trivial tasks to Tier 3/2. Route heavy coding/analyst to Tier 1.
4. SEARCH NODES: If `requires_search` is True, the model executor will AUTOMATICALLY provide a search-capable system prompt. However, YOU must write the `prompt` field to start with an explicit action: e.g. "Search the web for the current weather in Tokyo and report..." — do NOT write vague prompts like "find weather data".
5. If a task requires Search but fits no search-capable model within budget, prefer a cheap search-capable model over a non-searching one, OR output an 'error' node with a BUDGET_CONSTRAINT warning.
6. If the budget is mathematically impossible for the required tasks, output a single step with agent_type="error_handler" explaining the budget deficit. Keep it under 30 words.

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
    previous_results: Optional[List[Any]] = None
) -> SwarmTopology:
    """
    Architects or Repairs the swarm execution plan.
    """
    tiers = load_and_tier_models()
    
    def _fmt_model(m):
        p_in = float(m.get('pricing', {}).get('prompt', 0)) * 1_000_000
        p_out = float(m.get('pricing', {}).get('completion', 0)) * 1_000_000
        return f"{m['id']} (${p_in:.3f}/M in, ${p_out:.3f}/M out)"
    
    tier_summary = f"Tier 1 (Premium, >$5/M): {[_fmt_model(m) for m in tiers['tier1'][:3]]}\n"
    tier_summary += f"Tier 2 (Mid, >$0.50/M): {[_fmt_model(m) for m in tiers['tier2'][:3]]}\n"
    tier_summary += f"Tier 3 (Cheap, <=$0.50/M): {[_fmt_model(m) for m in tiers['tier3'][:5]]}"

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
        
        topology = validate_and_correct_budget(topology, budget, tiers)
        # Always overwrite the LLM's estimate with our authoritative recalculated value
        logger.info(f"Architect estimated cost: ${topology.total_estimated_cost:.6f} (LLM self-report overwritten by validator)")
        return topology
        
    except Exception as e:
        logger.error(f"Architect planning failed: {str(e)}")
        raise

def validate_and_correct_budget(topology: SwarmTopology, budget: float, tiers: Dict[str, List[Dict[str, Any]]]) -> SwarmTopology:
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
                    new_model = get_optimal_model(next_tier, 8192, tiers, required_capabilities=caps)
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
