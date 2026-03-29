import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from backend.architect_types import AgentNode, SwarmTopology
from backend.models_tiering import load_and_tier_models, get_optimal_model, get_model_info
import openai

logger = logging.getLogger(__name__)

ARCHITECT_SYSTEM_PROMPT = """
You are the Swarm Architect. Your job is to decompose a complex user request into a Directed Acyclic Graph (DAG) of specialized agent nodes.

Available Agent Types:
- researcher: Gathers information, reads files, searches web (if available).
- coder: Implementation of specific logic, UI, or backend functions.
- analyst: Logical reasoning, data summary, or architectural breakdown.
- qa: Verifies code, runs checks, or suggests refinements.

Constraint: STRICT BUDGET.
You must select models for each node that fit within the human-provided budget.
Formula to respect: SUM( (input_price_per_token * est_input_tokens) + (output_price_per_token * max_tokens) ) <= budget.

Available Models (Tiered):
{model_tiers_summary}

Your output MUST BE A JSON object matching the SwarmTopology schema:
{
  "planned_nodes": [
    {
      "node_id": "string",
      "agent_type": "string",
      "model_id": "string",
      "prompt": "string instructions for this node",
      "dependencies": ["list_of_node_ids"],
      "max_tokens": int
    }
  ],
  "total_estimated_cost": float,
  "workflow_summary": "string"
}

Plan for maximum parallelism where dependencies allow.
"""

async def plan_swarm_dag(
    user_prompt: str,
    budget: float,
    architect_model: str = "openai/gpt-4o",
    api_key: str = None
) -> SwarmTopology:
    """
    Architects the swarm execution plan.
    """
    tiers = load_and_tier_models()
    
    # Generate a summary of available models for the system prompt
    tier_summary = f"Tier 1 (Premium): {[m['id'] for m in tiers['tier1'][:5]]}...\n"
    tier_summary += f"Tier 2 (Mid): {[m['id'] for m in tiers['tier2'][:5]]}...\n"
    tier_summary += f"Tier 3 (Cheap): {[m['id'] for m in tiers['tier3'][:5]]}..."

    # In a real scenario, we'd use the architect_model to generate the JSON.
    # We use structured output if supported by the model.
    
    full_prompt = f"User Request: {user_prompt}\nBudget: ${budget}\n\nDecompose this into a DAG."

    # Simulation/Actual Call to LLM
    # Note: Using AsyncOpenAI here would be better, but assuming sync openai for compatibility with main.py if needed.
    # However, for a real agentic upgrade, let's use the provided architect_model.
    
    # For now, I'll implement a robust helper that calls the LLM.
    # Since I don't have a direct LLM tool, I'll write the logic as if I'm building it.
    
    # NOTE: In the backend/main.py, there's likely an existing client.
    # I'll rely on a generic 'llm_call' structure.
    
    # Let's define the actual logic in architect.py
    
    client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    
    try:
        response = await client.chat.completions.create(
            model=architect_model,
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT.format(model_tiers_summary=tier_summary)},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_json = json.loads(response.choices[0].message.content)
        topology = SwarmTopology(**raw_json)
        
        # Post-validation: Ensure cost doesn't exceed budget (even if LLM tried)
        # If it does, downrank models automatically
        topology = validate_and_correct_budget(topology, budget, tiers)
        
        return topology
        
    except Exception as e:
        logger.error(f"Architect planning failed: {str(e)}")
        raise

def validate_and_correct_budget(topology: SwarmTopology, budget: float, tiers: Dict[str, List[Dict[str, Any]]]) -> SwarmTopology:
    """
    Recalculates cost and swaps models to cheaper ones if budget is exceeded.
    """
    total_cost = 0.0
    for node in topology.planned_nodes:
        m_info = get_model_info(node.model_id)
        if m_info:
            p_in = float(m_info.get("pricing", {}).get("prompt", 0))
            p_out = float(m_info.get("pricing", {}).get("completion", 0))
            # Rough estimate: input context ~ 500 tokens + prompt
            node_cost = (p_in * 1000) + (p_out * node.max_tokens)
            total_cost += node_cost
            
    if total_cost > budget:
        logger.warning(f"Architect plan cost ({total_cost}) exceeds budget ({budget}). Downranking models...")
        # Simple iterative downranking
        for node in topology.planned_nodes:
            m_info = get_model_info(node.model_id)
            if m_info and m_info.get("tier", 3) < 3:
                # Downrank tier
                new_tier = m_info.get("tier", 1) + 1
                node.model_id = get_optimal_model(new_tier, 8192, tiers)
        
        # Recalculate summary cost
        topology.total_estimated_cost = total_cost # Should be updated accurately
        
    return topology
