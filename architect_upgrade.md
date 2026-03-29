Here is the holistic, 50-step implementation plan designed specifically to be fed directly into your AI coding agent (e.g., DeepThink, Cursor, Claude, Devin). It provides the exact structural boundaries, mathematical guardrails, and sequence of operations needed to safely vibecode the "Architect" DAG pattern without breaking your existing WebSocket infrastructure.

Copy and paste this entire response directly to your AI agent alongside the PRD.

🚀 THE ARCHITECT PATTERN: 50-Step Implementation Blueprint
Phase 1: CSV Caching & Model Tiering Engine (backend/models_tiering.py)
Objective: Parse OpenRouter models into a lightweight memory cache to avoid token bloat during the Architect's prompt injection.

Create a new utility file backend/models_tiering.py.

Define a Pydantic model ModelPricing to store prompt (P 
in
​
 ) and completion (P 
out
​
 ) costs. Normalize all pricing strictly to "Cost per 1 Token" (float) by dividing 1M token prices by 1,000,000.

Define a Pydantic model TieredModel with fields id, name, pricing, and context_length.

Create load_and_tier_models() to parse fallback_models.json (or the existing OpenRouter API proxy in main.py).

Implement Tier 1 filtering: Flagship reasoning models (e.g., Claude 3.5 Sonnet, GPT-4o) based on high completion cost (>$5.00/1M).

Implement Tier 2 filtering: Balanced models (e.g., Gemini 2.5 Flash, Claude 3 Haiku) based on medium completion cost ($0.50−$5.00/1M).

Implement Tier 3 filtering: Fast/Cheap routing models (e.g., Llama 3 8B) based on low completion cost (<$0.50/1M).

Filter the output to ONLY include the top 3-4 models per tier. A massive list will bloat the Architect's context window.

In backend/main.py, execute this utility during the FastAPI startup lifecycle so the Tiered_Model_Map dict is globally available in memory.

Phase 2: Pydantic Schemas & Mathematical Proofs (backend/architect_types.py)
Objective: Establish strict data typing and the Cost Bounding Formula.
10. Create backend/architect_types.py. Import BaseModel and List from Pydantic.
11. Define class AgentStep(BaseModel) with attributes: step_id: str, role: str, model_id: str, max_tokens: int, depends_on: List[str], and step_cost_usd: float.
12. Define class WorkflowPlan(BaseModel) with attributes: plan_rationale: str, estimated_total_cost_usd: float, and steps: List[AgentStep].
13. Create a server-side math utility: verify_workflow_cost(plan: WorkflowPlan, base_prompt_text: str, model_map: dict) -> float.
14. Inside the math utility, estimate T_base_prompt using a standard heuristic (len(prompt_text) // 4).
15. Implement the T 
upstream_out
​
  loop: For each step, calculate the upstream context by summing the max_tokens of all step IDs listed in its depends_on array.
16. Calculate the worst-case input context size: T 
in_max
​
 =T 
base_prompt
​
 +T 
upstream_out
​
 .
17. Calculate the Step Cost: C 
s
​
 =(T 
in_max
​
 ×P 
in
​
 )+(max_tokens×P 
out
​
 ).
18. Sum all step costs to calculate C 
total
​
 .
19. If C 
total
​
 >user_budget_usd, raise a custom BudgetInsufficientError.

Phase 3: The Architect Agent Logic (backend/architect.py)
Objective: Build the control node that plans the swarm execution topology.
20. Create backend/architect.py and instantiate an ArchitectAgent class.
21. Draft the SYSTEM_PROMPT. You MUST explicitly write the mathematical formula (C 
s
​
 =(T 
in_max
​
 ×P 
in
​
 )+(T 
out_limit
​
 ×P 
out
​
 )) directly in the prompt so the LLM understands how token mappings equal monetary cost.
22. Instruct the Architect: "Assign exactly one model_id to each role. Route trivial formatting/QA to Tier 3/2. Route heavy coding to Tier 1."
23. Instruct the Architect: "Define the execution topology using the depends_on arrays to establish a strict Directed Acyclic Graph (DAG)."
24. Instruct the Architect: "Balance max_tokens (T 
out_limit
​
 ) carefully. If the budget mathematically cannot cover the required tasks, output a single step with the role 'Error' rather than a broken workflow."
25. Create the async generation method: plan_workflow(user_prompt: str, budget: float) -> WorkflowPlan.
26. Construct the payload, injecting the stringified Tiered_Model_Map and the user's explicit budget into the prompt.
27. Call the OpenRouter API with a top-tier model (e.g., Claude 3.5 Sonnet) enforcing is_json=True.
28. Implement strict regex markdown stripping (removing ````jsonwrappers) on the response to prevent Pydantic decode failures. 29. Parse the cleaned JSON usingWorkflowPlan.model_validate_json(). 30. Pass the outputted JSON into the Python verify_workflow_cost()` function. If the Architect hallucinated the math and went over budget, trigger a 2-retry loop feeding the error back to the LLM to choose cheaper models.

Phase 4: Dynamic DAG Executor Engine (backend/dag_executor.py)
Objective: Replace hardcoded linear pipelines with an async topological graph resolver.
31. Create backend/dag_executor.py containing an AsyncDAGExecutor class.
32. Write a lightweight topological sorting algorithm to validate the Architect's graph. If a cycle is detected, throw an error.
33. Create a state dictionary results_registry = {} to store the raw text outputs of completed steps.
34. Create a dictionary of asyncio.Event() objects keyed by step_id. This allows parallel tasks to natively await their specific upstream dependencies without blocking independent parallel tasks.
35. Create the worker subroutine async def _execute_node(step: AgentStep, base_prompt: str).
36. Inside _execute_node, loop through step.depends_on and await events[dependency_id].wait().
37. Dynamically assemble the LLM context: Concatenate the base_prompt WITH the text from results_registry ONLY for the steps explicitly listed in depends_on.
38. Execute the API call for the specific node. CRITICAL: Strictly enforce the Architect's budget by passing max_tokens=step.max_tokens and model=step.model_id into the API payload.
39. Store the node's result in results_registry[step.step_id] and call events[step.step_id].set() to unblock downstream nodes.
40. Create the main run_dag(plan: WorkflowPlan) function. Use asyncio.gather(*[_execute_node(...) for step in plan.steps]) to launch all nodes simultaneously. (The asyncio.Event locks will naturally pause and sequence them).

Phase 5: Pipeline & WebSocket Integration (backend/main.py)
Objective: Wire the new engine into the existing Flowmind WebSocket ecosystem.
41. In backend/main.py, locate the swarm_message WebSocket command. Add extraction for the user_budget_usd payload parameter (defaulting to $0.50).
42. Refactor execute_live_swarm(). Rip out the old classify_intent heuristic and delete the static run_enterprise_loop, run_sniper_loop, and run_newsroom_loop functions.
43. Call ArchitectAgent.plan_workflow(message, budget).
44. Save the Architect's JSON plan to the _swarm_artifacts folder as 0_architect_plan.json for auditing.
45. Broadcast the Architect's plan to the UI via a chat event (formatted nicely in Markdown) so the user sees the math proof and assigned roles.
46. Emit a new event {"event": "architect_plan_ready", "plan": plan.model_dump()} to pass the DAG structure to the frontend.
47. Feed the plan into AsyncDAGExecutor.run_dag().
48. Inside _execute_node, broadcast dynamic start events via WebSockets: {"event": "station_update", "station": step.step_id, "status": "active"} and "status": "complete" when done.
49. Once the DAG finishes, filter results_registry for terminal nodes (nodes with no dependents) and save their outputs via FileSystemManager.write_file().

Phase 6: Dynamic Frontend Refactoring (frontend/src/App.tsx & FlowDashboard.tsx)
Objective: Make the React UI render any graph the Architect dreams up dynamically.
50. Vibecode the UI:
* Add a sleek numeric input field next to the Chat bar for "Budget ($)", bound to a userBudget state. Pass this in the WS swarm_message payload.
* Rip out the hardcoded origin, specFactory, planner, executor keys from NodeState. Refactor to support a dynamic Record<string, NodeStatus>.
* In FlowDashboard.tsx, listen for architect_plan_ready. Dynamically map the <ReactFlow> nodes array (using step_id as ID and role as Label) and the edges array (mapping depends_on).
* Implement a basic X/Y multiplier based on dependency depth (e.g., Generation 0 at X=50, Generation 1 at X=300) so the graph renders left-to-right cleanly.

🛠️ ACCEPTANCE CRITERIA SNIPPETS (For the AI Agent)
1. The CSV Tiering Python Logic:

Python
def load_and_tier_models(raw_models_list: list):
    tiers = {"1": [], "2": [], "3": []}
    for model in raw_models_list:
        # Normalize to cost per 1 token
        p_in = float(model.get("pricing", {}).get("prompt", 0))
        p_out = float(model.get("pricing", {}).get("completion", 0))
        
        if p_out > 0.000005:  # > $5 per 1M
            tiers["1"].append({"id": model["id"], "p_in": p_in, "p_out": p_out})
        elif p_out > 0.0000005: # > $0.50 per 1M
            tiers["2"].append({"id": model["id"], "p_in": p_in, "p_out": p_out})
        else:
            tiers["3"].append({"id": model["id"], "p_in": p_in, "p_out": p_out})
    
    # Return top 3 from each to save prompt context
    return {"1": tiers["1"][:3], "2": tiers["2"][:3], "3": tiers["3"][:3]}
2. The DAG Executor Dependency Resolution:

Python
async def _execute_node(self, step: AgentStep, initial_prompt: str, events: dict, results_registry: dict):
    # 1. Resolve Dependencies (Wait for upstream to finish)
    await asyncio.gather(*[events[dep].wait() for dep in step.depends_on])
    
    # 2. Assemble Context (Only from direct dependencies)
    upstream_context = "\n".join([results_registry[dep] for dep in step.depends_on])
    user_prompt = f"ORIGINAL PROMPT:\n{initial_prompt}\n\nUPSTREAM DATA:\n{upstream_context}"
    
    # 3. Execute with STRICT constraints
    output, usage = await self.llm.generate(
        system_prompt=f"You are: {step.role}",
        user_prompt=user_prompt,
        model_name=step.model_id,
        max_tokens=step.max_tokens # ENFORCES THE BUDGET LIMIT!
    )
    
    # 4. Save and Unblock Downstream
    results_registry[step.step_id] = output
    events[step.step_id].set()
3. The Exact Architect System Prompt Template:

Python
ARCHITECT_SYSTEM_PROMPT = """
You are "The Architect", the control node for a dynamic Directed Acyclic Graph (DAG) AI Swarm.
Your job is to read a User Request, analyze a User Budget, and generate an execution workflow.

AVAILABLE MODELS & PRICING (Cost per 1 Token):
{tiered_model_map}

THE MATHEMATICS (Cost Bounding Formula):
You must mathematically prove the budget. For each step (s), calculate the Maximum Worst-Case Cost (C_s):
1. T_in_max = {base_prompt_tokens} + Sum of (max_tokens of ALL upstream steps listed in depends_on)
2. C_s = (T_in_max * P_in) + (max_tokens * P_out)
3. C_total = Sum of all C_s.

CRITICAL DIRECTIVES:
1. C_total MUST BE <= ${user_budget_usd}.
2. Route trivial tasks (routing, UI components) to Tier 3/2 models. Route heavy architecture to Tier 1.
3. Determine the topological execution order via the `depends_on` array (DAG structure).
4. If the budget is mathematically impossible for the required tasks, DO NOT plan a broken workflow. Output a single step named 'error' explaining the budget deficit.

OUTPUT SCHEMA (STRICT JSON):
{{
    "plan_rationale": "Explanation of topology and math",
    "estimated_total_cost_usd": 0.00,
    "steps": [
        {{
            "step_id": "step_1",
            "role": "Database Planner",
            "model_id": "provider/model-name",
            "max_tokens": 2000,
            "depends_on": [],
            "step_cost_usd": 0.05
        }}
    ]
}}
"""