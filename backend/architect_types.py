from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AgentNode(BaseModel):
    node_id: str = Field(..., description="Unique ID for the node, e.g., 'researcher_1'")
    agent_type: str = Field(..., description="Role of the agent: 'researcher', 'coder', 'analyst', 'qa'")
    model_id: str = Field(..., description="Selected model ID based on budget/tiering")
    prompt: str = Field(..., description="Specific instructions for this agent node")
    dependencies: List[str] = Field(default_factory=list, description="List of node_ids this agent depends on")
    max_tokens: int = Field(default=2000, description="Token limit for this specific agent call")
    approval_required: bool = Field(default=False, description="Pause execution until human approves")
    critical: bool = Field(default=True, description="If True, failure triggers re-planning/repair")
    requires_tools: bool = Field(default=False, description="Set to True if the agent must have tool-calling capability")
    requires_search: bool = Field(default=False, description="Set to True if the agent must have real-time web search capability")

class SwarmTopology(BaseModel):
    planned_nodes: List[AgentNode] = Field(..., description="Execution sequence of agents in the DAG")
    total_estimated_cost: float = Field(..., description="Architect's pre-execution cost calculation")
    workflow_summary: str = Field(..., description="Brief summary of the parallel execution plan")

class SwarmResult(BaseModel):
    node_id: str
    content: str
    cost: float
    status: str = "success"
    error_log: Optional[str] = None

class SwarmExecutionReport(BaseModel):
    results: List[SwarmResult]
    final_cost: float
    complete: bool = True
