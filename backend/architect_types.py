from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AgentNode(BaseModel):
    node_id: str = Field(..., description="Unique ID for the node, e.g., 'researcher_1'")
    agent_type: str = Field(..., description="Role of the agent: 'researcher', 'coder', 'analyst', 'qa'")
    model_id: str = Field(..., description="Selected model ID based on budget/tiering")
    prompt: str = Field(..., description="Specific instructions for this agent node")
    dependencies: List[str] = Field(default_factory=list, description="List of node_ids this agent depends on")
    max_tokens: int = Field(default=2000, description="Token limit for this specific agent call")

class SwarmTopology(BaseModel):
    planned_nodes: List[AgentNode] = Field(..., description="Execution sequence of agents in the DAG")
    total_estimated_cost: float = Field(..., description="Architect's pre-execution cost calculation")
    workflow_summary: str = Field(..., description="Brief summary of the parallel execution plan")

class SwarmResult(BaseModel):
    node_id: str
    content: str
    cost: float
    status: str = "success"

class SwarmExecutionReport(BaseModel):
    results: List[SwarmResult]
    final_cost: float
    complete: bool = True
