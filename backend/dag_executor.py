import asyncio
import uuid
import logging
from typing import List, Dict, Any, Callable, Coroutine
from backend.architect_types import AgentNode, SwarmTopology, SwarmResult, SwarmExecutionReport
import openai

logger = logging.getLogger(__name__)

class AsyncDAGExecutor:
    def __init__(self, topology: SwarmTopology, websocket_callback: Callable[[Dict[str, Any]], Coroutine]):
        self.topology = topology
        self.websocket_callback = websocket_callback
        self.results: Dict[str, SwarmResult] = {}
        self.completion_events: Dict[str, asyncio.Event] = {
            node.node_id: asyncio.Event() for node in topology.planned_nodes
        }
        self.total_cost = 0.0

    async def execute_node(self, node: AgentNode, api_key: str):
        """Executes a single node after its dependencies are satisfied."""
        
        # 1. Wait for dependencies
        if node.dependencies:
            logger.info(f"Node {node.node_id} waiting for dependencies: {node.dependencies}")
            await asyncio.gather(*(self.completion_events[dep].wait() for dep in node.dependencies))
        
        # 2. Gather dependency context
        dep_context = ""
        for dep in node.dependencies:
            res = self.results.get(dep)
            if res:
                dep_context += f"Result from {dep}:\n{res.content}\n\n"

        # 3. Notify frontend - STARTING
        await self.websocket_callback({
            "type": "NODE_STATUS",
            "node_id": node.node_id,
            "status": "executing",
            "agent_type": node.agent_type
        })

        client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        
        # 4. Actual LLM Execution
        try:
            full_prompt = f"Previous Context:\n{dep_context}\n\nTask:\n{node.prompt}"
            
            response = await client.chat.completions.create(
                model=node.model_id,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=node.max_tokens
            )
            
            content = response.choices[0].message.content
            # OpenRouter cost headers (usage info)
            usage = response.usage
            # Use estimated pricing if real cost headers aren't available
            # For now, let's keep it simple.
            
            # Simple cost tracking based on known model info
            # In production, we'd use response.usage to be precise.
            cost = 0.0 # Placeholder, logic in main.py will track this
            
            result = SwarmResult(
                node_id=node.node_id,
                content=content,
                cost=cost,
                status="success"
            )
            self.results[node.node_id] = result
            
            # 5. Notify frontend - COMPLETED
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "completed",
                "content": content
            })

            logger.info(f"Node {node.node_id} completed successfully.")

        except Exception as e:
            logger.error(f"Node {node.node_id} failed: {str(e)}")
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "failed",
                "error": str(e)
            })
            # Even if it fails, signal completion so DAG doesn't hang
            self.results[node.node_id] = SwarmResult(node_id=node.node_id, content=f"ERROR: {str(e)}", cost=0.0, status="failed")

        finally:
            self.completion_events[node.node_id].set()

    async def run(self, api_key: str) -> SwarmExecutionReport:
        """Runs the entire DAG in parallel."""
        # Wrap each node execution in a task
        tasks = [asyncio.create_task(self.execute_node(node, api_key)) for node in self.topology.planned_nodes]
        
        await asyncio.gather(*tasks)
        
        report = SwarmExecutionReport(
            results=list(self.results.values()),
            final_cost=sum(r.cost for r in self.results.values()),
            complete=True
        )
        return report
