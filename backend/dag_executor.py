import asyncio
import os
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
        self.approval_events: Dict[str, asyncio.Event] = {
            node.node_id: asyncio.Event() for node in topology.planned_nodes if node.approval_required
        }
        self.total_cost = 0.0

    async def execute_node(self, node: AgentNode, api_key: str):
        """Executes a single node after its dependencies are satisfied."""
        
        # 1. Wait for dependencies
        if node.dependencies:
            logger.info(f"Node {node.node_id} waiting for dependencies: {node.dependencies}")
            await asyncio.gather(*(self.completion_events[dep].wait() for dep in node.dependencies))
        
        # 2. Handle Human-in-the-loop Approval
        if node.approval_required:
            logger.info(f"Node {node.node_id} waiting for manual approval.")
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "waiting_approval",
                "message": f"Agent {node.node_id} requires permission to proceed."
            })
            await self.approval_events[node.node_id].wait()
            logger.info(f"Node {node.node_id} approved.")

        # 3. Gather dependency context
        dep_context = ""
        for dep in node.dependencies:
            res = self.results.get(dep)
            if res:
                dep_context += f"Result from {dep}:\n{res.content}\n\n"

        # 4. Notify frontend - STARTING
        await self.websocket_callback({
            "type": "NODE_STATUS",
            "node_id": node.node_id,
            "status": "executing",
            "agent_type": node.agent_type
        })

        # FIX 2: Error handler nodes short-circuit — never call the LLM
        # The Architect's error message IS the final answer. No LLM needed.
        if node.agent_type == "error_handler":
            logger.warning(f"Node {node.node_id} is an error_handler. Short-circuiting LLM call.")
            self.results[node.node_id] = SwarmResult(
                node_id=node.node_id,
                content=f"⛔ Budget Error: {node.prompt}",
                cost=0.0,
                status="failed",
                error_log="Budget constraint detected by Architect."
            )
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "failed",
                "content": f"⛔ Budget Error: {node.prompt}"
            })
            self.completion_events[node.node_id].set()
            return

        client = openai.AsyncOpenAI(api_key=api_key or os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        
        # Build search-aware system prompt
        if node.requires_search:
            system_prompt = (
                "You are a live research specialist. Real-time web search results will be injected into your context "
                "automatically before you respond. Use those search results to answer accurately with current data. "
                "Always cite specific facts from the search results (e.g., temperatures, precipitation %)."
            )
        elif node.requires_tools:
            system_prompt = (
                "You are an expert AI assistant with access to tools and APIs. "
                "Use your available tools to fulfill the task accurately."
            )
        else:
            system_prompt = "You are a specialist AI agent. Complete the assigned task thoroughly and accurately."
        
        # 5. Actual LLM Execution
        try:
            full_prompt = f"Previous Context:\n{dep_context}\n\nTask:\n{node.prompt}" if dep_context.strip() else node.prompt
            
            # Build extra_body: inject OpenRouter Web Search Plugin for search nodes.
            # This middleware fetches live web results and prepends them to the LLM context
            # BEFORE the model responds — works with ANY model, no native search support needed.
            extra_body = {}
            if node.requires_search:
                extra_body["plugins"] = [{"id": "web", "max_results": 5}]
                logger.info(f"Node {node.node_id}: OpenRouter Web Search Plugin enabled (model={node.model_id})")
            
            response = await client.chat.completions.create(
                model=node.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=node.max_tokens,
                extra_body=extra_body or None
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Precise cost tracking using model info
            from backend.models_tiering import get_model_info
            m_info = get_model_info(node.model_id)
            node_cost = 0.0
            if m_info and usage:
                p_in = float(m_info.get("pricing", {}).get("prompt", 0))
                p_out = float(m_info.get("pricing", {}).get("completion", 0))
                node_cost = (p_in * usage.prompt_tokens) + (p_out * usage.completion_tokens)
            
            self.total_cost += node_cost
            
            # Emit live cost update
            await self.websocket_callback({
                "type": "COST_UPDATE",
                "total_cost": self.total_cost,
                "node_cost": node_cost,
                "node_id": node.node_id
            })

            result = SwarmResult(
                node_id=node.node_id,
                content=content,
                cost=node_cost,
                status="success"
            )
            self.results[node.node_id] = result
            
            # 6. Notify frontend - COMPLETED
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "completed",
                "content": content
            })

        except Exception as e:
            logger.error(f"Node {node.node_id} failed: {str(e)}")
            await self.websocket_callback({
                "type": "NODE_STATUS",
                "node_id": node.node_id,
                "status": "failed",
                "error": str(e)
            })
            self.results[node.node_id] = SwarmResult(
                node_id=node.node_id, 
                content=f"ERROR: {str(e)}", 
                cost=0.0, 
                status="failed",
                error_log=str(e)
            )

        finally:
            self.completion_events[node.node_id].set()

    def approve_node(self, node_id: str):
        """Called externally from main.py WebSocket handler."""
        if node_id in self.approval_events:
            self.approval_events[node_id].set()

    async def run(self, api_key: str) -> SwarmExecutionReport:
        """Runs the entire DAG in parallel."""
        tasks = [asyncio.create_task(self.execute_node(node, api_key)) for node in self.topology.planned_nodes]
        await asyncio.gather(*tasks)
        
        report = SwarmExecutionReport(
            results=list(self.results.values()),
            final_cost=self.total_cost,
            complete=True
        )
        return report
