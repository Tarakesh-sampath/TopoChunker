import json
import re
from typing import Annotated, Dict, List, Literal, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from core.types import Blueprint, RoutingPath, RiskProfile
from .tools import get_inspector_tools
from .llm_factory import create_llm

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    file_path: str
    iteration_count: int
    observations: List[str]
    meta_info: Dict
    complexity_score: float
    rule_adherence_score: float
    thoughts: List[str]
    final_decision: Dict | None

class CustomToolNode:
    """A ToolNode that prints logs so you can see what's happening."""
    def __init__(self, tools: list):
        self.tools_map = {t.name: t for t in tools}

    def __call__(self, state: AgentState) -> Dict:
        last_message = state["messages"][-1]
        tool_outputs = []
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return {"messages": []}

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            print(f"🛠️ [工具执行] {tool_name} | 参数: {args}")
            
            tool = self.tools_map[tool_name]
            result = tool.invoke(args)
            
            print(f"📥 [工具结果] {tool_name} -> {result}")
            
            tool_outputs.append(ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_call["id"]
            ))
            
        return {"messages": tool_outputs}

class InspectorAgent:
    def __init__(
        self,
        model_type: str = "deepseek",
        model_name: str = "deepseek-chat",
        max_iterations: int = 6,
        temperature: float = 0.1,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.tools = get_inspector_tools()
        
        self.base_llm = llm or create_llm(
            model_type=model_type,
            model_name=model_name,
            temperature=temperature
        )

        self.llm = self.base_llm.bind_tools(self.tools)
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", CustomToolNode(self.tools))
        workflow.add_node("extract_decision", self._extract_decision_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue, {
            "continue": "tools",
            "finalize": "extract_decision"
        })
        workflow.add_edge("tools", "agent")
        workflow.add_edge("extract_decision", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()

    def _agent_node(self, state: AgentState) -> Dict:
        iteration = state.get("iteration_count", 0) + 1
        print(f"\n[Agent 节点] 迭代 {iteration}/{self.max_iterations}")

        messages = state.get("messages", [])
        if not messages:
            # 强化 Prompt，给死命令
            system_content = f"""You are the TopoChunker Inspector.
CRITICAL RULES:
1. You MUST call `analyze_metadata` first. DO NOT guess.
2. If `is_scanned` is True, you MUST stop and choose "VISUAL_TOPOLOGY".
3. If `is_scanned` is False, you MUST call `probe_layout_complexity` and `test_rule_adherence` before deciding.
4. DO NOT make any decision until you have the tool results.
"""
            messages = [
                SystemMessage(content=system_content), 
                HumanMessage(content=f"Analyze document: {state['file_path']}")
            ]

        response = self.llm.invoke(messages)
        return {
            "messages": [response],
            "iteration_count": iteration,
            "thoughts": [response.content] if response.content else []
        }

    def _should_continue(self, state: AgentState) -> Literal["continue", "finalize"]:
        last_msg = state["messages"][-1]
        
        # 如果有工具调用，必须继续执行
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "continue"
            
        # 只有在拿到足够数据或者达到最大次数时才结束
        if state["iteration_count"] >= self.max_iterations:
            return "finalize"
            
        # 检查是否已经拿到了关键指标
        has_metrics = False
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                has_metrics = True
                break
        
        if has_metrics:
            return "finalize"
            
        return "continue"

    def _extract_decision_node(self, state: AgentState) -> AgentState:
        # 从历史消息中提取所有工具返回的数据
        metrics = {}
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    metrics.update(data)
                except: continue

        print(f"📊 [最终决策依据] 收集到的指标: {metrics}")

        prompt = f"""Based on these real document metrics: {json.dumps(metrics)}
Select the routing path.

LOGIC:
- VISUAL_TOPOLOGY IF:
    - `is_scanned` is True
    - OR `image_count` > 0
    - OR `table_count` > 0
    - OR `complexity_score` > 0.7
- STRUCTURAL_RULE IF:
    - No images/tables (`image_count` == 0 and `table_count` == 0)
    - AND `rule_adherence_score` > 0.6
- SEMANTIC_FLOW otherwise.

Return JSON:
{{
  "selected_path": "...",
  "confidence_score": 0.9,
  "risk_score": 0.1,
  "reason": "..."
}}
"""
        response = self.base_llm.invoke([SystemMessage(content="You are a strict logic extractor."), HumanMessage(content=prompt)])
        try:
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            state["final_decision"] = json.loads(match.group()) if match else None
        except:
            state["final_decision"] = {"selected_path": "SEMANTIC_FLOW", "confidence_score": 0.5, "risk_score": 0.5, "reason": "Logic fallback"}
        
        # 同步更新指标到 state 供 Blueprint 使用
        state["meta_info"] = {"is_scanned": metrics.get("is_scanned", False)}
        state["complexity_score"] = metrics.get("complexity_score", 0.0)
        state["rule_adherence_score"] = metrics.get("rule_adherence_score", 0.0)
        
        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        return state

    def inspect(self, file_path: str) -> Blueprint:
        initial_state = {
            "messages": [],
            "file_path": file_path,
            "iteration_count": 0,
            "observations": [],
            "meta_info": {},
            "complexity_score": 0.0,
            "rule_adherence_score": 0.0,
            "thoughts": [],
            "final_decision": None
        }
        
        final_state = self.workflow.invoke(initial_state)
        d = final_state["final_decision"]
        
        return Blueprint(
            selected_path=RoutingPath(d["selected_path"]),
            confidence_score=d["confidence_score"],
            risk_assessment=RiskProfile(score=d["risk_score"], reason=d["reason"]),
            execution_plan={
                "metrics": {
                    "is_scanned": final_state["meta_info"].get("is_scanned", False),
                    "complexity": final_state["complexity_score"],
                    "rules": final_state["rule_adherence_score"]
                },
                "reasoning_chain": final_state["thoughts"]
            }
        )
