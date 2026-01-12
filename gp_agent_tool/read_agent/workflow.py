"""
Read Agent 工作流（独立版本）
"""

from langgraph.graph import END, StateGraph

from logging_utils import get_logger
from .nodes import plan_node, execute_plan_node, read_node, answer_node
from .state import ReadAgentState


logger = get_logger(__name__)


def _route_after_execute(state: ReadAgentState) -> str:
    """路由函数：根据 next_route 决定下一步"""
    next_route = state.get("next_route", "")
    return next_route or "read"


def build_read_agent_workflow() -> StateGraph:
    """构建 Read Agent 工作流"""
    workflow = StateGraph(ReadAgentState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_plan", execute_plan_node)
    workflow.add_node("read", read_node)
    workflow.add_node("answer", answer_node)

    workflow.set_entry_point("plan")

    workflow.add_edge("plan", "execute_plan")

    workflow.add_conditional_edges(
        "execute_plan",
        _route_after_execute,
        {
            "read": "read",
            "answer": "answer",
        },
    )

    workflow.add_edge("read", "execute_plan")
    workflow.add_edge("answer", END)

    return workflow


def compile_read_agent_workflow():
    """编译 Read Agent 工作流"""
    wf = build_read_agent_workflow()
    return wf.compile()


