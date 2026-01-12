"""
Read Agent 状态定义（独立于 textMSA 项目）。
"""

from __future__ import annotations

from typing import Optional, TypedDict

try:
    from typing import NotRequired  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from typing_extensions import NotRequired

from logging_utils import get_logger


logger = get_logger(__name__)


class FileInfo(TypedDict, total=False):
    """
    外部传入的文件信息。

    注意：字段严格按照用户约定，不新增额外字段。
    """

    file_name: str
    file_path: str
    description: str
    preview: str


class PlanHistory(TypedDict, total=False):
    """计划历史记录"""

    file_name: str  # 文件名
    file_path: str  # 文件路径
    plan_detail: str  # 计划详情
    result: Optional[str]  # 执行结果
    order_reasoning: NotRequired[str]  # 顺序理由


class ReadAgentState(TypedDict, total=False):
    """Read Agent 的状态（简化版）"""

    # 用户查询
    user_query: str
    # 文件列表（外部传入的 file info）
    files: list[FileInfo]
    # 文件概览字符串（外部已经格式化好）
    file_overview: str
    # 语言
    language: NotRequired[str]
    # 历史计划
    history_plans: list[PlanHistory]
    # 当前计划索引
    current_plan_index: int
    # 最终答案
    final_answer: NotRequired[Optional[str]]
    # 下一步路由
    next_route: NotRequired[str]
    # 用户/项目 ID（可选）
    user_id: NotRequired[str]
    project_id: NotRequired[str]


def build_initial_state(
    user_query: str,
    files: list[FileInfo],
    file_overview: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    language: str = "zh",
) -> ReadAgentState:
    """构建初始状态"""
    state: ReadAgentState = {
        "user_query": user_query,
        "files": files,
        "file_overview": file_overview,
        "language": language,
        "history_plans": [],
        "current_plan_index": 0,
    }

    if user_id:
        state["user_id"] = user_id
    if project_id:
        state["project_id"] = project_id

    logger.info(
        "Read Agent initial state ready",
        extra={
            "files_len": len(files),
            "user_id": user_id,
            "project_id": project_id,
            "language": language,
        },
    )
    return state


