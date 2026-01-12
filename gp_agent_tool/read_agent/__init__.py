from typing import Optional

from .state import FileInfo, build_initial_state
from .workflow import compile_read_agent_workflow


def run_read_agent(
    user_query: str,
    files: list[FileInfo],
    file_overview: str,
    language: str = "zh",
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """
    运行 Read Agent，返回最终状态。

    参数:
        user_query: 用户问题或任务描述。
        files: 外部传入的文件信息列表，元素为 FileInfo:
            {
              "file_name": str,
              "file_path": str,
              "description": str,
              "preview": str,
            }
        file_overview: 已经由外部构建好的文件概览字符串，用于规划 Prompt。
        language: 语言代码，默认 "zh"。
        user_id: 可选用户 ID。
        project_id: 可选项目 ID。
    返回:
        最终状态字典，至少包含 "final_answer" 字段。
    """
    initial_state = build_initial_state(
        user_query=user_query,
        files=files,
        file_overview=file_overview,
        user_id=user_id,
        project_id=project_id,
        language=language,
    )
    app = compile_read_agent_workflow()
    final_state = app.invoke(initial_state)
    return final_state


