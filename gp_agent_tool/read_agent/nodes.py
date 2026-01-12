"""
Read Agent 各个节点的实现（独立版本，不依赖 textMSA）。

注意：
- 不在内部执行任何文件系统读取，所有文件信息由外部通过 FileInfo 传入。
- 对于文本文件，直接使用 preview 作为内容来源。
- 对于数据/图像文件，仍允许在生成的代码或多模态模型中使用 file_path，但本模块自身不做 I/O。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from llm_client import run_llm, run_multimodal_llm
from logging_utils import get_logger
from .python_repl_tool import PythonREPL
from .prompts import (
    format_answer_prompt,
    format_code_generation_prompt,
    format_code_retry_prompt,
    format_data_preview_analysis_prompt,
    format_plan_prompt,
    format_text_summary_prompt,
)
from .state import FileInfo, PlanHistory, ReadAgentState


logger = get_logger(__name__)


def _normalize_language(language: Optional[str]) -> str:
    if not language:
        return "en"
    lower = language.lower()
    if lower.startswith("zh"):
        return "zh"
    if lower.startswith("en"):
        return "en"
    return "en"


def _get_localized_message(messages: dict[str, str], language: Optional[str]) -> str:
    lang = _normalize_language(language)
    return messages.get(lang, messages.get("en", ""))


def _is_image_file(path_str: str) -> bool:
    path = Path(path_str)
    return path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".tiff",
        ".webp",
    }


def _is_data_file(filename: str) -> bool:
    data_extensions = {
        ".csv",
        ".h5ad",
        ".json",
        ".parquet",
        ".xlsx",
        ".xls",
        ".h5",
        ".hdf5",
        ".feather",
        ".pkl",
        ".pickle",
    }
    return Path(filename).suffix.lower() in data_extensions


def _strip_if_main_block(code: str) -> str:
    """
    修复代码中的 if __name__ == "__main__": 块
    
    在 REPL 环境中，__name__ 可能不是 "__main__"，导致代码不执行。
    此函数将 if __name__ == "__main__": 块中的内容提取出来，直接执行。
    """
    import re
    
    # 检查是否包含 if __name__ == "__main__": 模式
    pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:'
    
    if not re.search(pattern, code):
        return code  # 没有 main 块，直接返回
    
    lines = code.split('\n')
    result_lines = []
    non_main_lines = []
    main_content_lines = []
    in_main_block = False
    main_block_indent = None
    
    for line in lines:
        # 检查是否是 if __name__ == "__main__": 行
        if re.match(r'\s*if\s+__name__\s*==\s*["\']__main__["\']\s*:', line):
            in_main_block = True
            main_block_indent = len(line) - len(line.lstrip())
            continue
        
        if in_main_block:
            # 在 main 块中
            if not line.strip():  # 空行
                main_content_lines.append('')
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # 如果缩进小于等于 main_block_indent，说明 main 块结束了
            if current_indent <= main_block_indent:
                in_main_block = False
                # 这一行不属于 main 块，添加到非 main 块
                non_main_lines.append(line)
                continue
            
            # 提取 main 块中的内容，去除缩进（main_block_indent + 4）
            indent_to_remove = main_block_indent + 4
            if len(line) >= indent_to_remove:
                main_content_lines.append(line[indent_to_remove:])
            else:
                # 如果缩进不够，可能是使用了 tab 或其他缩进方式，直接去除所有前导空白
                main_content_lines.append(line.lstrip())
        else:
            # 不在 main 块中，保留原行
            non_main_lines.append(line)
    
    # 组合代码：非 main 块 + main 块内容
    fixed_code = '\n'.join(non_main_lines)
    if main_content_lines:
        if fixed_code and not fixed_code.endswith('\n'):
            fixed_code += '\n'
        fixed_code += '\n'.join(main_content_lines)
    
    if main_content_lines:
        logger.info(
            "Read Agent - 修复了 if __name__ == '__main__': 块",
            extra={
                "original_code_length": len(code),
                "fixed_code_length": len(fixed_code),
                "main_block_lines": len(main_content_lines),
            },
        )
    
    return fixed_code


def _parse_json_response(response_content: str) -> dict[str, Any]:
    """从 LLM 响应中解析 JSON，失败时返回空 dict。

    兼容以下几种常见格式：
    - 直接输出 JSON：{"plans": [...], "reasoning": "..."}
    - 使用 ```json / ``` 包裹的代码块：
        ```json
        { ... }
        ```
    """
    cleaned = response_content.strip()

    # 先尝试直接解析整段内容
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 再尝试处理被 ``` / ```json / ```python 等代码块包裹的情况
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # 去掉第一行 ``` / ```json / ```python
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # 去掉最后一行 ```
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
        cleaned_block = "\n".join(lines).strip()

        # 1）优先尝试直接把代码块内容当 JSON 解析
        try:
            return json.loads(cleaned_block)
        except Exception:
            pass

        # 2）如果是 ```python 之类的代码块，可能包含变量赋值，尝试从中抽取第一个 {...} 结构
        brace_start = cleaned_block.find("{")
        brace_end = cleaned_block.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            candidate = cleaned_block[brace_start : brace_end + 1].strip()
            try:
                return json.loads(candidate)
            except Exception:
                return {}

        return {}

    # 最后一次兜底：从任意文本中提取第一个 {...} 结构再尝试解析
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = cleaned[brace_start : brace_end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {}


def plan_node(state: ReadAgentState):
    """生成执行计划（顺序规划）"""
    user_query = state["user_query"]
    file_overview = state["file_overview"]
    files = state["files"]
    language = state.get("language", "en")

    plan_prompt = format_plan_prompt(
        user_query=user_query,
        file_overview=file_overview,
        language=language,
    )
    response_text = run_llm(
        plan_prompt,
        temperature=0.1,
        max_tokens=8000,
        node_name="plan_node",
    )
    parsed = _parse_json_response(response_text)
    plans = parsed.get("plans", [])
    reasoning = parsed.get("reasoning", "")

    if not plans:
        warning_msg = _get_localized_message(
            {
                "zh": "计划生成失败，返回空计划列表",
                "en": "Plan generation failed, returning empty plan list",
            },
            language,
        )
        logger.warning(warning_msg)
        plans = []

    # 将计划与外部传入的文件信息对齐（按 file_name / file_path 匹配）
    file_index = {(f.get("file_name"), f.get("file_path")): f for f in files}

    history_plans: list[PlanHistory] = []
    for plan in plans:
        file_name = plan.get("file_name", "")
        file_path = plan.get("file_path", "")
        key = (file_name, file_path)
        file_info: Optional[FileInfo] = file_index.get(key)  # 目前主要用于日志和一致性校验
        if not file_info:
            logger.warning(
                "Plan references file not found in input files",
                extra={"file_name": file_name, "file_path": file_path},
            )

        plan_history: PlanHistory = {
            "file_name": file_name,
            "file_path": file_path,
            "plan_detail": plan.get("plan_detail", ""),
            "result": None,
        }
        order_reasoning = plan.get("order_reasoning", "")
        if order_reasoning:
            plan_history["order_reasoning"] = order_reasoning
        history_plans.append(plan_history)

    logger.info(
        "Plan node completed",
        extra={
            "plan_count": len(history_plans),
            "reasoning_preview": reasoning[:200]
            + ("..." if len(reasoning) > 200 else ""),
        },
    )

    return {
        "current_plan_index": 0,
        "history_plans": history_plans,
    }


def execute_plan_node(state: ReadAgentState):
    """路由节点：判断是否还有计划需要执行"""
    current_plan_index = state.get("current_plan_index", 0)
    history_plans = state.get("history_plans", [])

    if current_plan_index >= len(history_plans):
        next_route = "answer"
    else:
        next_route = "read"
    return {
        "next_route": next_route,
    }


def read_node(state: ReadAgentState):
    """执行单个计划项"""
    files = state["files"]
    history_plans = state.get("history_plans", [])
    current_plan_index = state.get("current_plan_index", 0)
    language = state.get("language", "en")

    if current_plan_index >= len(history_plans):
        warning_msg = _get_localized_message(
            {
                "zh": "current_plan_index 超出范围",
                "en": "current_plan_index out of range",
            },
            language,
        )
        logger.warning(warning_msg)
        return {}

    # 收集之前已读取的结果
    previous_results_list = []
    for i in range(current_plan_index):
        prev_plan = history_plans[i]
        prev_result = prev_plan.get("result")
        if prev_result:
            previous_results_list.append(
                {
                    "file_name": prev_plan.get("file_name", ""),
                    "file_path": prev_plan.get("file_path", ""),
                    "plan_detail": prev_plan.get("plan_detail", ""),
                    "result": prev_result,
                }
            )

    if previous_results_list:
        previous_results_str = json.dumps(
            previous_results_list,
            ensure_ascii=False,
            indent=2,
        )
    else:
        previous_results_str = _get_localized_message(
            {
                "zh": "尚未读取任何文件。",
                "en": "No previous files have been read yet.",
            },
            language,
        )

    current_plan = history_plans[current_plan_index]
    file_name = current_plan.get("file_name", "")
    file_path = current_plan.get("file_path", "")
    plan_detail = current_plan.get("plan_detail", "")

    # 根据 file_name + file_path 找到对应的 FileInfo
    file_info: Optional[FileInfo] = None
    for f in files:
        if f.get("file_name") == file_name and f.get("file_path") == file_path:
            file_info = f
            break

    if not file_info:
        warning_msg = _get_localized_message(
            {
                "zh": f"文件信息不存在: {file_name} ({file_path})",
                "en": f"File info not found: {file_name} ({file_path})",
            },
            language,
        )
        logger.warning(warning_msg)
        history_plans[current_plan_index]["result"] = warning_msg
        return {
            "history_plans": history_plans,
            "current_plan_index": current_plan_index + 1,
        }

    result = ""

    # 判断文件类型并处理（通过扩展名简单区分）
    if _is_image_file(file_path):
        # 图像文件：使用多模态模型分析
        if language == "zh":
            text_content = f"文件: {file_name}\n路径: {file_path}\n类型: 图像文件\n\n请分析图像内容并回答: {plan_detail}"
        else:
            text_content = f"File: {file_name}\nPath: {file_path}\nType: Image file\n\nPlease analyze the image content and answer: {plan_detail}"

        content_payload = [{"image": f"file://{file_path}"}, {"text": text_content}]
        try:
            result = run_multimodal_llm(content_payload, node_name="read_node_image")
        except Exception as exc:  # noqa: BLE001
            error_msg = _get_localized_message(
                {
                    "zh": f"[错误] 图像分析失败: {exc}",
                    "en": f"[Error] Image analysis failed: {exc}",
                },
                language,
            )
            logger.error(error_msg, exc_info=True)
            result = error_msg

    elif _is_data_file(file_name):
        # 数据文件：不在此处读取文件，只生成和执行分析代码
        repl = PythonREPL()
        code = ""
        execution_result = None
        execution_success = False
        analysis_guidance = ""

        try:
            user_query = state.get("user_query", "")
            preview_analysis_prompt = format_data_preview_analysis_prompt(
                user_query=user_query,
                file_info={
                    "file_name": file_name,
                    "file_path": file_path,
                    "preview": file_info.get("preview", ""),
                    "description": file_info.get("description", ""),
                },
                previous_results=previous_results_str,
                language=language,
            )
            guidance_response = run_llm(
                preview_analysis_prompt,
                temperature=0.1,
                max_tokens=2000,
                use_codegen=False,
                node_name="read_node_preview_analysis",
            )
            parsed = _parse_json_response(guidance_response)
            analysis_guidance = parsed.get("guidance", "") or guidance_response.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("数据预览分析失败: %s", exc, exc_info=True)

        for attempt in range(5):
            try:
                if attempt == 0:
                    prompt = format_code_generation_prompt(
                        instruction=plan_detail,
                        file_info={
                            "file_name": file_name,
                            "file_path": file_path,
                            "preview": file_info.get("preview", ""),
                            "description": file_info.get("description", ""),
                        },
                        previous_results=previous_results_str,
                        analysis_guidance=analysis_guidance,
                        language=language,
                    )
                else:
                    if execution_result is None or not hasattr(
                        execution_result, "stderr"
                    ):
                        break
                    prompt = format_code_retry_prompt(
                        user_query=user_query,
                        instruction=plan_detail,
                        file_info={
                            "file_name": file_name,
                            "file_path": file_path,
                            "preview": file_info.get("preview", ""),
                            "description": file_info.get("description", ""),
                        },
                        previous_code=code,
                        error_message=getattr(execution_result, "stderr", "") or "",
                        previous_results=previous_results_str,
                        language=language,
                    )

                response_text = run_llm(
                    prompt,
                    temperature=0.1,
                    max_tokens=5000,
                    use_codegen=True,
                    node_name=f"read_node_codegen_attempt_{attempt + 1}",
                )

                # 代码生成节点约定优先返回 JSON：{"code": "..."}，
                # 但在实际日志中，经常会直接返回 ```python 代码块。
                # 这里先尝试按 JSON 解析；若失败，则将整个响应当作原始代码字符串处理。
                parsed = _parse_json_response(response_text)
                parsed_code = parsed.get("code", "")
                if not parsed_code:
                    text = response_text.strip()
                    if text.startswith("```"):
                        lines = text.splitlines()
                        # 去掉起始 ``` / ```python 等
                        if lines and lines[0].lstrip().startswith("```"):
                            lines = lines[1:]
                        # 去掉结尾 ```
                        if lines and lines[-1].lstrip().startswith("```"):
                            lines = lines[:-1]
                        parsed_code = "\n".join(lines)
                    else:
                        parsed_code = text

                code = (parsed_code or "").strip()
                # 去除顶层 if __name__ == "__main__": 结构，避免在 REPL 中被触发
                code = _strip_if_main_block(code)
                if not code:
                    result = _get_localized_message(
                        {
                            "zh": "[错误] 代码生成失败",
                            "en": "[Error] Code generation failed",
                        },
                        language,
                    )
                    break

                execution_result = repl.run(code)
                stdout = getattr(execution_result, "stdout", "") or ""
                stderr = getattr(execution_result, "stderr", "") or ""

                if stderr.strip():
                    logger.warning(
                        "代码执行 stderr 非空（attempt=%s）: %s",
                        attempt + 1,
                        stderr[:500],
                    )
                    if attempt == 2:
                        result = _get_localized_message(
                            {
                                "zh": f"[错误] 代码执行失败: {stderr}",
                                "en": f"[Error] Code execution failed: {stderr}",
                            },
                            language,
                        )
                else:
                    result = stdout or _get_localized_message(
                        {
                            "zh": "[成功] 代码执行完成，但无输出",
                            "en": "[Success] Code execution completed, but no output",
                        },
                        language,
                    )
                    execution_success = True
                    break
            except Exception as exc:  # noqa: BLE001
                logger.error("代码生成或执行异常: %s", exc, exc_info=True)
                if attempt == 2:
                    result = _get_localized_message(
                        {
                            "zh": f"[错误] 代码生成或执行异常: {exc}",
                            "en": f"[Error] Code generation or execution exception: {exc}",
                        },
                        language,
                    )

        if not execution_success and not result:
            result = _get_localized_message(
                {
                    "zh": "[错误] 代码执行失败，已重试3次",
                    "en": "[Error] Code execution failed, retried 3 times",
                },
                language,
            )

    else:
        # 文本文件：直接使用 preview 作为内容（不读取磁盘）
        preview = file_info.get("preview", "")
        if not preview:
            result = _get_localized_message(
                {
                    "zh": "[错误] 该文本文件未提供预览内容，无法分析",
                    "en": "[Error] No preview content provided for this text file",
                },
                language,
            )
        else:
            summary_prompt = format_text_summary_prompt(
                instruction=plan_detail,
                file_content=preview,
                previous_results=previous_results_str,
                language=language,
            )
            result = run_llm(
                summary_prompt,
                temperature=0.1,
                max_tokens=4000,
                node_name="read_node_text_summary",
            )

    history_plans[current_plan_index]["result"] = result

    logger.info(
        "Read node completed",
        extra={
            "file_name": file_name,
            "file_path": file_path,
        },
    )

    return {
        "history_plans": history_plans,
        "current_plan_index": current_plan_index + 1,
    }


def answer_node(state: ReadAgentState):
    """汇总所有计划结果，生成最终答案"""
    user_query = state["user_query"]
    history_plans = state.get("history_plans", [])
    language = state.get("language", "en")

    execution_results = []
    for plan in history_plans:
        result_str = plan.get("result", "")
        is_error = bool(result_str and (result_str.startswith("[错误]") or result_str.startswith("[Error]")))
        execution_results.append(
            {
                "file_name": plan.get("file_name", ""),
                "file_path": plan.get("file_path", ""),
                "plan_detail": plan.get("plan_detail", ""),
                "result": result_str,
                "success": not is_error,
            }
        )

    answer_prompt = format_answer_prompt(
        user_query=user_query,
        execution_results=execution_results,
        language=language,
    )
    response_text = run_llm(
        answer_prompt,
        temperature=0.1,
        max_tokens=8000,
        node_name="answer_node",
    )
    parsed = _parse_json_response(response_text)
    final_answer = parsed.get("final_answer", "")

    if not final_answer:
        # Fallback：简单地汇总每个文件的状态
        completion_msg = _get_localized_message(
            {
                "zh": "执行完成。\n\n",
                "en": "Execution completed.\n\n",
            },
            language,
        )
        success_status = _get_localized_message(
            {
                "zh": "成功",
                "en": "Success",
            },
            language,
        )
        failed_status = _get_localized_message(
            {
                "zh": "失败",
                "en": "Failed",
            },
            language,
        )
        file_prefix = _get_localized_message(
            {
                "zh": "- 文件",
                "en": "- File",
            },
            language,
        )
        final_answer = completion_msg
        for plan in history_plans:
            file_name = plan.get("file_name", "")
            result = plan.get("result", "")
            is_error = bool(result and (result.startswith("[错误]") or result.startswith("[Error]")))
            status = success_status if result and not is_error else failed_status
            final_answer += f"{file_prefix}: {file_name} - {status}\n"

    return {
        "final_answer": final_answer,
    }


