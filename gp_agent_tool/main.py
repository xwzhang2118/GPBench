import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, TypedDict

from experience.dataset_summary_info import dataset_summary_info
from experience.experience_info import experience_info
from experience.create_masked_dataset_summary import create_masked_dataset_summary
from compute_dataset_feature import process_one_phenotype
from experience.get_matched_experience import get_matched_experience
from llm_client import run_llm
from read_agent import run_read_agent


def _detect_language(text: str) -> str:
    """
    根据输入文本中英文字符占比判断语言。
    如果英文字符占比 >= 50%，则返回 'en'，否则返回 'zh'。
    """
    if not text:
        return "zh"  # 默认为中文
    
    total_chars = 0
    english_chars = 0
    
    for char in text:
        if char.isalpha():  # 只统计字母字符
            total_chars += 1
            if char.isascii() and char.isalpha():  # 英文字母
                english_chars += 1
    
    if total_chars == 0:
        return "zh"  # 如果没有字母字符，默认为中文
    
    english_ratio = english_chars / total_chars
    return "en" if english_ratio >= 0.5 else "zh"


def _build_similarity_prompt(query_dataset_summary: dict, language: str = "zh") -> str:
    """构造相似数据集查询 prompt。"""
    if language == "en":
        return (
            "Based on the statistical information of the following dataset, "
            "find the datasets with the most similar distribution to this dataset, "
            "and provide detailed reasons.\n"
            "Please clearly list the names of these similar datasets in your answer, "
            "and each name must be in the format species/phenotype_name, "
            "for example human/bmi, mouse/height, etc."
            f"\nStatistical information: {query_dataset_summary}"
        )
    else:
        return (
            "根据以下数据集的统计信息，找出与该数据集分布最相似的几个数据集，并给出详细原因。\n"
            "请在回答中明确列出这些相似数据集的名称，且每个名称的格式必须为 species/phenotype_name，"
            "例如 human/bmi、mouse/height 等。"
            f"\n统计信息：{query_dataset_summary}"
        )


def _build_method_prompt(user_query: str, language: str = "zh") -> str:
    """构造方法推荐 prompt。"""
    if language == "en":
        return (
            "Based on user requirements and experimental performance of similar datasets, "
            "recommend suitable algorithms and provide detailed reasoning.\n"
            f"\nUser requirements: {user_query}"
        )
    else:
        return (
            "根据用户需求与相似数据集的实验表现，推荐适合的算法，并给出详细的推荐理由。\n"
            f"\n用户需求：{user_query}"
        )


def _build_file_overview(info: dict) -> str:
    """将文件元信息格式化为 overview 字符串，供 read agent 使用。"""
    name = info.get("file_name", "")
    desc = info.get("description", "")
    path = info.get("file_path", "")
    preview = info.get("preview", "")
    return f"文件名: {name}\n描述: {desc}\n路径: {path}\n预览:\n{preview}"


def _call_read_agent(prompt: str, info: dict, language: str = "zh") -> str:
    """
    使用本地 read_agent 调用，返回 final_answer 文本。
    只依赖单文件信息，避免外部传参。
    """
    files = [
        {
            "file_name": info.get("file_name", ""),
            "file_path": info.get("file_path", ""),
            "description": info.get("description", ""),
            "preview": info.get("preview", ""),
        }
    ]
    overview = _build_file_overview(info)
    state = run_read_agent(
        user_query=prompt,
        files=files,
        file_overview=overview,
        language=language,
    )
    return state.get("final_answer", "") or ""


def _normalize_list_with_reason(
    task_prompt: str,
    raw_answer: str,
    *,
    language: str = "zh",
    node_name: str = "normalize_list_with_reason",
) -> Optional[dict]:
    """
    使用二次 LLM 调用，将自由文本规范化为
    {"items": [...], "reason": "..."} 结构；返回 JSON 或 markdown JSON 代码块。
    """
    if language == "en":
        normalize_prompt = (
            "You are an assistant responsible for result normalization. "
            "Now there is an upstream LLM's answer that needs to be refined into structured JSON.\n\n"
            "[Task Description]\n"
            f"{task_prompt}\n\n"
            "[Upstream Answer]\n"
            f"{raw_answer}\n\n"
            "Please summarize from the upstream answer:\n"
            "1. A string list items, giving the recommended items in order (such as dataset names or method names);\n"
            "2. A string reason, briefly explaining the overall rationale.\n\n"
            "Return format requirement: Strictly output a JSON object, in the form:\n"
            '{"items": ["item1", "item2"], "reason": "overall rationale"}\n'
            "You can directly output JSON, or wrap JSON in a markdown code block ```json; do not output any additional text."
        )
    else:
        normalize_prompt = (
            "你是一个负责结果规范化的助手。现在有一个上游 LLM 的中文回答，需要你将其提炼为结构化 JSON。\n\n"
            "【任务描述】\n"
            f"{task_prompt}\n\n"
            "【上游回答】\n"
            f"{raw_answer}\n\n"
            "请根据上游回答，总结出：\n"
            "1. 一个字符串列表 items，依次给出推荐的项目（如数据集名称或方法名称）；\n"
            "2. 一个字符串 reason，简要说明总体理由。\n\n"
            "返回格式要求：严格输出一个 JSON 对象，形如：\n"
            '{"items": ["item1", "item2"], "reason": "总体理由"}\n'
            "你可以直接输出 JSON，或用 markdown 代码块 ```json 包裹 JSON；不要输出任何额外文字。"
        )

    try:
        norm_text = run_llm(
            prompt=normalize_prompt,
            temperature=0.1,
            max_tokens=512,
            use_codegen=True,
            node_name=node_name,
        )
    except Exception:
        return None

    text = norm_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    items = data.get("items", [])
    reason = data.get("reason", "")
    if not isinstance(items, list):
        return None

    parsed_items = [str(x).strip() for x in items if str(x).strip()]
    return {"items": parsed_items, "reason": str(reason).strip()}


class _ParsedList(TypedDict):
    items: list[str]
    reason: str


def _read_agent_list(
    prompt: str,
    info: dict,
    fallback_parser: Optional[Callable[[str], Iterable[str]]] = None,
    language: str = "zh",
) -> _ParsedList:
    """
    调用 read_agent 并解析 JSON 列表，同时带上总体理由；
    若解析失败则回退到简单拆分，reason 置空。
    """
    answer = _call_read_agent(prompt, info, language)

    # 先用二次 LLM 规范化为 JSON 结构
    normalized = _normalize_list_with_reason(prompt, answer, language=language)
    if normalized and normalized.get("items"):
        return normalized  # type: ignore[return-value]

    # 若规范化失败或 items 为空，则退回到简单解析
    parser = fallback_parser
    return {"items": list(parser(answer)), "reason": ""}


def get_recommend_method(
    query_dataset_path: Optional[str],
    user_query: str,
    masked_dataset_name: Optional[str] = None,
) -> dict:
    """
    根据用户查询和数据集路径，推荐适合的分析方法，并返回理由。

    参数
    ----
    query_dataset_path : str | None
        目标数据集目录，包含 genetype.npz / phenotype.npz。
        若为 None，则不基于具体数据集查找相似数据集，而是直接基于完整经验表推荐方法。
    user_query : str
        用户的分析需求描述。
    masked_dataset_name : str, optional
        若提供，则在参考库中过滤该 species_phenotype。
    返回
    ----
    dict
        {
            "similar_datasets": {"items": [...], "reason": "..."},
            "methods": {"items": [...], "reason": "..."},
        }
    """
    # 在方法开始时检测语言
    detected_language = _detect_language(user_query)

    # 若未提供数据集路径，则直接基于完整经验表推荐方法
    if not query_dataset_path:
        matched_experience_info = experience_info.copy()
        method_prompt = _build_method_prompt(user_query, detected_language)
        method_result = _read_agent_list(method_prompt, matched_experience_info, language=detected_language)
        reason_msg = (
            "Dataset path not provided, recommending methods based on complete experience table only."
            if detected_language == "en"
            else "未提供数据集路径，仅基于完整经验表推荐方法。"
        )
        return {
            "similar_datasets": {
                "items": [],
                "reason": reason_msg,
            },
            "methods": method_result,
        }

    # 1) 获取查询数据集的统计信息
    query_dataset_summary = process_one_phenotype(query_dataset_path)

    # 2) 处理参考数据集（可选屏蔽指定数据集）
    if masked_dataset_name:
        ref_dataset_summary_path, ref_dataset_summary_preview = create_masked_dataset_summary(
            [masked_dataset_name]
        )
    else:
        ref_dataset_summary_path = dataset_summary_info["file_path"]
        ref_dataset_summary_preview = dataset_summary_info["preview"]

    ref_dataset_summary_info = dataset_summary_info.copy()
    ref_dataset_summary_info["file_path"] = ref_dataset_summary_path
    ref_dataset_summary_info["preview"] = ref_dataset_summary_preview

    # 3) 调用 read agent 获取相似数据集名称
    similarity_prompt = _build_similarity_prompt(query_dataset_summary, detected_language)
    similar_result = _read_agent_list(similarity_prompt, ref_dataset_summary_info, language=detected_language)
    similar_dataset_names = similar_result["items"]
    if not similar_dataset_names:
        return {
            "similar_datasets": {"items": [], "reason": similar_result.get("reason", "")},
            "methods": {"items": [], "reason": ""},
        }

    # 4) 筛选匹配的经验表
    matched_experience_path, matched_experience_preview = get_matched_experience(
        similar_dataset_names, experience_info["file_path"]
    )
    matched_experience_info = experience_info.copy()
    matched_experience_info["file_path"] = matched_experience_path
    matched_experience_info["preview"] = matched_experience_preview

    # 5) 调用 LLM 推荐方法
    method_prompt = _build_method_prompt(user_query, detected_language)
    method_result = _read_agent_list(method_prompt, matched_experience_info, language=detected_language)

    return {"similar_datasets": similar_result, "methods": method_result}


def _parse_args() -> argparse.Namespace:
    """CLI 参数解析。"""
    parser = argparse.ArgumentParser(
        description="根据数据集统计信息与用户需求推荐分析方法",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="可选：待分析数据集目录，包含 genetype.npz 和 phenotype.npz；不提供则仅基于经验表推荐方法",
    )
    parser.add_argument(
        "-q",
        "--user-query",
        required=True,
        help="用户对分析需求的描述",
    )
    parser.add_argument(
        "-m",
        "--mask",
        dest="masked_dataset_name",
        help="可选：需要在参考库中屏蔽的 species_phenotype",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="可选：将结果保存到 JSON 文件路径；不提供则直接打印",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = get_recommend_method(
        query_dataset_path=args.dataset,
        user_query=args.user_query,
        masked_dataset_name=args.masked_dataset_name,
    )

    if args.output:
        path = Path(args.output)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"result saved to: {path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()