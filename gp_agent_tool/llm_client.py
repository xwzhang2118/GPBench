from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from dashscope import MultiModalConversation
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import (
    get_codegen_llm_config,
    get_llm_config,
    get_multimodal_llm_config,
)
from logging_utils import get_logger


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _base_llm_config() -> Dict[str, Any]:
    return get_llm_config()


@lru_cache(maxsize=1)
def _base_codegen_llm_config() -> Dict[str, Any]:
    return get_codegen_llm_config()


def _build_chat_llm(
    *, temperature: float, max_tokens: int, use_codegen: bool = False
) -> ChatOpenAI:
    base_config = _base_codegen_llm_config() if use_codegen else _base_llm_config()
    params: Dict[str, Any] = {
        "model": base_config.get("model"),
        "api_key": base_config.get("api_key"),
        "base_url": base_config.get("base_url"),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if base_config.get("timeout_seconds") is not None:
        params["timeout"] = base_config["timeout_seconds"]
    if base_config.get("max_retries") is not None:
        params["max_retries"] = base_config["max_retries"]
    return ChatOpenAI(**params)


def run_llm(
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    use_codegen: bool = False,
    node_name: str = "unknown",
) -> str:
    """单轮对话 LLM 调用，返回文本内容。"""
    base_config = _base_codegen_llm_config() if use_codegen else _base_llm_config()
    model_name = base_config.get("model", "unknown")

    logger.info(
        "[LLM Input] Node: %s | Model: %s | UseCodegen: %s",
        node_name,
        model_name,
        use_codegen,
    )
    logger.info("[LLM Input Full] Node: %s\n%s", node_name, prompt)

    llm = _build_chat_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        use_codegen=use_codegen,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    response_content = getattr(response, "content", "") or ""

    logger.info("[LLM Output] Node: %s | Model: %s", node_name, model_name)
    logger.info("[LLM Output Full] Node: %s\n%s", node_name, response_content)

    return response_content


def run_multimodal_llm(
    content_payload: List[dict],
    *,
    node_name: str = "unknown",
) -> str:
    """多模态 LLM 调用，当前用于图像分析。"""
    multimodal_config = get_multimodal_llm_config()
    model_name = multimodal_config.get("model", "unknown")

    messages = [{"role": "user", "content": content_payload}]

    logger.info(
        "[Multimodal LLM Input] Node: %s | Model: %s",
        node_name,
        model_name,
    )

    response = MultiModalConversation.call(
        api_key=multimodal_config["api_key"],
        model=model_name,
        messages=messages,
    )

    text = ""
    if response.output and response.output.choices:
        text = response.output.choices[0].message.content[0].get("text", "") or ""

    logger.info(
        "[Multimodal LLM Output] Node: %s | Model: %s",
        node_name,
        model_name,
    )
    logger.info("[Multimodal LLM Output Full] Node: %s\n%s", node_name, text)

    return text


