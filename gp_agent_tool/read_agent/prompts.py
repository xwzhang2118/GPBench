"""
Read Agent Prompt 模板（对齐 textMSA 版本，移除 textmsa 依赖）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from logging_utils import get_logger


logger = get_logger(__name__)


def _normalize_language(language: Optional[str]) -> str:
    """Normalize language input; default to English."""
    if not language:
        return "en"
    lower = language.lower()
    if lower.startswith("zh"):
        return "zh"
    if lower.startswith("en"):
        return "en"
    return "en"


def _get_prompt(prompt_map: dict[str, str], language: Optional[str]) -> str:
    """Select prompt by language with English fallback."""
    lang = _normalize_language(language)
    return prompt_map.get(lang, prompt_map["en"])


PLAN_PROMPT = {
    "en": """
You are a file reading planning assistant. Based on the user question and the provided file overview, create a SEQUENTIAL reading plan that lists all files that need to be processed in a specific order.

User question:
{user_query}

File overview:
{file_overview}

**Planning Rules:**
1. **Select appropriate files:**
   - For text files (.txt, .md, .py, .js, config files, README, notes, documentation): Include them in the plan for direct reading
   - For image files (.png, .jpg, .jpeg, .gif, .bmp, .svg): Include them in the plan for image analysis
   - For data files (.csv, .h5ad, .json, .parquet, .xlsx, etc.): Include them in the plan for code-based analysis
   - Only include files that are relevant to answering the user question

2. **Plan structure:**
   - Each plan item should specify: file_name, file_path, plan_detail, and order_reasoning
   - plan_detail should describe what needs to be done with this file (e.g., "Read and summarize the content", "Analyze the data and extract key statistics", "Describe the image content")
   - order_reasoning should explain WHY this file should be read at this position in the sequence (e.g., "Read first to understand the project structure", "Read after config files to understand data format", "Read last to synthesize all information")

3. **Sequential ordering is critical:**
   - List files in the EXACT order they should be processed sequentially
   - Even if files are independent, you must determine an optimal reading order
   - Consider: foundational files first (README, configs), then supporting files, then data files, then analysis files
   - The order_reasoning for each file should explain its position relative to other files
   - Provide overall reasoning that explains the entire sequence strategy

Return JSON:
{{
  "plans": [
    {{
      "file_name": "filename.ext",
      "file_path": "/path/to/file",
      "plan_detail": "Description of what to do with this file",
      "order_reasoning": "Why this file should be read at this position (e.g., 'Read first because it contains project overview', 'Read after file X to understand context', etc.)"
    }}
  ],
  "reasoning": "Overall reasoning for the entire sequential reading plan, explaining the strategy and why files are ordered this way"
}}

**Important Notes:**
- Do not use placeholders or example values
- The order matters: files will be read sequentially, and each read can see results from previous reads
- Provide clear reasoning for the order, even if files seem independent
""",
    "zh": """
你是文件阅读规划助手。根据用户问题和外部提供的文件概览字符串，制定顺序阅读计划，按特定顺序列出所有需要处理的文件。

用户问题：
{user_query}

文件概览：
{file_overview}

**规划规则：**
1. **选择合适的文件：**
   - 对于文本文件（.txt, .md, .py, .js, 配置文件, README, 笔记, 文档等）：包含在计划中，用于直接读取
   - 对于图像文件（.png, .jpg, .jpeg, .gif, .bmp, .svg 等）：包含在计划中，用于图像分析
   - 对于数据文件（.csv, .h5ad, .json, .parquet, .xlsx 等）：包含在计划中，用于基于代码的分析
   - 只包含与回答用户问题相关的文件

2. **计划结构：**
   - 每个计划项应指定：file_name, file_path, plan_detail, 和 order_reasoning
   - plan_detail 应描述需要对该文件做什么（例如："读取并总结内容"、"分析数据并提取关键统计信息"、"描述图像内容"）
   - order_reasoning 应解释为什么该文件应该在此顺序位置读取（例如："首先读取以了解项目结构"、"在配置文件之后读取以了解数据格式"、"最后读取以综合所有信息"）

3. **顺序至关重要：**
   - 按顺序处理的精确顺序列出文件
   - 即使文件是独立的，也必须确定最优的阅读顺序
   - 考虑：基础文件优先（README、配置文件），然后是支持文件，然后是数据文件，最后是分析文件
   - 每个文件的 order_reasoning 应解释其相对于其他文件的位置
   - 提供整体推理，解释整个顺序策略

请返回JSON：
{{
  "plans": [
    {{
      "file_name": "文件名.ext",
      "file_path": "/路径/到/文件",
      "plan_detail": "对该文件需要做什么的描述",
      "order_reasoning": "为什么该文件应该在此位置读取（例如：'首先读取因为它包含项目概览'、'在文件X之后读取以了解上下文'等）"
    }}
  ],
  "reasoning": "整个顺序阅读计划的整体推理，解释策略以及为什么文件按此顺序排列"
}}

**重要提示：**
- 不要使用占位符或示例值
- 顺序很重要：文件将按顺序读取，每次读取都可以看到之前读取的结果
- 即使文件看起来是独立的，也要为顺序提供清晰的推理
""",
}


DATA_PREVIEW_ANALYSIS_PROMPT = {
    "en": """
You are a data analysis planning assistant. Based on the user query and data file preview, generate guidance information to help with subsequent code generation.

User query:
{user_query}

File information:
{file_info}

**Previous Reading Results:**
{previous_results}

**Task:**
Analyze the data preview and user query, then generate structured guidance that includes:
1. **Data Characteristics**: Key features of the data (structure, columns, data types, size, etc.)
2. **Analysis Objectives**: What needs to be analyzed based on the user query
3. **Code Generation Strategy**: Recommended approach for generating analysis code (which libraries to use, key operations needed, etc.)
4. **Important Notes**: Potential pitfalls, data quality issues, or special considerations

**Return Format:**
You MUST return a JSON object (not markdown code block) with the following structure:
{{
  "guidance": "A comprehensive guidance text that summarizes data characteristics, analysis objectives, code generation strategy, and important notes"
}}
""",
    "zh": """
你是数据分析规划助手。根据用户查询和数据文件预览，生成指导信息以帮助后续的代码生成。

用户查询：
{user_query}

文件信息：
{file_info}

**之前的读取结果：**
{previous_results}

**任务：**
分析数据预览和用户查询，然后生成结构化的指导信息，包括：
1. **数据特征**：数据的关键特征（结构、列、数据类型、大小等）
2. **分析目标**：根据用户查询需要分析什么
3. **代码生成策略**：生成分析代码的推荐方法（使用哪些库、需要的关键操作等）
4. **重要注意事项**：潜在的陷阱、数据质量问题或特殊考虑

**返回格式：**
你必须返回一个 JSON 对象（不要使用 markdown 代码块），格式如下：
{{
  "guidance": "综合的指导文本，总结数据特征、分析目标、代码生成策略和重要注意事项"
}}
""",
}


CODE_GENERATION_PROMPT = {
    "en": """
You are a code generation assistant. Generate Python code to analyze the specified data file according to the given instruction.

Instruction:
{instruction}

File information:
{file_info}

**Previous Reading Results:**
{previous_results}

    **Analysis Guidance:**
    {analysis_guidance}
    
    **Python Library Guidelines for Structured Data Files:**
- **CSV files**: Use `pandas.read_csv()` for reading. Use `pandas.DataFrame.to_csv()` for writing.
- **JSON data files**: Use `json.load()` / `json.dump()` or `pandas.read_json()` for reading. Use `json.dump()` / `json.dumps()` or `pandas.DataFrame.to_json()` for writing.
- **h5ad files (AnnData)**: Use `scanpy.read_h5ad()` or `anndata.read_h5ad()` for reading. Use `adata.write_h5ad()` for writing.
- **HDF5 files**: Use `h5py.File()` or `pandas.read_hdf()` for reading. Use `h5py.File().create_dataset()` or `pandas.DataFrame.to_hdf()` for writing.
    - **Parquet files**: Use `pandas.read_parquet()` or `pyarrow.parquet.read_table()` for reading. Use `pandas.DataFrame.to_parquet()` or `pyarrow.parquet.write_table()` for writing.
    - **Excel files**: Use `pandas.read_excel()` (requires `openpyxl` or `xlrd` engine) for reading. Use `pandas.DataFrame.to_excel()` for writing.
    
    **Output Requirements:**
- Read files using the provided real paths.
- Input files may be outside the working directory; read them from their provided paths without relocating.
- Write all output files into the working directory (no outputs outside the working directory).
- **IMPORTANT**: Print the analysis results to stdout so they can be captured. Use print() statements to output key findings, statistics, summaries, etc.
- **NO PLOTTING**: **ABSOLUTELY FORBIDDEN** to use any visualization libraries (e.g., matplotlib, seaborn, plotly, etc.) for plotting during code generation. **DO NOT** generate any plots, images, or visualization outputs. If visualization is needed for analysis, use print() to output numerical results, statistical summaries, and other text information instead.
        - Do not fabricate data or use example data.
        - The generated code must be directly usable on the actual input data files. Do NOT generate placeholder code that waits for future/real data, and do NOT use any mock/simulated data. Always perform analysis based on the provided file paths and current data.
- When errors occur, surface detailed error information (stack trace and contextual details) so failures are easy to debug.
- **CRITICAL - Error Handling Rules:**
  - **NEVER** use try-except blocks that silently swallow exceptions without re-raising them or printing detailed error information
  - If you use try-except, you MUST either:
    (1) Re-raise the exception after logging/printing it, OR
    (2) Print the full error details (including traceback) to stderr using `traceback.print_exc()` or `sys.stderr.write()`
  - **DO NOT** catch exceptions and only print a simple error message without the full traceback
  - **DO NOT** catch exceptions and continue execution silently - this makes debugging impossible
  - If error handling is needed, prefer letting exceptions propagate naturally, or use proper error handling that preserves error information
- **Consider previous reading results**: You can reference information from previously read files to better understand the context and generate more relevant analysis code.

**Result Output Requirements (for stdout output):**
- The output must strictly adhere to the instruction and only report what is found in the execution results
- Use natural narrative text, not lists or structured formats
- Do NOT include any suggestions, recommendations, or advice beyond what is in the execution results
- Do NOT provide any recommendations or suggestions that go beyond the scope of the execution results

**Return Format:**
You MUST return a JSON object (not markdown code block) with the following structure:
{{
  "code": "The Python code as a string (no markdown code block, just raw code)"
}}
""",
    "zh": """
你是代码生成助手。根据给定的指令生成 Python 代码来分析指定的数据文件。

指令：
{instruction}

文件信息：
{file_info}

**之前的读取结果：**
{previous_results}

**分析指导：**
{analysis_guidance}

**结构化数据文件的 Python 库使用指南：**
- **CSV 文件**：使用 `pandas.read_csv()` 读取。使用 `pandas.DataFrame.to_csv()` 写入。
- **JSON 数据文件**：使用 `json.load()` / `json.dump()` 或 `pandas.read_json()` 读取。使用 `json.dump()` / `json.dumps()` 或 `pandas.DataFrame.to_json()` 写入。
- **h5ad 文件（AnnData）**：使用 `scanpy.read_h5ad()` 或 `anndata.read_h5ad()` 读取。使用 `adata.write_h5ad()` 写入。
- **HDF5 文件**：使用 `h5py.File()` 或 `pandas.read_hdf()` 读取。使用 `h5py.File().create_dataset()` 或 `pandas.DataFrame.to_hdf()` 写入。
- **Parquet 文件**：使用 `pandas.read_parquet()` 或 `pyarrow.parquet.read_table()` 读取。使用 `pandas.DataFrame.to_parquet()` 或 `pyarrow.parquet.write_table()` 写入。
- **Excel 文件**：使用 `pandas.read_excel()`（需要 `openpyxl` 或 `xlrd` 引擎）读取。使用 `pandas.DataFrame.to_excel()` 写入。

    **输出要求：**
- 读取文件时使用提供的真实路径。
- 输入文件可能不在工作目录中，请直接使用提供的路径读取，不要搬移。
- 将所有输出文件写入工作目录（不要写到工作目录以外）。
- **重要**：将分析结果打印到 stdout，以便可以捕获。使用 print() 语句输出关键发现、统计信息、摘要等。
- **禁止画图**：代码生成过程中**绝对禁止**使用任何可视化库（如 matplotlib、seaborn、plotly 等）进行画图操作，**禁止**生成任何图表、图像文件或可视化输出。如果分析需要可视化，请使用 print() 输出数值结果、统计摘要等文本信息。
        - 不要伪造数据或使用示例数据。
        - 生成的代码必须直接面向实际输入数据文件，可立即运行。不要生成依赖“未来补充真实数据”的占位代码，也不要使用任何模拟/伪造/示例数据，必须基于当前提供的文件路径和真实数据执行分析。
- 发生错误时需要输出详细的错误信息（包含堆栈与关键上下文），方便定位问题。
- **关键 - 错误处理规则：**
  - **绝对禁止**使用会静默吞掉异常而不重新抛出或打印详细错误信息的 try-except 代码块
  - 如果使用 try-except，你必须：
    (1) 在记录/打印后重新抛出异常，或者
    (2) 使用 `traceback.print_exc()` 或 `sys.stderr.write()` 将完整错误详情（包括堆栈跟踪）打印到 stderr
  - **不要**捕获异常后只打印简单错误信息而不包含完整堆栈跟踪
  - **不要**捕获异常后静默继续执行 - 这会使调试变得不可能
  - 如果需要错误处理，优先让异常自然传播，或使用能保留错误信息的正确错误处理方式
- **考虑之前的读取结果**：你可以参考之前读取文件的信息，以更好地理解上下文并生成更相关的分析代码。

**结果输出要求（stdout / stderr 区分）：**
- 正常的分析结果（成功执行）必须使用 print() 打印到 stdout。
- 错误信息、异常和完整堆栈必须写入 stderr（要么让异常自然抛出，要么使用 `traceback.print_exc()` / `sys.stderr.write()`）。
- 输出内容必须严格忠于指令，只报告执行结果中发现的内容。
- 使用自然的叙述性文字，不要使用列表或结构化格式。
- 不包含任何建议、推荐或超出执行结果范围的建议。

**返回格式：**
你必须返回一个 JSON 对象（不要使用 markdown 代码块），格式如下：
{{
  "code": "Python 代码字符串（不要使用 markdown 代码块，直接返回原始代码）"
}}
""",
}


CODE_RETRY_PROMPT = {
    "en": """
You are a code generation assistant. The previous code execution failed. Please fix the code based on the error message and try again.

User query:
{user_query}

Original instruction:
{instruction}

File information:
{file_info}

**Previous Reading Results:**
{previous_results}

Previous code:
{previous_code}

Error message (stderr):
{error_message}

Please analyze the error, fix the code, and return the corrected version. Consider the previous reading results for context.

**CRITICAL - Error Handling Rules:**
- **NEVER** use try-except blocks that silently swallow exceptions without re-raising them or printing detailed error information
- If you use try-except, you MUST either:
  (1) Re-raise the exception after logging/printing it, OR
  (2) Print the full error details (including traceback) to stderr using `traceback.print_exc()` or `sys.stderr.write()`
- **DO NOT** catch exceptions and only print a simple error message without the full traceback
- **DO NOT** catch exceptions and continue execution silently - this makes debugging impossible
- If error handling is needed, prefer letting exceptions propagate naturally, or use proper error handling that preserves error information

**Result Output Requirements (for stdout output):**
- The output must strictly adhere to the instruction and only report what is found in the execution results
- Use natural narrative text, not lists or structured formats
- Do NOT include any suggestions, recommendations, or advice beyond what is in the execution results
- Do NOT provide any recommendations or suggestions that go beyond the scope of the execution results

**Return Format:**
You MUST return a JSON object (not markdown code block) with the following structure:
{{
  "code": "The corrected Python code as a string (no markdown code block, just raw code)"
}}
""",
    "zh": """
你是代码生成助手。之前的代码执行失败了。请根据错误信息修复代码并重试。

用户问题：
{user_query}

原始指令：
{instruction}

文件信息：
{file_info}

**之前的读取结果：**
{previous_results}

之前的代码：
{previous_code}

错误信息（stderr）：
{error_message}

请分析错误，修复代码，并返回修正后的版本。考虑之前的读取结果以获取上下文。

**关键 - 错误处理规则：**
- **绝对禁止**使用会静默吞掉异常而不重新抛出或打印详细错误信息的 try-except 代码块
- 如果使用 try-except，你必须：
  (1) 在记录/打印后重新抛出异常，或者
  (2) 使用 `traceback.print_exc()` 或 `sys.stderr.write()` 将完整错误详情（包括堆栈跟踪）打印到 stderr
- **不要**捕获异常后只打印简单错误信息而不包含完整堆栈跟踪
- **不要**捕获异常后静默继续执行 - 这会使调试变得不可能
- 如果需要错误处理，优先让异常自然传播，或使用能保留错误信息的正确错误处理方式

**结果输出要求（针对 stdout 输出）：**
- 输出必须严格忠于指令，只报告执行结果中发现的内容
- 使用自然的叙述性文字，不要使用列表或结构化格式
- 不包含任何建议、推荐或超出执行结果范围的建议
- 不要提供任何超出执行结果范围的推荐或建议

**返回格式：**
你必须返回一个 JSON 对象（不要使用 markdown 代码块），格式如下：
{{
  "code": "修正后的 Python 代码字符串（不要使用 markdown 代码块，直接返回原始代码）"
}}
""",
}


ANSWER_PROMPT = {
    "en": """
You are a report generation assistant. Based on the user question and all execution results, produce a comprehensive narrative answer.

User question:
{user_query}

Execution results:
{execution_results}

Return JSON:
{{
  "final_answer": "The narrative final answer based entirely on execution results."
}}
""",
    "zh": """
你是报告生成助手。根据用户问题和所有执行结果，生成综合性的叙述性最终答案。

用户问题：
{user_query}

执行结果：
{execution_results}

请返回 JSON：
{{
  "final_answer": "完全基于执行结果的叙述性最终答案。"
}}
""",
}


def _safe_to_jsonable(data: object):
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if isinstance(data, dict):
        return {k: _safe_to_jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_safe_to_jsonable(v) for v in data]
    return str(data)


def _fmt_json(data: object) -> str:
    if data is None:
        return "null"
    if isinstance(data, str):
        return data
    return json.dumps(_safe_to_jsonable(data), ensure_ascii=False, indent=2)


def format_plan_prompt(
    user_query: str,
    file_overview: str,
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(PLAN_PROMPT, language).format(
        user_query=user_query,
        file_overview=file_overview,
    )


def format_data_preview_analysis_prompt(
    user_query: str,
    file_info: dict[str, Any],
    previous_results: str = "",
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(DATA_PREVIEW_ANALYSIS_PROMPT, language).format(
        user_query=user_query,
        file_info=_fmt_json(file_info),
        previous_results=previous_results or "No previous files have been read yet.",
    )


def format_code_generation_prompt(
    instruction: str,
    file_info: dict[str, Any],
    previous_results: str = "",
    analysis_guidance: str = "",
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(CODE_GENERATION_PROMPT, language).format(
        instruction=instruction,
        file_info=_fmt_json(file_info),
        previous_results=previous_results or "No previous files have been read yet.",
        analysis_guidance=analysis_guidance or "No analysis guidance available.",
    )


def format_code_retry_prompt(
    user_query: str,
    instruction: str,
    file_info: dict[str, Any],
    previous_code: str,
    error_message: str,
    previous_results: str = "",
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(CODE_RETRY_PROMPT, language).format(
        user_query=user_query,
        instruction=instruction,
        file_info=_fmt_json(file_info),
        previous_code=previous_code,
        error_message=error_message,
        previous_results=previous_results or "No previous files have been read yet.",
    )


TEXT_SUMMARY_PROMPT = {
    "en": """
You are a text analysis assistant. Analyze the following file content and answer the user's question based on the content and previous reading results.

User question/requirement: {instruction}

Previous Reading Results:
{previous_results}

Current File Content:
{file_content}
""",
    "zh": """
你是文本分析助手。请分析以下文件内容，并根据内容和之前的读取结果回答用户问题。

用户问题/需求：{instruction}

之前的读取结果：
{previous_results}

当前文件内容：
{file_content}
""",
}


def format_text_summary_prompt(
    instruction: str,
    file_content: str,
    previous_results: str = "",
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(TEXT_SUMMARY_PROMPT, language).format(
        instruction=instruction,
        file_content=file_content,
        previous_results=previous_results
        or ("尚未读取任何文件。" if _normalize_language(language) == "zh" else "No previous files have been read yet."),
    )


def format_answer_prompt(
    user_query: str,
    execution_results: list[dict[str, Any]],
    language: Optional[str] = "en",
) -> str:
    return _get_prompt(ANSWER_PROMPT, language).format(
        user_query=user_query,
        execution_results=_fmt_json(execution_results),
    )


