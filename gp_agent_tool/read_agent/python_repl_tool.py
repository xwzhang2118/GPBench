"""
简化版 Python 代码解释器工具。

参考 textMSA 中的 PythonREPL 实现，仅保留「执行代码」部分逻辑：
- 接收一段 Python 代码字符串
- 在受控环境中执行（支持表达式和多行脚本）
- 捕获 stdout / stderr
- 返回结构化的执行结果对象，便于上层判断是否成功
"""

from __future__ import annotations

import contextlib
import io
import time
from dataclasses import dataclass
from typing import Any, Optional

from logging_utils import get_logger


logger = get_logger(__name__)


@dataclass
class PythonREPLExecutionResult:
    """代码执行结果"""

    stdout: str
    stderr: str
    execution_time: float
    success: bool
    error: Optional[Exception] = None


class PythonREPL:
    """
    轻量级 Python 代码执行器。

    设计目标：
    - 与 langchain_experimental.utilities.PythonREPL 的接口尽量兼容（提供 run(code)）
    - 保留跨调用的全局执行环境（可以在多次执行中复用变量）
    - 捕获 stdout / stderr，供上层逻辑使用
    """

    def __init__(self, max_code_length: int = 10_000) -> None:
        self._max_code_length = max_code_length
        # 共享全局环境，便于多次执行之间复用变量
        self._exec_globals: dict[str, Any] = {}
        logger.info(
            "PythonREPL initialized",
            extra={"max_code_length": max_code_length},
        )

    def run(self, code: str) -> PythonREPLExecutionResult:
        """执行一段 Python 代码并返回执行结果。"""
        if not code:
            return PythonREPLExecutionResult(
                stdout="",
                stderr="代码为空",
                execution_time=0.0,
                success=False,
            )

        if len(code) > self._max_code_length:
            return PythonREPLExecutionResult(
                stdout="",
                stderr=f"代码长度超过限制 ({len(code)} > {self._max_code_length})",
                execution_time=0.0,
                success=False,
            )

        logger.info("Executing Python code", extra={"code_length": len(code)})
        logger.debug("Code to execute", extra={"code_preview": code[:500]})

        start_time = time.perf_counter()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # 判定使用 eval 还是 exec
        try:
            code_obj = compile(code, "<python-repl>", "eval")
            use_eval = True
        except SyntaxError:
            code_obj = compile(code, "<python-repl>", "exec")
            use_eval = False

        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
                stderr_buf
            ):
                if use_eval:
                    result = eval(code_obj, self._exec_globals)  # noqa: S307
                else:
                    exec(code_obj, self._exec_globals)  # noqa: S102
                    result = None

            execution_time = time.perf_counter() - start_time

            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
            success = True

            # eval 模式下如果有返回值，将其追加到 stdout，便于查看
            if use_eval and result is not None:
                result_str = result if isinstance(result, str) else str(result)
                if stdout and not stdout.endswith("\n"):
                    stdout += "\n"
                stdout += result_str

            logger.info(
                "Code execution completed",
                extra={
                    "execution_time": execution_time,
                    "stdout_length": len(stdout),
                    "stderr_length": len(stderr),
                    "success": success,
                },
            )
            if stdout:
                logger.info("Code execution stdout", extra={"stdout": stdout})
            if stderr:
                logger.warning("Code execution stderr", extra={"stderr": stderr})

            return PythonREPLExecutionResult(
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                success=success,
            )

        except BaseException as exc:  # noqa: BLE001
            # 捕获 SystemExit / KeyboardInterrupt 等，避免上层进程被直接退出
            execution_time = time.perf_counter() - start_time
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()

            # 将异常信息追加到 stderr，便于上层展示
            if stderr:
                stderr = f"{stderr}\n{exc}"
            else:
                stderr = str(exc)

            logger.error(
                "Code execution failed",
                extra={
                    "execution_time": execution_time,
                    "error": stderr,
                },
                exc_info=True,
            )

            return PythonREPLExecutionResult(
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                success=False,
                error=exc if isinstance(exc, Exception) else None,
            )


__all__ = ["PythonREPL", "PythonREPLExecutionResult"]


