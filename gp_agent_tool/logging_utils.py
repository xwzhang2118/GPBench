import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    简单的 logger 封装，避免依赖旧项目。

    - 默认使用 INFO 级别。
    - 只在根 logger 尚未配置 handler 时添加一个 StreamHandler。
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    return logger



