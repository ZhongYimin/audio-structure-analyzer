"""结果输出与序列化。"""

from __future__ import annotations

import json

from ..core import AnalysisResult


def to_json(result: AnalysisResult, indent: int = 2) -> str:
    """将分析结果序列化为 JSON 字符串。"""
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)


def write_json(result: AnalysisResult, path: str) -> None:
    """将分析结果写入 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(result))
