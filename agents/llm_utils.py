"""Small helpers shared by the agents (no LLM/network dependencies)."""
import json
import re
from typing import Any, Dict, Optional


def text_of(content: Any) -> str:
    """LangChain message content may be a str or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Pull the first JSON object out of an LLM reply (tolerates ```json fences and
    chatter around it). Returns None if nothing parseable is found. Works with
    any model -- no dependence on provider-specific structured-output features.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = cleaned.find("{", start + 1)
    return None
