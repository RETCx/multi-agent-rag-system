def _extract_text(content) -> str:
    """
    Extract plain text from AIMessage content.

    Handles two response formats:
    - Chat Completions API: content is a plain string
    - Responses API (Azure gateway): content is a list of typed blocks
      e.g. [{"type": "reasoning", ...}, {"type": "text", "text": "..."}]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content)
