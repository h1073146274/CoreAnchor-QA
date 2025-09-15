import json
import re
from typing import List

def parse_json_array_block(s: str) -> List[str]:
    if not s:
        return []
    m = re.search(r"\[.*?\]", s, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue
        if t[0] in "-*•" or re.match(r"^\d+[\.\-\)]\s+", t):
            t = re.sub(r"^\s*(?:\d+[\.\-\)]\s+|[-*•]\s+)", "", t)
            lines.append(t)
        elif "?" in t:
            lines.append(t)
    return lines[:3]
