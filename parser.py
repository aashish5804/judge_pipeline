"""
Parses the judge's raw text into a structured verdict. Judges don't always
return clean JSON (markdown fences, leading prose, trailing commentary,
truncated output). Recovery strategy, in order:

  1. Try json.loads on the raw text as-is.
  2. Strip ```json ... ``` fences and retry.
  3. Regex-extract the first {...} block (brace-matched) and retry.
  4. If still unparseable, return a FAILED verdict with the lowest possible
     score and a flag — never silently drop the case, never invent scores.
"""
import json
import re


class ParseResult:
    def __init__(self, ok, data=None, error=None, recovered_via=None):
        self.ok = ok
        self.data = data
        self.error = error
        self.recovered_via = recovered_via  # None | "fence_strip" | "brace_match"


def _strip_fences(text):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_first_brace_block(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_verdict(raw_text):
    # 1. direct
    try:
        return ParseResult(True, json.loads(raw_text))
    except json.JSONDecodeError:
        pass

    # 2. strip fences
    fenced = _strip_fences(raw_text)
    if fenced:
        try:
            return ParseResult(True, json.loads(fenced), recovered_via="fence_strip")
        except json.JSONDecodeError:
            pass

    # 3. brace-matched extraction
    block = _extract_first_brace_block(raw_text)
    if block:
        try:
            return ParseResult(True, json.loads(block), recovered_via="brace_match")
        except json.JSONDecodeError as e:
            return ParseResult(False, error=f"brace block still invalid: {e}")

    return ParseResult(False, error="no JSON object found in judge response")


def validate_pointwise_schema(data, expected_criteria):
    """Returns (ok, missing_criteria). Doesn't raise — caller decides how to
    treat a partially-valid verdict."""
    if "per_criterion" not in data:
        return False, list(expected_criteria)
    missing = [c for c in expected_criteria if c not in data["per_criterion"]]
    return (len(missing) == 0), missing
