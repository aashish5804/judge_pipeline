import json
from rubric import RUBRIC

POINTWISE_SCHEMA = """{
  "per_criterion": {
    "<criterion_name>": {"score": <1-5 integer>, "rationale": "<1-2 sentences, must cite specific text from the output>"}
  },
  "overall_rationale": "<2-3 sentence summary of the verdict>"
}"""

PAIRWISE_SCHEMA = """{
  "winner": "<first|second|tie>",
  "confidence": <0.0-1.0>,
  "rationale": "<2-3 sentences comparing the two on the rubric, must cite specific text>"
}"""


def _rubric_block():
    lines = []
    for name, spec in RUBRIC.items():
        anchors = "; ".join(f"{k}={v}" for k, v in spec["anchors"].items())
        lines.append(f"- {name} (weight {spec['weight']}): {spec['definition']}\n  Anchors: {anchors}")
    return "\n".join(lines)


def build_pointwise_prompt(case, output_text):
    return f"""You are an impartial evaluation judge. Score the MODEL OUTPUT below against the
RUBRIC. Ground every score in the OUTPUT text itself and, when given, the EXPECTED
OUTPUT as a reference. Do not reward length, confident tone, or politeness on their
own — only reward what the rubric defines. If the output is verbose but does not add
support for its claims, do not give it credit for the extra length.

INPUT:
{case['input']}

SYSTEM PROMPT GIVEN TO THE MODEL:
{case.get('system_prompt', '(none)')}

MODEL OUTPUT TO JUDGE:
{output_text}

EXPECTED OUTPUT (reference, may be partial or absent):
{case.get('expected_output', '(none provided — judge reference-free on rubric + input alone)')}

RUBRIC:
{_rubric_block()}

Respond with ONLY a single JSON object matching exactly this schema, no prose before
or after, no markdown fences:
{POINTWISE_SCHEMA}
"""


def build_pairwise_prompt(case, order="AB"):
    out_a, out_b = case["outputs"]["config_a"], case["outputs"]["config_b"]
    first, second = (out_a, out_b) if order == "AB" else (out_b, out_a)
    return f"""You are an impartial evaluation judge comparing two candidate responses to the
SAME input. Decide which is better according to the RUBRIC below. Do not let the
order of presentation, response length, or confident phrasing influence your
decision — judge only what the rubric defines. If one response is longer but does
not add rubric-relevant substance, that length should NOT count in its favor.

INPUT:
{case['input']}

SYSTEM PROMPT GIVEN TO THE MODEL:
{case.get('system_prompt', '(none)')}

RESPONSE "FIRST":
{first}

RESPONSE "SECOND":
{second}

EXPECTED OUTPUT (reference, may be absent):
{case.get('expected_output', '(none provided)')}

RUBRIC:
{_rubric_block()}

Respond with ONLY a single JSON object matching exactly this schema, no prose before
or after, no markdown fences:
{PAIRWISE_SCHEMA}
"""
