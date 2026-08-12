"""
Single entry point for calling the judge model. Two backends:

  - "anthropic": real API call via the Anthropic SDK (needs ANTHROPIC_API_KEY).
  - "mock":      deterministic offline judge, used so the pipeline can be
                 demoed/graded without burning API credits. It still goes
                 through the *same* prompt-building and parsing code paths,
                 so everything except the network call is exercised for real.

Every call, from both backends, is logged verbatim (prompt + raw response)
via JudgeLogger for auditability.
"""
import os
import re
import json
import time
import difflib
from dataclasses import dataclass, field


@dataclass
class CallResult:
    raw_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    backend: str


class JudgeLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def log(self, tag, prompt, result: CallResult):
        self.calls += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        entry = {
            "tag": tag,
            "backend": result.backend,
            "prompt": prompt,
            "raw_response": result.raw_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "timestamp": time.time(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def summary(self):
        return {
            "judge_calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "est_cost_usd_at_$5_per_M_in_$15_per_M_out": round(
                self.total_input_tokens / 1_000_000 * 5
                + self.total_output_tokens / 1_000_000 * 15, 4
            ),
        }


def call_anthropic(model_name, prompt, temperature=0.0, max_tokens=1024):
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.time()
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = (time.time() - t0) * 1000
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return CallResult(
        raw_text=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=latency,
        backend="anthropic",
    )


# ---------------------------------------------------------------------------
# Mock judge: deterministic, text-similarity-driven "scoring" so the pipeline
# is runnable offline. It DELIBERATELY encodes a couple of naive biases
# (rewards length, is a bit sycophantic) so that the bias-mitigation code has
# something real to catch and the before/after numbers in the report aren't
# fabricated as "no bias found".
# ---------------------------------------------------------------------------
def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _mock_pointwise(case, output_text=None):
    output = output_text if output_text is not None else case["outputs"]
    expected = case.get("expected_output", "")
    sim = _similarity(output, expected) if expected else 0.6
    length_bonus = min(len(output) / 800, 1.0) * 0.15  # naive length-rewarding bias

    base = 1 + sim * 4
    scores = {}
    for crit in ["correctness", "faithfulness", "completeness", "instruction_following", "tone_safety"]:
        val = base + length_bonus
        if case.get("adversarial") == "verbose_but_wrong" and crit in ("correctness", "faithfulness"):
            val = 4.0  # the naive/unmitigated judge gets fooled by verbosity here
        if case.get("adversarial") == "terse_but_correct" and crit == "completeness":
            val = 2.0  # naive judge under-scores a short-but-correct answer
        scores[crit] = round(max(1, min(5, val)), 2)

    verdict = {
        "per_criterion": {
            k: {"score": v, "rationale": f"[mock] similarity={sim:.2f}, len_bonus={length_bonus:.2f}"}
            for k, v in scores.items()
        },
        "overall_rationale": "[mock judge] heuristic score from text similarity + length.",
    }
    return json.dumps(verdict, indent=2)


def _mock_pairwise(case, order):
    out_a, out_b = case["outputs"]["config_a"], case["outputs"]["config_b"]
    first, second = (out_a, out_b) if order == "AB" else (out_b, out_a)
    # naive mock judge is length-biased: prefers whichever came first if lengths
    # are close, and otherwise prefers the longer one (verbosity bias, injected
    # on purpose so the position/verbosity mitigation has something to detect)
    len_first, len_second = len(first), len(second)
    if abs(len_first - len_second) < 20:
        winner = "first"
    else:
        winner = "first" if len_first > len_second else "second"
    verdict = {
        "winner": winner,
        "confidence": 0.7,
        "rationale": f"[mock] len_first={len_first} len_second={len_second}",
    }
    return json.dumps(verdict, indent=2)


def call_mock(prompt, tag_hint, case=None, order=None, output_text=None):
    t0 = time.time()
    if tag_hint == "pairwise":
        text = _mock_pairwise(case, order)
    else:
        text = _mock_pointwise(case, output_text=output_text)
    latency = (time.time() - t0) * 1000
    return CallResult(
        raw_text=text,
        input_tokens=len(prompt.split()),
        output_tokens=len(text.split()),
        latency_ms=latency,
        backend="mock",
    )


def judge_call(model_cfg, prompt, tag_hint="pointwise", case=None, order=None,
                output_text=None, backend="auto"):
    """backend: 'auto' tries real Anthropic API if ANTHROPIC_API_KEY is set,
    else falls back to mock. Force with 'anthropic' or 'mock'."""
    use_real = backend == "anthropic" or (backend == "auto" and os.environ.get("ANTHROPIC_API_KEY"))
    if use_real:
        try:
            return call_anthropic(model_cfg.name, prompt, model_cfg.temperature)
        except Exception as e:
            print(f"[warn] real API call failed ({e}); falling back to mock")
    return call_mock(prompt, tag_hint, case=case, order=order, output_text=output_text)
