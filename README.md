# LLM-as-Judge Evaluation Pipeline

Turns `{ input, system_prompt, model_output, expected_output?, criteria? }` test
cases into a structured quality verdict, and measures the judge's own biases
instead of trusting it blindly.

## Prerequisites
- Python 3.10+
- `pip install anthropic` (only needed if you use the real API backend)
- Optional: an `ANTHROPIC_API_KEY` environment variable

## Environment variables
- `ANTHROPIC_API_KEY` — real judge calls. If unset, the pipeline runs on a
  deterministic **mock judge** (see "Backends" below) so it's fully runnable
  offline / without spending credits.

## Install
```bash
git clone <this-repo>
cd judge_pipeline
pip install anthropic --break-system-packages   # skip if you only use --backend mock
```

## Run a suite -> produce a report
```bash
python3 pipeline.py --suite test_suite.json --backend mock
# or, with a real key exported:
export ANTHROPIC_API_KEY=sk-...
python3 pipeline.py --suite test_suite.json --backend anthropic
```
Outputs:
- `report.json` — full suite report (per-config scores, A/B comparison, bias
  metrics, judge validation, cost).
- `position_bias_detail.json` — every pairwise call in both orders, raw.
- `logs/judge_calls.jsonl` — every judge prompt + raw response, one line per
  call, for audit/replay.

## Backends
| Backend     | What it does |
|---|---|
| `mock`      | Deterministic offline judge. Deliberately encodes a couple of naive biases (rewards length, is order-sensitive) so the bias-mitigation code has something real to catch — the report numbers are genuine detections, not fabricated "no bias found" placeholders. |
| `anthropic` | Real call to the Anthropic Messages API using the configured judge model. |
| `auto`      | Uses `anthropic` if `ANTHROPIC_API_KEY` is set, else falls back to `mock`. |

## Judge & generator configured independently
`config.py` defines `DEFAULT_JUDGE`, `DEFAULT_GENERATOR_A`, `DEFAULT_GENERATOR_B`
as separate objects with their own `name`/`family`/`temperature`. `pipeline.py`
never derives the judge from the generator config. `self_enhancement_risk()`
compares `judge.family` to `generator.family` and the flag is written into
`report.json` rather than assumed away.

## File map
- `rubric.py` — explicit weighted rubric with 1-5 anchors per criterion.
- `prompts.py` — pointwise (reference-based) and pairwise (A-vs-B) prompt builders.
- `parser.py` — structured-verdict parsing with malformed-JSON recovery.
- `bias.py` — position-bias order-swap check, score-clustering check, probe scoring.
- `aggregate.py` — per-case -> suite report, A/B comparison.
- `validate.py` — agreement with gold labels, Cohen's kappa, test-retest flip rate.
- `llm_client.py` — real Anthropic call + mock judge + call logging.
- `pipeline.py` — orchestrator / CLI.
- `test_suite.json` — 5 normal cases + 3 adversarial probes (verbose-but-wrong,
  terse-but-correct, confidently-wrong), each with gold labels for validation.
