# Content to paste into the submission template

Sections 2.1–2.3.4 below are filled from the actual `report.json` produced by
`python3 pipeline.py --suite test_suite.json --backend mock`. Section 2.4 gives
you talking points, not final prose — write those in your own words, the
template says so explicitly. Section 2.5 tells you exactly what to screenshot.

---

## 2.1 Architecture Diagram / Flowchart

Text version — draw this as boxes/arrows in the doc:

```
test_suite.json (JSON, list of cases)
        |
        v
build_pointwise_prompt() / build_pairwise_prompt()   [prompts.py]
   - embeds full weighted rubric + 1-5 anchors
   - explicit "do not reward length/tone" instruction  <-- verbosity/sycophancy
                                                             mitigation lives HERE
        |
        v
judge_call()  [llm_client.py]  --calls-->  Anthropic API (or mock judge)
        |
        v
parse_verdict()  [parser.py]
   1. json.loads direct
   2. strip ```json fences, retry        <-- malformed-JSON fallback
   3. brace-match extraction, retry
   4. else: parse_ok=False, never invented
        |
        v
   +---------------------------+
   | position_bias_check()     |  <-- runs EVERY pairwise case in BOTH
   | [bias.py]                 |      orders (A-first / B-first); only
   |  declares a winner when   |      declares a winner if both orders
   |  both orders agree        |      agree -- this is where the A/B
   +---------------------------+      order-swap mitigation sits
        |
        v
aggregate_suite()  [aggregate.py]  -> per-case scores -> pass rate, mean
                                       score, per-criterion means
        |
        v
compare_configs()  -> A/B suite report + declared winner
        |
        v
report.json  +  position_bias_detail.json  +  logs/judge_calls.jsonl (audit)
```

## 2.2 Setup & Run Instructions

**Prerequisites:** Python 3.10+, `pip install anthropic` (only if using the real
API backend), optional `ANTHROPIC_API_KEY`.

**Environment variables:** `ANTHROPIC_API_KEY` (real judge calls; unset = mock
judge backend, no secret values committed anywhere).

**Install:**
```bash
git clone <this-repo>
cd judge_pipeline
pip install anthropic --break-system-packages
```

**Run a suite -> produce a report:**
```bash
python3 pipeline.py --suite test_suite.json --backend mock
```

**Run an A/B comparison:** the same command already runs both `config_a` and
`config_b` from the suite file and writes the comparison into
`report.json["ab_comparison"]`.

**Judge & generator configured independently?** Yes — `config.py` defines
`DEFAULT_JUDGE`, `DEFAULT_GENERATOR_A`, `DEFAULT_GENERATOR_B` as separate
objects (own `name`/`family`/`temperature`). `self_enhancement_risk()` compares
families and writes the flag into the report instead of assuming it away.

---

## 2.3 Evaluation Results

### 2.3.1 Judging mode & rubric

**Judging mode used:** Pointwise, reference-based (primary — produces the
per-case scores used for the suite report) **plus** pairwise A-vs-B
(used specifically for the A/B comparison and the position-bias check).

| Criterion | Definition / score anchors | Weight |
|---|---|---|
| Correctness | Are factual claims/conclusions accurate vs. expected_output? 1=major error … 5=fully correct | 0.30 |
| Faithfulness | Grounded in input/context, no hallucination. 1=fabricates facts … 5=fully traceable | 0.20 |
| Completeness | Addresses every part of the request. 1=misses most … 5=fully addresses | 0.20 |
| Instruction-following | Obeys explicit system_prompt constraints. 1=ignores … 5=precise | 0.20 |
| Tone / safety | Appropriate tone, no unsafe content. 1=unsafe … 5=fully appropriate | 0.10 |

### 2.3.2 Bias handling

| Bias | Mitigation implemented (in code) | Metric & result (before -> after) |
|---|---|---|
| Position (A/B order) | Every pairwise case run in BOTH orders (`bias.position_bias_check`); a winner is only declared if both orders agree, else `no_decision_order_disagreement` | Raw order-driven flip rate: **37.5%** (3/8 cases changed winner purely from order) → after mitigation, those 3 cases are withheld as `no_decision` instead of silently picking one |
| Verbosity / length | Rubric prompt explicitly instructs the judge not to reward length; padded-answer adversarial probe (`probe_verbose_but_wrong`) included in suite | Naive/unmitigated judge was **fooled** on the verbose-but-wrong probe (correctness=4.0, faithfulness=4.0 despite wrong year) — this is the raw failure mode the mitigation targets |
| Self-enhancement | Judge (`claude-opus-4-8`, family=anthropic) configured independently from both generators (family=openai) | `self_enhancement_risk` = **False** for both configs (families differ) |
| Sycophancy / style | Per-criterion grounding required in rationale; confidently-wrong probe (`probe_confidently_wrong`) included | Judge was **not fooled**: correctness/faithfulness scored 1.75 despite confident tone |
| Score clustering | Weighted-mean scoring across 5 criteria (not a single 1-5 number) + explicit anchors per score | Std dev across 16 scored outputs = **1.101** → not clustered (`clustered: false`) |

### 2.3.3 Judge validation

| Method | Result | Notes |
|---|---|---|
| Agreement with human/gold | **37.5%** within ±1 point | n=16 (8 cases × 2 configs) |
| Cohen's kappa (pass/fail @ 3.5 threshold) | **0.20** | observed agreement 56.2%, n=16 — low kappa, expected: mock judge is a naive text-similarity heuristic, not a real LLM |
| Test-retest flip rate | **0.0%** (0 flips / 6 reruns) | 3 cases × 3 reruns, ±0.3 simulated jitter |
| Adversarial probe outcome | 1 of 3 fooled | verbose-but-wrong: fooled; terse-but-correct: not fooled; confidently-wrong: not fooled |

### 2.3.4 A/B comparison & declared winner

| Config | Pass rate | Mean score | Win rate |
|---|---|---|---|
| Config A — gpt-4o outputs | 0.50 | 3.668 | 0.20 (pairwise, mitigated) |
| Config B — gpt-4o-mini outputs | 0.125 | 2.562 | — |

**Declared winner + justification:** Config A — higher pass rate (50% vs
12.5%) and higher mean weighted score (3.67 vs 2.56) even after the
position-bias mitigation discounted 3 disputed cases to "no decision."

---

## 2.4 Design Decisions & Trade-offs — talking points (write these yourself)

- **Judging mode:** pointwise reference-based for per-case scoring (lets you
  report pass rate / mean score per criterion), pairwise for A/B + position
  bias specifically because position bias is *only* observable when the judge
  sees two things in an order.
- **Malformed JSON:** 3-tier fallback (direct parse → strip fences → brace-match)
  before giving up; a genuinely unparseable response is marked failed, never
  silently scored — explain why that matters (a fabricated score is worse than
  a visible failure).
- **Judge model family vs generator family:** picked a judge from a different
  family than either generator on purpose; explain the self-enhancement risk
  this avoids.
- **Which bias worried you most:** the numbers above show position bias had
  the highest raw flip rate (37.5%) and verbosity actually fooled the judge on
  one probe — pick whichever you find most concerning and say why, referencing
  the actual numbers.
- **Would you let this judge gate a release:** talk about the low kappa (0.20)
  from the *mock* backend — that's a genuine reason not to trust it blind, and
  a good hook for "keep a human in the loop for anything near the pass
  threshold." If you switch to `--backend anthropic` with a real key, rerun
  and cite the real kappa instead.

## 2.5 Subtask Evidence & Reflection

The template says this section is mandatory, must be written without AI
assistance, and requires disclosing which AI tools you used — so I'm not
writing it for you. Here's exactly what to pull for the screenshots:

- **Auditable prompt + response screenshot:** open `logs/judge_calls.jsonl`,
  take any one line, it has `prompt` and `raw_response` verbatim.
- **Position-bias result screenshot:** open `position_bias_detail.json` —
  shows both orders' raw calls and the flip per case.
- **Adversarial probe outcome:** `report.json["adversarial_probes"]` has all
  three probes with their fooled/not-fooled verdicts.
- **AI usage disclosure:** be specific and honest — e.g. "used Claude to build
  the pipeline code (rubric, prompts, parser, bias checks, aggregation); wrote
  the design-decision and reflection answers myself."
