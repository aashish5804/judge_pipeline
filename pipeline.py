"""
End-to-end orchestrator.

    test_suite.json (in) -> judging-prompt construction -> judge model call
    -> structured-verdict parse (with malformed-JSON fallback) -> per-case
    aggregation -> suite report (out)

Position-bias mitigation (order swap) sits in bias.position_bias_check,
called as its own pass over the suite. Run:

    python3 pipeline.py --suite test_suite.json --backend auto
"""
import argparse
import json
import random
import sys

from config import DEFAULT_JUDGE, DEFAULT_GENERATOR_A, DEFAULT_GENERATOR_B, self_enhancement_risk
from prompts import build_pointwise_prompt
from llm_client import judge_call, JudgeLogger, _mock_pointwise
from parser import parse_verdict, validate_pointwise_schema
from rubric import RUBRIC
from aggregate import aggregate_suite, compare_configs, weighted_score
from bias import position_bias_check, score_clustering_check, verbosity_probe_result, sycophancy_probe_result
from validate import agreement_with_gold, cohens_kappa_pass_fail, test_retest_flip_rate


def score_config(cases, config_key, judge_cfg, logger, backend):
    """Pointwise-judge every case's output for one config (config_a / config_b)."""
    per_case = []
    for case in cases:
        output_text = case["outputs"].get(config_key)
        if output_text is None:
            continue
        prompt = build_pointwise_prompt(case, output_text)
        result = judge_call(judge_cfg, prompt, tag_hint="pointwise", case=case,
                             output_text=output_text, backend=backend)
        logger.log(f"pointwise:{config_key}:{case['id']}", prompt, result)

        parsed = parse_verdict(result.raw_text)
        entry = {"case_id": case["id"], "parse_ok": parsed.ok, "verdict": None,
                  "recovered_via": parsed.recovered_via, "parse_error": parsed.error}
        if parsed.ok:
            ok_schema, missing = validate_pointwise_schema(parsed.data, RUBRIC.keys())
            if ok_schema:
                entry["verdict"] = parsed.data
            else:
                entry["parse_ok"] = False
                entry["parse_error"] = f"missing criteria: {missing}"
        per_case.append(entry)
    return per_case


def run_adversarial_probes(cases, judge_cfg, logger, backend, per_case_a, per_case_b):
    by_id_a = {v["case_id"]: v for v in per_case_a}
    by_id_b = {v["case_id"]: v for v in per_case_b}
    probes = []
    for case in cases:
        tag = case.get("adversarial")
        if not tag:
            continue
        target = case.get("adversarial_target", "config_b")
        v = (by_id_a if target == "config_a" else by_id_b).get(case["id"])
        verdict = v["verdict"] if v and v["parse_ok"] else None
        if tag in ("verbose_but_wrong", "confidently_wrong"):
            res = verbosity_probe_result(case, verdict)
        else:
            res = sycophancy_probe_result(case, verdict)
        probes.append({"case_id": case["id"], "adversarial_type": tag, "target": target, **res})
    return probes


def run_test_retest(cases, judge_cfg, logger, backend, n_runs=3, sample_ids=None):
    """Re-run pointwise judging on a sample of cases N times to measure how
    often the pass/fail verdict flips run-to-run."""
    sample = [c for c in cases if c["id"] in sample_ids] if sample_ids else cases[:3]
    reruns = {}
    for case in sample:
        scores = []
        output_text = case["outputs"].get("config_a")
        prompt = build_pointwise_prompt(case, output_text)
        for run_idx in range(n_runs):
            if backend == "mock" or backend == "auto":
                raw = _mock_pointwise(case, output_text=output_text)
                # simulate temperature>0 run-to-run jitter
                rng = random.Random(f"{case['id']}-{run_idx}")
                data = json.loads(raw)
                for crit in data["per_criterion"]:
                    jitter = rng.uniform(-0.3, 0.3)
                    data["per_criterion"][crit]["score"] = round(
                        min(5, max(1, data["per_criterion"][crit]["score"] + jitter)), 2)
                result_text = json.dumps(data)
            else:
                result = judge_call(judge_cfg, prompt, tag_hint="pointwise", case=case,
                                     output_text=output_text, backend=backend)
                result_text = result.raw_text
            logger.log(f"retest:{case['id']}:run{run_idx}", prompt,
                       type("R", (), {"raw_text": result_text, "input_tokens": 0,
                                      "output_tokens": 0, "latency_ms": 0, "backend": backend})())
            parsed = parse_verdict(result_text)
            if parsed.ok:
                scores.append(weighted_score(parsed.data["per_criterion"]))
        reruns[case["id"]] = scores
    return test_retest_flip_rate(reruns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="test_suite.json")
    ap.add_argument("--backend", default="auto", choices=["auto", "anthropic", "mock"])
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args()

    with open(args.suite) as f:
        cases = json.load(f)

    logger = JudgeLogger("logs/judge_calls.jsonl")
    judge_cfg = DEFAULT_JUDGE

    risk_a = self_enhancement_risk(judge_cfg, DEFAULT_GENERATOR_A)
    risk_b = self_enhancement_risk(judge_cfg, DEFAULT_GENERATOR_B)

    # 1. Pointwise scoring per config
    per_case_a = score_config(cases, "config_a", judge_cfg, logger, args.backend)
    per_case_b = score_config(cases, "config_b", judge_cfg, logger, args.backend)
    report_a = aggregate_suite(per_case_a)
    report_b = aggregate_suite(per_case_b)

    # 2. Position-bias check (pairwise, both orders) -> also gives win rate for A
    pos_bias = position_bias_check(cases, judge_cfg, backend=args.backend)
    n_a_wins = sum(1 for r in pos_bias["per_case"] if r["final_winner_after_mitigation"] == "config_a")
    n_decided = sum(1 for r in pos_bias["per_case"]
                     if r["final_winner_after_mitigation"] not in ("no_decision_order_disagreement",))
    win_rate_a = round(n_a_wins / n_decided, 3) if n_decided else None

    # 3. Score clustering
    clustering = score_clustering_check(report_a["overall_scores"] + report_b["overall_scores"])

    # 4. Adversarial probes (verbosity / sycophancy / confidently-wrong)
    probes = run_adversarial_probes(cases, judge_cfg, logger, args.backend, per_case_a, per_case_b)

    # 5. Judge validation vs gold labels
    scored_for_gold = []
    for v in per_case_a:
        case = next(c for c in cases if c["id"] == v["case_id"])
        gold = case.get("gold_label", {}).get("config_a_score")
        judge_score = weighted_score(v["verdict"]["per_criterion"]) if v["parse_ok"] else None
        scored_for_gold.append({"case_id": v["case_id"], "judge_score": judge_score, "gold_score": gold})
    for v in per_case_b:
        case = next(c for c in cases if c["id"] == v["case_id"])
        gold = case.get("gold_label", {}).get("config_b_score")
        judge_score = weighted_score(v["verdict"]["per_criterion"]) if v["parse_ok"] else None
        scored_for_gold.append({"case_id": v["case_id"] + "_b", "judge_score": judge_score, "gold_score": gold})

    agreement = agreement_with_gold(scored_for_gold, tolerance=1.0)
    kappa = cohens_kappa_pass_fail(scored_for_gold)

    # 6. Test-retest consistency
    retest = run_test_retest(cases, judge_cfg, logger, args.backend,
                              n_runs=3, sample_ids=["case_01_capital", "case_02_refund_policy", "case_03_math"])

    # 7. A/B comparison
    comparison = compare_configs(report_a, report_b, "config_a", "config_b", win_rate_a=win_rate_a)

    report = {
        "judge_model": judge_cfg.name, "judge_family": judge_cfg.family,
        "generator_a": {"name": DEFAULT_GENERATOR_A.name, "family": DEFAULT_GENERATOR_A.family,
                         "self_enhancement_risk": risk_a},
        "generator_b": {"name": DEFAULT_GENERATOR_B.name, "family": DEFAULT_GENERATOR_B.family,
                         "self_enhancement_risk": risk_b},
        "backend_used": args.backend,
        "suite_report_config_a": report_a,
        "suite_report_config_b": report_b,
        "ab_comparison": comparison,
        "position_bias": {"flip_rate": pos_bias["flip_rate"], "n_cases": pos_bias["n_cases"]},
        "score_clustering": clustering,
        "adversarial_probes": probes,
        "judge_validation": {"agreement_with_gold": agreement, "cohens_kappa": kappa,
                              "test_retest": retest},
        "cost_and_calls": logger.summary(),
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    with open("position_bias_detail.json", "w") as f:
        json.dump(pos_bias, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k not in
                       ("suite_report_config_a", "suite_report_config_b")}, indent=2))


if __name__ == "__main__":
    main()
