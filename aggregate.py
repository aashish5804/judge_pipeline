from rubric import RUBRIC, PASS_THRESHOLD


def weighted_score(per_criterion):
    total = 0.0
    for name, spec in RUBRIC.items():
        s = per_criterion.get(name, {}).get("score")
        if s is None:
            continue
        total += s * spec["weight"]
    return round(total, 3)


def aggregate_suite(per_case_verdicts):
    """per_case_verdicts: list of {case_id, verdict(dict|None), parse_ok}
    Returns a suite report: pass rate, mean overall score, mean per-criterion."""
    scored = [v for v in per_case_verdicts if v["parse_ok"]]
    n = len(per_case_verdicts)
    n_scored = len(scored)

    overall_scores = [weighted_score(v["verdict"]["per_criterion"]) for v in scored]
    passed = [s for s in overall_scores if s >= PASS_THRESHOLD]

    per_criterion_means = {}
    for crit in RUBRIC:
        vals = [v["verdict"]["per_criterion"].get(crit, {}).get("score")
                 for v in scored if crit in v["verdict"].get("per_criterion", {})]
        vals = [x for x in vals if x is not None]
        per_criterion_means[crit] = round(sum(vals) / len(vals), 2) if vals else None

    return {
        "n_cases": n,
        "n_parsed_ok": n_scored,
        "n_parse_failures": n - n_scored,
        "pass_rate": round(len(passed) / n_scored, 3) if n_scored else None,
        "mean_overall_score": round(sum(overall_scores) / n_scored, 3) if n_scored else None,
        "per_criterion_means": per_criterion_means,
        "overall_scores": overall_scores,
    }


def compare_configs(report_a, report_b, name_a="config_a", name_b="config_b", win_rate_a=None):
    winner = None
    if report_a["mean_overall_score"] is not None and report_b["mean_overall_score"] is not None:
        if report_a["mean_overall_score"] > report_b["mean_overall_score"]:
            winner = name_a
        elif report_b["mean_overall_score"] > report_a["mean_overall_score"]:
            winner = name_b
        else:
            winner = "tie"
    return {
        name_a: {"pass_rate": report_a["pass_rate"], "mean_score": report_a["mean_overall_score"]},
        name_b: {"pass_rate": report_b["pass_rate"], "mean_score": report_b["mean_overall_score"]},
        "pairwise_win_rate_config_a": win_rate_a,
        "declared_winner": winner,
    }
