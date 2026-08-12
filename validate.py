import statistics
from rubric import PASS_THRESHOLD


def agreement_with_gold(scored_cases, tolerance=1.0):
    """scored_cases: list of {case_id, judge_score, gold_score}. Agreement =
    fraction within `tolerance` points on the 1-5 scale. Also returns a
    Pearson correlation as a second, stricter signal."""
    pairs = [(c["judge_score"], c["gold_score"]) for c in scored_cases
             if c["judge_score"] is not None and c["gold_score"] is not None]
    if not pairs:
        return {"agreement_rate": None, "correlation": None, "n": 0}
    within = sum(1 for j, g in pairs if abs(j - g) <= tolerance)
    agreement = within / len(pairs)

    if len(pairs) >= 2 and len(set(p[0] for p in pairs)) > 1 and len(set(p[1] for p in pairs)) > 1:
        js = [p[0] for p in pairs]
        gs = [p[1] for p in pairs]
        try:
            corr = statistics.correlation(js, gs)
        except statistics.StatisticsError:
            corr = None
    else:
        corr = None

    return {"agreement_rate": round(agreement, 3), "correlation": round(corr, 3) if corr else None,
            "n": len(pairs), "tolerance": tolerance}


def cohens_kappa_pass_fail(scored_cases):
    """Binarize judge_score and gold_score at PASS_THRESHOLD, then compute
    Cohen's kappa on the resulting pass/fail agreement — a chance-corrected
    agreement measure, stricter than raw agreement rate."""
    pairs = [(c["judge_score"] >= PASS_THRESHOLD, c["gold_score"] >= PASS_THRESHOLD)
              for c in scored_cases if c["judge_score"] is not None and c["gold_score"] is not None]
    n = len(pairs)
    if n == 0:
        return {"kappa": None, "n": 0}

    po = sum(1 for j, g in pairs if j == g) / n

    p_judge_pass = sum(1 for j, _ in pairs if j) / n
    p_gold_pass = sum(1 for _, g in pairs if g) / n
    pe = (p_judge_pass * p_gold_pass) + ((1 - p_judge_pass) * (1 - p_gold_pass))

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)
    return {"kappa": round(kappa, 3), "observed_agreement": round(po, 3), "n": n}


def test_retest_flip_rate(reruns):
    """reruns: list of per-case-score lists, one list per case, each
    containing N re-run scores. Flip = pass/fail label differs from the
    first run's label at PASS_THRESHOLD. Returns overall flip rate."""
    total, flips = 0, 0
    per_case = []
    for case_id, scores in reruns.items():
        if len(scores) < 2:
            continue
        first_label = scores[0] >= PASS_THRESHOLD
        case_flips = sum(1 for s in scores[1:] if (s >= PASS_THRESHOLD) != first_label)
        total += len(scores) - 1
        flips += case_flips
        per_case.append({"case_id": case_id, "scores": scores,
                          "flips": case_flips, "runs": len(scores)})
    return {"per_case": per_case, "flip_rate": round(flips / total, 3) if total else None}
