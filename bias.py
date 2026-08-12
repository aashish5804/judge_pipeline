"""
Concrete mitigations + measurements for the five biases called out in the
assignment. Each function returns a small dict of *evidence*, not a claim.
"""
import statistics
from prompts import build_pairwise_prompt
from llm_client import judge_call
from parser import parse_verdict


def position_bias_check(cases, judge_cfg, backend="auto"):
    """For every case with two configs, run the pairwise judge in BOTH
    orders (A-first and B-first) and see whether the winner flips purely
    because of presentation order. Mitigation = average the two orders /
    require agreement before declaring a winner; report the flip rate as
    the measured evidence."""
    results = []
    flips = 0
    for case in cases:
        if "config_a" not in case.get("outputs", {}):
            continue
        r_ab = judge_call(judge_cfg, build_pairwise_prompt(case, "AB"),
                           tag_hint="pairwise", case=case, order="AB", backend=backend)
        r_ba = judge_call(judge_cfg, build_pairwise_prompt(case, "BA"),
                           tag_hint="pairwise", case=case, order="BA", backend=backend)

        v_ab = parse_verdict(r_ab.raw_text)
        v_ba = parse_verdict(r_ba.raw_text)

        # normalize "first"/"second" back to config_a/config_b for each order
        def normalize(order, winner):
            if winner == "tie":
                return "tie"
            if order == "AB":
                return "config_a" if winner == "first" else "config_b"
            return "config_a" if winner == "second" else "config_b"

        w_ab = normalize("AB", v_ab.data.get("winner")) if v_ab.ok else "PARSE_FAIL"
        w_ba = normalize("BA", v_ba.data.get("winner")) if v_ba.ok else "PARSE_FAIL"

        flipped = w_ab != w_ba
        if flipped:
            flips += 1
        # mitigation: only declare a winner when both orders agree; else "no_decision"
        final = w_ab if w_ab == w_ba else "no_decision_order_disagreement"

        results.append({
            "case_id": case["id"], "winner_AB_order": w_ab, "winner_BA_order": w_ba,
            "flipped": flipped, "final_winner_after_mitigation": final,
            "raw_calls": [
                {"order": "AB", "prompt": build_pairwise_prompt(case, "AB"), "response": r_ab.raw_text},
                {"order": "BA", "prompt": build_pairwise_prompt(case, "BA"), "response": r_ba.raw_text},
            ],
        })
    flip_rate = flips / len(results) if results else 0.0
    return {"per_case": results, "flip_rate": round(flip_rate, 3), "n_cases": len(results)}


def score_clustering_check(all_scores):
    """all_scores: flat list of weighted overall scores across the suite.
    Low std dev means the judge is clustering everything near one value
    (e.g. always 4/5) instead of discriminating. Mitigation is few-shot
    anchors in the rubric prompt / switching to pairwise, which we've
    already put in prompts.py."""
    if len(all_scores) < 2:
        return {"std_dev": None, "clustered": None}
    sd = statistics.pstdev(all_scores)
    return {"std_dev": round(sd, 3), "clustered": sd < 0.4, "n": len(all_scores)}


def verbosity_probe_result(case, verdict_data):
    """Given the parsed verdict for a known verbose-but-wrong adversarial
    case, check whether correctness/faithfulness were (correctly) scored
    low despite the length. fooled=True means the bias mitigation did NOT
    work for this case."""
    if not verdict_data or "per_criterion" not in verdict_data:
        return {"fooled": None, "reason": "unparseable verdict"}
    corr = verdict_data["per_criterion"].get("correctness", {}).get("score")
    faith = verdict_data["per_criterion"].get("faithfulness", {}).get("score")
    fooled = (corr is not None and corr >= 4) or (faith is not None and faith >= 4)
    return {"fooled": bool(fooled), "correctness": corr, "faithfulness": faith}


def sycophancy_probe_result(case, verdict_data):
    """Confidently-wrong probe: the answer is written in a very confident
    tone but is factually wrong. A sycophantic/style-biased judge rewards
    the confident tone; a mitigated judge still catches the error via
    per-criterion grounding."""
    return verbosity_probe_result(case, verdict_data)  # same check, different case tag
