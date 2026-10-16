"""Field-level scoring and per-row aggregates for the extraction comparison.

A trial is one extraction of one case by one row (model and sampling). Scores
are payor exact match (as payor family) and set precision and recall for CPT
and ICD-10 codes. Failed or truncated trials score zero on every field.
"""

import math
from typing import Any, Dict, Iterable, List, Optional

from evaluation.ground_truth import payor_family

FORMATS = ("pdf", "fhir")


def normalize_cpt(code: Any) -> str:
    return str(code).strip().upper()


def normalize_icd(code: Any) -> str:
    """ICD-10 codes match ignoring dots (M54.5 == M545)"""
    return str(code).strip().upper().replace(".", "")


def precision_recall(predicted: Iterable[str], expected: Iterable[str]):
    """Set precision and recall; an empty prediction scores 0.0 against codes"""
    predicted, expected = set(predicted), set(expected)
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else (0.0 if expected else 1.0)
    recall = hits / len(expected) if expected else 1.0
    return precision, recall


def trial_cost(
    model_calls: List[Dict[str, Any]], pricing: Optional[Dict[str, Any]]
) -> Optional[float]:
    """Cost in USD from per-1K token prices; None if any price or count is missing"""
    prices = (pricing or {}).get("models", {})
    total = 0.0
    for call in model_calls:
        price = prices.get(call.get("model_id"))
        tokens_in, tokens_out = call.get("input_tokens"), call.get("output_tokens")
        if price is None or tokens_in is None or tokens_out is None:
            return None
        total += tokens_in / 1000 * price["input"] + tokens_out / 1000 * price["output"]
    return total


def _sum(values: List[Optional[int]]) -> Optional[int]:
    """Sum of counts; None if any count is missing"""
    if any(value is None for value in values):
        return None
    return sum(values)


def score_trial(
    trial: Dict[str, Any],
    expected: Dict[str, Any],
    pricing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score one trial record against a case's expected values"""
    calls = trial.get("model_calls") or []
    result = trial.get("result")
    stop_reasons = [call.get("stop_reason") for call in calls]
    failed = trial.get("status") != "ok" or result is None
    truncated = "max_tokens" in stop_reasons

    scored = {
        key: trial.get(key)
        for key in (
            "row",
            "model_id",
            "sampling",
            "sampling_note",
            "case_id",
            "format",
            "trial",
            "status",
            "error",
            "duration_ms",
        )
    }
    scored.update(
        failed=failed,
        truncated=truncated,
        stop_reasons=stop_reasons,
        input_tokens=_sum([call.get("input_tokens") for call in calls]),
        output_tokens=_sum([call.get("output_tokens") for call in calls]),
        latency_ms=(
            _sum([call.get("latency_ms") for call in calls]) if calls else None
        ),
        cost_usd=trial_cost(calls, pricing),
        payor_correct=False,
        cpt_precision=0.0,
        cpt_recall=0.0,
        icd_precision=0.0,
        icd_recall=0.0,
    )
    if failed or truncated:
        return scored

    payor = payor_family(result.get("payor_name") or "")
    scored["payor_correct"] = payor is not None and payor == expected["payor"]
    for field, normalize in (("cpt", normalize_cpt), ("icd", normalize_icd)):
        predicted = {normalize(c) for c in result.get(f"{field}_codes") or []}
        truth = {normalize(c) for c in expected[f"{field}_codes"]}
        precision, recall = precision_recall(predicted, truth)
        scored[f"{field}_precision"], scored[f"{field}_recall"] = precision, recall
    return scored


def percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile; None for no values"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _aggregate(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trials)
    latencies = [t["latency_ms"] for t in trials if t["latency_ms"] is not None]
    costs = [t["cost_usd"] for t in trials]
    priced = all(cost is not None for cost in costs)
    return {
        "n_trials": n,
        "n_cases": len({t["case_id"] for t in trials}),
        "payor_accuracy": _mean([1.0 if t["payor_correct"] else 0.0 for t in trials]),
        "cpt_precision": _mean([t["cpt_precision"] for t in trials]),
        "cpt_recall": _mean([t["cpt_recall"] for t in trials]),
        "icd_precision": _mean([t["icd_precision"] for t in trials]),
        "icd_recall": _mean([t["icd_recall"] for t in trials]),
        "failure_rate": _mean([1.0 if t["failed"] else 0.0 for t in trials]),
        "truncation_rate": _mean([1.0 if t["truncated"] else 0.0 for t in trials]),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        # One trial is one extraction of one case, so cost per case is per trial
        "mean_cost_per_case_usd": _mean(costs) if priced else None,
        "total_cost_usd": sum(costs) if priced else None,
    }


def summarize(scored_trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-row aggregates, split by format.

    Returns {row: {"model_id", "sampling", "sampling_note", "pdf": {...},
    "fhir": {...}}}; a format key is present only if the row has such trials.
    """
    summary: Dict[str, Dict[str, Any]] = {}
    for trial in scored_trials:
        entry = summary.setdefault(
            trial["row"],
            {
                "model_id": trial["model_id"],
                "sampling": trial["sampling"],
                "sampling_note": trial["sampling_note"],
            },
        )
        entry.setdefault(trial["format"], []).append(trial)
    for entry in summary.values():
        for fmt in FORMATS:
            if fmt in entry:
                entry[fmt] = _aggregate(entry[fmt])
    return summary
