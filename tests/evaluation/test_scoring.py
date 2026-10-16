"""Scoring and aggregates for the extraction model comparison."""

import pytest

from evaluation.extraction import scoring

EXPECTED = {
    "payor": "Aetna",
    "cpt_codes": ["97110", "97140"],
    "icd_codes": ["M25.561", "M54.5"],
}
PRICING = {
    "as_of": "2026-10-16",
    "source": "test",
    "currency": "USD",
    "unit": "per 1K tokens",
    "models": {"model-a": {"input": 0.003, "output": 0.015}},
}


def make_trial(
    result=None,
    status="ok",
    calls=None,
    row="model-a",
    case_id="case-1.pdf",
    fmt="pdf",
    trial=1,
    error=None,
):
    if calls is None:
        calls = [call()]
    return {
        "row": row,
        "model_id": row,
        "sampling": {"temperature": 0.1, "top_p": None},
        "sampling_note": None,
        "case_id": case_id,
        "format": fmt,
        "trial": trial,
        "status": status,
        "error": error,
        "result": result,
        "model_calls": calls,
        "duration_ms": 1500,
    }


def call(
    model_id="model-a", tokens_in=1000, tokens_out=200, latency=900, stop="end_turn"
):
    return {
        "model_id": model_id,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "latency_ms": latency,
        "stop_reason": stop,
    }


def result(
    payor="Aetna Better Health", cpt=("97110", "97140"), icd=("M25.561", "M54.5")
):
    return {"payor_name": payor, "cpt_codes": list(cpt), "icd_codes": list(icd)}


def test_ok_trial_scores_every_field():
    scored = scoring.score_trial(make_trial(result()), EXPECTED, PRICING)

    assert scored["payor_correct"] is True
    assert scored["cpt_precision"] == scored["cpt_recall"] == 1.0
    assert scored["icd_precision"] == scored["icd_recall"] == 1.0
    assert scored["failed"] is False and scored["truncated"] is False
    assert scored["input_tokens"] == 1000 and scored["output_tokens"] == 200
    assert scored["latency_ms"] == 900
    assert scored["cost_usd"] == pytest.approx(0.003 + 0.2 * 0.015)


def test_partial_codes_give_precision_and_recall():
    scored = scoring.score_trial(
        make_trial(result(cpt=["97110", "99999"], icd=["M54.5"])), EXPECTED, PRICING
    )

    assert scored["cpt_precision"] == 0.5 and scored["cpt_recall"] == 0.5
    assert scored["icd_precision"] == 1.0 and scored["icd_recall"] == 0.5


def test_wrong_payor_family_is_incorrect():
    scored = scoring.score_trial(make_trial(result(payor="Humana")), EXPECTED)

    assert scored["payor_correct"] is False


def test_codes_are_normalized_and_icd_ignores_dots():
    scored = scoring.score_trial(
        make_trial(result(cpt=[" 97110 ", "97140"], icd=["m25561", "M54.5 "])),
        EXPECTED,
    )

    assert scored["cpt_precision"] == scored["cpt_recall"] == 1.0
    assert scored["icd_precision"] == scored["icd_recall"] == 1.0


def test_empty_predictions_score_zero():
    scored = scoring.score_trial(
        make_trial(result(payor="", cpt=[], icd=[])), EXPECTED, PRICING
    )

    assert scored["payor_correct"] is False
    assert scored["cpt_precision"] == scored["cpt_recall"] == 0.0
    assert scored["icd_precision"] == scored["icd_recall"] == 0.0
    assert scored["failed"] is False


def test_failed_trial_counts_against_model_but_keeps_usage():
    calls = [call(), call(tokens_in=500, tokens_out=100, latency=400)]
    scored = scoring.score_trial(
        make_trial(None, status="error", calls=calls, error="ValueError: bad JSON"),
        EXPECTED,
        PRICING,
    )

    assert scored["failed"] is True
    assert scored["payor_correct"] is False
    assert scored["cpt_precision"] == scored["icd_recall"] == 0.0
    assert scored["input_tokens"] == 1500 and scored["output_tokens"] == 300
    assert scored["latency_ms"] == 1300
    assert scored["cost_usd"] == pytest.approx(1.5 * 0.003 + 0.3 * 0.015)
    assert scored["error"] == "ValueError: bad JSON"


def test_ok_status_without_result_is_failed():
    scored = scoring.score_trial(make_trial(None), EXPECTED)

    assert scored["failed"] is True


def test_truncated_trial_is_flagged_and_scores_zero():
    scored = scoring.score_trial(
        make_trial(result(), calls=[call(stop="max_tokens")]), EXPECTED, PRICING
    )

    assert scored["truncated"] is True
    assert scored["stop_reasons"] == ["max_tokens"]
    assert scored["payor_correct"] is False
    assert scored["cpt_recall"] == 0.0


def test_missing_price_gives_no_cost():
    scored = scoring.score_trial(
        make_trial(result(), calls=[call(model_id="unpriced")]), EXPECTED, PRICING
    )

    assert scored["cost_usd"] is None
    assert scoring.score_trial(make_trial(result()), EXPECTED)["cost_usd"] is None


def test_missing_token_count_gives_no_cost_or_total():
    scored = scoring.score_trial(
        make_trial(result(), calls=[call(tokens_out=None)]), EXPECTED, PRICING
    )

    assert scored["cost_usd"] is None
    assert scored["output_tokens"] is None


def test_trial_without_model_calls_costs_nothing():
    scored = scoring.score_trial(
        make_trial(result(), calls=[], fmt="fhir"), EXPECTED, PRICING
    )

    assert scored["cost_usd"] == 0.0
    assert scored["latency_ms"] is None
    assert scored["input_tokens"] == 0


def test_nearest_rank_percentiles():
    values = list(range(1, 21))  # 1..20

    assert scoring.percentile(values, 50) == 10
    assert scoring.percentile(values, 95) == 19
    assert scoring.percentile([7], 95) == 7
    assert scoring.percentile([], 50) is None


def test_summarize_splits_pdf_and_fhir():
    trials = [
        make_trial(result(), case_id="a.pdf", trial=1, calls=[call(latency=100)]),
        make_trial(result(), case_id="a.pdf", trial=2, calls=[call(latency=300)]),
        make_trial(None, status="error", case_id="b.pdf", calls=[call(latency=200)]),
        make_trial(result(), case_id="a.pdf", calls=[call(stop="max_tokens")]),
        make_trial(result(), case_id="a.fhir", fmt="fhir", calls=[]),
    ]
    scored = [scoring.score_trial(t, EXPECTED, PRICING) for t in trials]

    summary = scoring.summarize(scored)
    pdf, fhir = summary["model-a"]["pdf"], summary["model-a"]["fhir"]

    assert summary["model-a"]["model_id"] == "model-a"
    assert pdf["n_trials"] == 4 and pdf["n_cases"] == 2
    assert pdf["payor_accuracy"] == 0.5
    assert pdf["cpt_recall"] == 0.5
    assert pdf["failure_rate"] == 0.25
    assert pdf["truncation_rate"] == 0.25
    assert pdf["latency_p50_ms"] == 200
    assert pdf["latency_p95_ms"] == 900
    per_trial = 0.003 + 0.2 * 0.015
    assert pdf["total_cost_usd"] == pytest.approx(4 * per_trial)
    assert pdf["mean_cost_per_case_usd"] == pytest.approx(per_trial)
    assert fhir["n_trials"] == 1 and fhir["payor_accuracy"] == 1.0
    assert fhir["latency_p50_ms"] is None
    assert fhir["total_cost_usd"] == 0.0


def test_summarize_cost_is_none_when_any_trial_is_unpriced():
    trials = [
        make_trial(result()),
        make_trial(result(), trial=2, calls=[call(model_id="unpriced")]),
    ]
    summary = scoring.summarize(
        [scoring.score_trial(t, EXPECTED, PRICING) for t in trials]
    )

    assert summary["model-a"]["pdf"]["total_cost_usd"] is None
    assert summary["model-a"]["pdf"]["mean_cost_per_case_usd"] is None
    assert "fhir" not in summary["model-a"]
