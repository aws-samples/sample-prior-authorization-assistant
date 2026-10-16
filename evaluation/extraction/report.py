"""Markdown and CSV reports for the extraction model comparison."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.ground_truth import load_cases

CSV_FIELDS = (
    "row",
    "model_id",
    "temperature",
    "top_p",
    "sampling_note",
    "case_id",
    "format",
    "trial",
    "status",
    "failed",
    "truncated",
    "payor_correct",
    "cpt_precision",
    "cpt_recall",
    "icd_precision",
    "icd_recall",
    "stop_reasons",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "duration_ms",
    "cost_usd",
    "error",
)


def fmt_ratio(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else str(int(round(value)))


def fmt_usd(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def fmt_int(value: Optional[int]) -> str:
    return "n/a" if value is None else str(value)


def fmt_sampling(sampling: Optional[Dict[str, Any]], note: Optional[str]) -> str:
    """'temperature 0.1', 'temperature 0.1, top_p 0.9' or 'model default'"""
    sampling = sampling or {}
    parts = [
        f"{name} {sampling[name]}"
        for name in ("temperature", "top_p")
        if sampling.get(name) is not None
    ]
    text = ", ".join(parts) or "model default"
    return f"{text} ({note})" if note else text


def _cell(value: Any) -> str:
    """Table cell text, with pipes and newlines escaped"""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: List[str], rows: List[List[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines


def _summary_table(summary: Dict[str, Dict[str, Any]], fmt: str) -> List[str]:
    header = [
        "Row",
        "Model ID",
        "Sampling",
        "Cases",
        "Trials",
        "Payor accuracy",
        "CPT P",
        "CPT R",
        "ICD P",
        "ICD R",
        "Failure rate",
        "Truncation rate",
        "Latency p50 (ms)",
        "Latency p95 (ms)",
        "Mean cost per case",
        "Total cost",
    ]
    rows = []
    for row, entry in summary.items():
        stats = entry.get(fmt)
        if stats is None:
            continue
        rows.append(
            [
                row,
                entry["model_id"],
                fmt_sampling(entry["sampling"], entry["sampling_note"]),
                stats["n_cases"],
                stats["n_trials"],
                fmt_ratio(stats["payor_accuracy"]),
                fmt_ratio(stats["cpt_precision"]),
                fmt_ratio(stats["cpt_recall"]),
                fmt_ratio(stats["icd_precision"]),
                fmt_ratio(stats["icd_recall"]),
                fmt_ratio(stats["failure_rate"]),
                fmt_ratio(stats["truncation_rate"]),
                fmt_ms(stats["latency_p50_ms"]),
                fmt_ms(stats["latency_p95_ms"]),
                fmt_usd(stats["mean_cost_per_case_usd"]),
                fmt_usd(stats["total_cost_usd"]),
            ]
        )
    if not rows:
        return [f"No {fmt.upper()} trials."]
    return _table(header, rows)


def _trial_table(scored_trials: List[Dict[str, Any]]) -> List[str]:
    header = [
        "Row",
        "Case",
        "Trial",
        "Status",
        "Payor ok",
        "CPT P",
        "CPT R",
        "ICD P",
        "ICD R",
        "Stop reasons",
        "Input tokens",
        "Output tokens",
        "Latency (ms)",
        "Cost",
        "Error",
    ]
    rows = []
    for t in scored_trials:
        status = t["status"]
        if t["truncated"]:
            status += " (truncated)"
        rows.append(
            [
                t["row"],
                t["case_id"],
                t["trial"],
                status,
                "yes" if t["payor_correct"] else "no",
                fmt_ratio(t["cpt_precision"]),
                fmt_ratio(t["cpt_recall"]),
                fmt_ratio(t["icd_precision"]),
                fmt_ratio(t["icd_recall"]),
                ", ".join(str(r) for r in t["stop_reasons"]) or "none",
                fmt_int(t["input_tokens"]),
                fmt_int(t["output_tokens"]),
                fmt_ms(t["latency_ms"]),
                fmt_usd(t["cost_usd"]),
                t["error"] or "",
            ]
        )
    return _table(header, rows)


def _limits(
    scored_trials: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    cases: List[Dict[str, Any]],
) -> List[str]:
    pdf_cases = sorted({t["case_id"] for t in scored_trials if t["format"] == "pdf"})
    fhir_cases = sorted({t["case_id"] for t in scored_trials if t["format"] == "fhir"})
    pdf_trials = manifest.get("trials")
    if pdf_trials is None:
        pdf_trials = max(
            (t["trial"] for t in scored_trials if t["format"] == "pdf"), default=0
        )
    unreviewed = sorted(
        case["case_id"]
        for case in cases
        if case["case_id"] in pdf_cases
        and not (case.get("review") or {}).get("payor", "").startswith("confirmed")
    )
    lines = [
        "## Limits",
        "",
        f"- {len(pdf_cases)} PDF cases exercise a model; each runs {pdf_trials} "
        "trials per row.",
        f"- {len(fhir_cases)} FHIR cases are model-independent controls (codes "
        "are parsed from the bundle with no model call); each runs once.",
        "- Only the `medical_extraction` step runs; no other workflow step, "
        "Knowledge Base or deployed stack is used.",
        "- The cases are synthetic sample documents from `samples/`. Expected "
        "values are what those documents state, not clinical ground truth.",
    ]
    if unreviewed:
        lines.append(
            "- PDF expected payor values were not separately reviewed (per the "
            "case files' `review` notes): " + ", ".join(unreviewed) + "."
        )
    lines += [
        "- Failed or truncated trials score zero on every field and count "
        "against their row.",
        "- With this few cases and trials, the results describe these documents "
        "only. This report does not name a winner or rank models beyond the "
        "numbers shown.",
    ]
    return lines


def _pricing(pricing: Optional[Dict[str, Any]]) -> List[str]:
    lines = ["## Price table", ""]
    if not pricing:
        return lines + ["No price table was provided; costs are n/a."]
    lines += [
        f"As of {pricing.get('as_of', 'n/a')}, source: {pricing.get('source', 'n/a')}"
        f" ({pricing.get('currency', 'USD')}, {pricing.get('unit', 'per 1K tokens')}"
        "). Models missing from the table show cost n/a.",
        "",
    ]
    rows = [
        [model_id, price.get("input"), price.get("output")]
        for model_id, price in sorted(pricing.get("models", {}).items())
    ]
    return lines + _table(["Model ID", "Input", "Output"], rows)


def render_markdown(
    scored_trials: List[Dict[str, Any]],
    summary: Dict[str, Dict[str, Any]],
    manifest: Dict[str, Any],
    pricing: Optional[Dict[str, Any]] = None,
    cases: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render the comparison report.

    pricing defaults to manifest["pricing"]; cases default to the case files.
    """
    if pricing is None:
        pricing = manifest.get("pricing")
    if cases is None:
        cases = load_cases()

    lines = ["# Extraction model comparison", ""]
    for key, label in (
        ("git_commit", "Git commit"),
        ("started_at", "Started"),
        ("finished_at", "Finished"),
    ):
        if manifest.get(key):
            lines.append(f"- {label}: {manifest[key]}")
    lines.append("")
    lines += _limits(scored_trials, manifest, cases) + [""]

    lines += ["## PDF results (model-dependent)", ""]
    lines += _summary_table(summary, "pdf") + [""]
    lines += ["## FHIR controls (model-independent)", ""]
    lines += _summary_table(summary, "fhir") + [""]

    lines += ["## Unavailable models", ""]
    unavailable = manifest.get("unavailable_models") or []
    if unavailable:
        rows = [[m["model_id"], m["reason"]] for m in unavailable]
        lines += _table(["Model ID", "Reason"], rows)
    else:
        lines.append("None.")
    lines.append("")

    lines += ["## Lifecycle checks", ""]
    checks = manifest.get("lifecycle_checks") or []
    if checks:
        rows = [
            [c["model_id"], c["status"], c["source"], c["checked_at"]] for c in checks
        ]
        lines += _table(["Model ID", "Status", "Source", "Checked at"], rows)
    else:
        lines.append("None recorded.")
    lines.append("")

    lines += _pricing(pricing) + [""]

    notes = manifest.get("notes") or []
    if notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in notes] + [""]

    lines += [f"## Every trial ({len(scored_trials)})", ""]
    lines += _trial_table(scored_trials) + [""]
    return "\n".join(lines)


def write_csv(scored_trials: List[Dict[str, Any]], path) -> None:
    """Write one row per scored trial"""
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for t in scored_trials:
            sampling = t.get("sampling") or {}
            row = {field: t.get(field) for field in CSV_FIELDS}
            row["temperature"] = sampling.get("temperature")
            row["top_p"] = sampling.get("top_p")
            row["stop_reasons"] = ";".join(str(r) for r in t["stop_reasons"])
            writer.writerow(row)
