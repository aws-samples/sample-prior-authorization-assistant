"""Ground truth for the extraction model comparison.

Builds one case per document in samples/: the payor, CPT codes and ICD-10
codes the document states. FHIR cases are read from FHIR resources; PDF cases
are read from the PDF text and must be confirmed by a person before scoring.

Usage:
    python -m evaluation.ground_truth build   # write evaluation/cases/*.json
    python -m evaluation.ground_truth check   # validate cases, cross-check pairs
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CASES = Path(__file__).resolve().parent / "cases"

# Payor families the pipeline supports (SUPPORTED_PAYORS in the CDK stack)
PAYOR_FAMILIES = ("Aetna", "Anthem", "Humana")

CPT_SYSTEM = "http://www.ama-assn.org/go/cpt"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"

ICD10_PATTERN = re.compile(r"\b[A-TV-Z]\d{2}\.\d{1,4}\b")
ICD10_CODE = re.compile(r"[A-TV-Z]\d{2}(?:\.\d{1,4})?")
CPT_CODE = re.compile(r"\d{5}")
CPT_INLINE_PATTERN = re.compile(r"\bCPT\s+(\d{5})\b")
CPT_ROW_PATTERN = re.compile(r"^(\d{5})\b")
SERVICE_TABLE_HEADER = re.compile(r"^Code\s+Service\s+Description")
SERVICE_TABLE_END = re.compile(r"^(Total|Subtotal|DIAGNOSIS|Diagnosis|CLINICAL)\b")

REQUIRED_FIELDS = {
    "case_id": str,
    "source_file": str,
    "format": str,
    "expected": dict,
    "provenance": dict,
    "confirmed": bool,
}


def payor_family(name: str) -> Optional[str]:
    """Map a payor organization name to its supported payor family"""
    for family in PAYOR_FAMILIES:
        if family.lower() in name.lower():
            return family
    return None


def build_fhir_case(path: Path) -> Dict[str, Any]:
    """Read payor, CPT and ICD-10 codes from a FHIR bundle's resources"""
    bundle = json.loads(path.read_text())
    resources = [entry["resource"] for entry in bundle.get("entry", [])]
    by_ref = {f"{r['resourceType']}/{r.get('id')}": r for r in resources}

    cpt, icd = {}, {}
    payor_name, payor_ref = None, None
    for resource in resources:
        kind = resource["resourceType"]
        for coding in _codings(resource):
            if coding.get("system") == CPT_SYSTEM:
                cpt.setdefault(coding["code"], kind)
            elif coding.get("system") == ICD10_SYSTEM:
                icd.setdefault(coding["code"], kind)
        if kind == "Coverage" and resource.get("payor"):
            payor_ref = resource["payor"][0].get("reference")
            payor_name = by_ref.get(payor_ref, {}).get("name")

    return _case(
        path,
        "fhir",
        payor_name=payor_name,
        payor_location=f"Coverage.payor -> {payor_ref}",
        cpt=cpt,
        icd=icd,
        confirmed=True,
    )


def build_pdf_case(path: Path) -> Dict[str, Any]:
    """Read payor, CPT and ICD-10 codes from a PDF's text, with page numbers"""
    import pypdf

    cpt, icd = {}, {}
    payor_name, payor_page = None, None
    for page_number, page in enumerate(pypdf.PdfReader(str(path)).pages, start=1):
        in_table = False
        for line in (page.extract_text() or "").splitlines():
            line = line.strip()
            if SERVICE_TABLE_HEADER.match(line):
                in_table = True
                continue
            if in_table and SERVICE_TABLE_END.match(line):
                in_table = False
            if in_table:
                match = CPT_ROW_PATTERN.match(line)
                if match:
                    cpt.setdefault(match.group(1), f"page {page_number}")
            for code in CPT_INLINE_PATTERN.findall(line):
                cpt.setdefault(code, f"page {page_number}")
            for code in ICD10_PATTERN.findall(line):
                icd.setdefault(code, f"page {page_number}")
            payor = re.match(r"(?:Payor|Payer)\s*:\s*(.+)", line)
            if payor and payor_name is None:
                payor_name, payor_page = payor.group(1).strip(), page_number

    return _case(
        path,
        "pdf",
        payor_name=payor_name,
        payor_location=f"page {payor_page}",
        cpt=cpt,
        icd=icd,
        confirmed=False,
    )


def validate_case(case: Dict[str, Any]) -> List[str]:
    """Return schema errors for a case (empty when valid)"""
    errors = []
    for field, kind in REQUIRED_FIELDS.items():
        if not isinstance(case.get(field), kind):
            errors.append(f"{field}: expected {kind.__name__}")
    if errors:
        return errors
    if case["format"] not in ("pdf", "fhir"):
        errors.append("format: must be 'pdf' or 'fhir'")
    if not (ROOT / case["source_file"]).exists():
        errors.append(f"source_file: {case['source_file']} does not exist")
    expected = case["expected"]
    if expected.get("payor") not in PAYOR_FAMILIES:
        errors.append(f"expected.payor: must be one of {PAYOR_FAMILIES}")
    for field, pattern in (("cpt_codes", CPT_CODE), ("icd_codes", ICD10_CODE)):
        codes = expected.get(field)
        if not isinstance(codes, list) or not codes:
            errors.append(f"expected.{field}: must be a non-empty list")
            continue
        if codes != sorted(set(codes)):
            errors.append(f"expected.{field}: must be sorted and unique")
        for code in codes:
            if not pattern.fullmatch(code):
                errors.append(f"expected.{field}: invalid code {code!r}")
            if code not in case["provenance"].get(field, {}):
                errors.append(f"provenance.{field}: missing location for {code}")
    return errors


def sample_documents() -> List[Path]:
    return sorted(p for p in SAMPLES.iterdir() if p.suffix in (".pdf", ".json"))


def case_path(document: Path) -> Path:
    fmt = "pdf" if document.suffix == ".pdf" else "fhir"
    return CASES / f"{document.stem}.{fmt}.json"


def load_cases() -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(CASES.glob("*.json"))]


def cross_check(cases: List[Dict[str, Any]]) -> List[str]:
    """Compare expected values of PDF and FHIR cases that share a case name"""
    by_name: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for case in cases:
        by_name.setdefault(case["case_id"].rsplit(".", 1)[0], {})[case["format"]] = case
    differences = []
    for name, pair in sorted(by_name.items()):
        if set(pair) != {"pdf", "fhir"}:
            continue
        for field in ("payor", "cpt_codes", "icd_codes"):
            pdf_value = pair["pdf"]["expected"][field]
            fhir_value = pair["fhir"]["expected"][field]
            if pdf_value != fhir_value:
                differences.append(f"{name} {field}: pdf={pdf_value} fhir={fhir_value}")
    return differences


def build_all() -> List[Path]:
    """Write a case per sample, keeping confirmed PDF cases unchanged"""
    CASES.mkdir(exist_ok=True)
    written = []
    for document in sample_documents():
        path = case_path(document)
        if path.exists() and document.suffix == ".pdf":
            existing = json.loads(path.read_text())
            if existing.get("confirmed"):
                continue
        builder = build_pdf_case if document.suffix == ".pdf" else build_fhir_case
        path.write_text(json.dumps(builder(document), indent=2) + "\n")
        written.append(path)
    return written


def _case(path, fmt, payor_name, payor_location, cpt, icd, confirmed):
    return {
        "case_id": f"{path.stem}.{fmt}",
        "source_file": str(path.relative_to(ROOT)),
        "format": fmt,
        "expected": {
            "payor": payor_family(payor_name or ""),
            "cpt_codes": sorted(cpt),
            "icd_codes": sorted(icd),
        },
        "provenance": {
            "payor": {"text": payor_name, "location": payor_location},
            "cpt_codes": dict(sorted(cpt.items())),
            "icd_codes": dict(sorted(icd.items())),
        },
        "confirmed": confirmed,
    }


def _codings(obj):
    if isinstance(obj, dict):
        if "system" in obj and "code" in obj:
            yield obj
        for value in obj.values():
            yield from _codings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _codings(value)


def main(argv: List[str]) -> int:
    command = argv[1] if len(argv) > 1 else "check"
    if command == "build":
        for path in build_all():
            print(f"wrote {path.relative_to(ROOT)}")
        return 0
    cases = load_cases()
    failed = False
    for case in cases:
        for error in validate_case(case):
            print(f"{case.get('case_id')}: {error}")
            failed = True
    for difference in cross_check(cases):
        print(f"cross-check: {difference}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
