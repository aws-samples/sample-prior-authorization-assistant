"""Ground-truth cases for the extraction model comparison."""

import copy
import json

import pytest

from evaluation import ground_truth as gt

CASES = gt.load_cases()
BY_ID = {case["case_id"]: case for case in CASES}


def test_every_sample_has_exactly_one_case():
    expected = {gt.case_path(doc).name for doc in gt.sample_documents()}
    actual = {path.name for path in gt.CASES.glob("*.json")}

    assert actual == expected


@pytest.mark.parametrize("case_id", sorted(BY_ID))
def test_case_is_valid(case_id):
    assert gt.validate_case(BY_ID[case_id]) == []


@pytest.mark.parametrize(
    "case_id", sorted(cid for cid, case in BY_ID.items() if case["format"] == "fhir")
)
def test_fhir_case_is_reproducible_from_the_bundle(case_id):
    case = BY_ID[case_id]

    assert gt.build_fhir_case(gt.ROOT / case["source_file"]) == case


@pytest.mark.parametrize(
    "case_id", sorted(cid for cid, case in BY_ID.items() if case["format"] == "pdf")
)
def test_pdf_case_values_appear_on_their_recorded_page(case_id):
    pypdf = pytest.importorskip("pypdf")
    case = BY_ID[case_id]
    pages = pypdf.PdfReader(str(gt.ROOT / case["source_file"])).pages

    def text_on(location):
        number = int(location.removeprefix("page "))
        return pages[number - 1].extract_text() or ""

    for field in ("cpt_codes", "icd_codes"):
        for code in case["expected"][field]:
            assert code in text_on(case["provenance"][field][code])
    payor = case["provenance"]["payor"]
    assert payor["text"] in text_on(payor["location"])


def test_paired_pdf_and_fhir_cases_agree():
    # Recorded 2026-10-16: each PDF states the same payor, CPT and ICD codes as
    # its FHIR bundle. A difference here means a sample or the builder changed.
    assert gt.cross_check(CASES) == []


def test_cross_check_reports_differences():
    cases = copy.deepcopy(CASES)
    pdf = next(
        c for c in cases if c["case_id"] == "anthem-knee-surgery-pa-approved.pdf"
    )
    pdf["expected"]["icd_codes"] = ["M23.202"]

    assert gt.cross_check(cases) == [
        "anthem-knee-surgery-pa-approved icd_codes: pdf=['M23.202'] fhir=['M23.201']"
    ]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["expected"].update(payor="Cigna"), "expected.payor"),
        (lambda c: c["expected"].update(cpt_codes=["2988"]), "invalid code"),
        (lambda c: c["expected"].update(icd_codes=[]), "non-empty list"),
        (lambda c: c["provenance"]["cpt_codes"].clear(), "missing location"),
        (lambda c: c.update(format="xml"), "format"),
        (lambda c: c.update(source_file="samples/missing.pdf"), "does not exist"),
    ],
)
def test_validator_rejects_bad_cases(mutate, message):
    case = copy.deepcopy(BY_ID["anthem-knee-surgery-pa-approved.fhir"])
    mutate(case)

    assert any(message in error for error in gt.validate_case(case))


def test_case_files_are_formatted_as_written():
    for path in gt.CASES.glob("*.json"):
        assert (
            path.read_text()
            == json.dumps(json.loads(path.read_text()), indent=2) + "\n"
        )


@pytest.mark.parametrize(
    "case_id", sorted(cid for cid, case in BY_ID.items() if case["format"] == "pdf")
)
def test_pdf_builder_reproduces_confirmed_values(case_id):
    pytest.importorskip("pypdf")
    case = BY_ID[case_id]

    rebuilt = gt.build_pdf_case(gt.ROOT / case["source_file"])

    assert rebuilt["expected"] == case["expected"]
    assert rebuilt["provenance"] == case["provenance"]
