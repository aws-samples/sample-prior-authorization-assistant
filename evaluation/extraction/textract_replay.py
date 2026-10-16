"""Replay recorded Textract calls for the sample PDFs.

Fixtures in evaluation/fixtures/textract/ hold every Textract call the PDF
extraction path made for each sample (recorded 2026-10-16 against a
temporary bucket), including errors. Replaying them keeps the production
parsing code in the loop without S3 or Textract, and gives every model the
same text input.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from botocore.exceptions import ClientError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "textract"

_OPERATION_NAMES = {
    "start_document_text_detection": "StartDocumentTextDetection",
    "get_document_text_detection": "GetDocumentTextDetection",
    "detect_document_text": "DetectDocumentText",
    "analyze_document": "AnalyzeDocument",
}


class FixtureMismatch(AssertionError):
    """The code under test made a Textract call the fixture does not contain"""


class ReplayTextractClient:
    """Stands in for a boto3 Textract client, returning recorded responses"""

    def __init__(self, document: str):
        fixture = json.loads((FIXTURES / f"{Path(document).stem}.json").read_text())
        self._calls: List[Dict[str, Any]] = list(fixture["calls"])
        self.calls_made = 0

    def __getattr__(self, operation: str):
        if operation not in _OPERATION_NAMES:
            raise AttributeError(operation)

        def call(**params):
            if not self._calls:
                raise FixtureMismatch(f"unexpected extra Textract call: {operation}")
            recorded = self._calls.pop(0)
            if recorded["operation"] != operation:
                raise FixtureMismatch(
                    f"expected {recorded['operation']}, got {operation}"
                )
            self.calls_made += 1
            if "error" in recorded:
                error = recorded["error"]
                raise ClientError(
                    {"Error": {"Code": error["code"], "Message": error["message"]}},
                    _OPERATION_NAMES[operation],
                )
            return recorded["response"]

        return call

    @property
    def remaining(self) -> int:
        return len(self._calls)
