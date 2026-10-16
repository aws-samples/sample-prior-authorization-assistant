"""Replay recorded Textract calls for the sample PDFs.

Fixtures in evaluation/fixtures/textract/ hold every Textract call the PDF
extraction path made for each sample (recorded 2026-10-16 against a
temporary bucket), including errors. Replaying them keeps the production
parsing code in the loop without S3 or Textract, and gives every model the
same text input.

The replay is strict: each call must match the next recorded call's
operation and parameters (with the recording's bucket placeholder), and
callers should check that every recorded call was used. Because the
production code catches Textract errors and falls back to other calls, the
first mismatch is also kept on the client (``mismatch``) so it is reported
under its own message.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "textract"
BUCKET_PLACEHOLDER = "<bucket>"

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

    def __init__(self, document: str, bucket: str):
        fixture = json.loads((FIXTURES / f"{Path(document).stem}.json").read_text())
        self._calls: List[Dict[str, Any]] = list(fixture["calls"])
        self._bucket = bucket
        self.calls_made = 0
        self.mismatch: Optional[str] = None

    def __getattr__(self, operation: str):
        if operation not in _OPERATION_NAMES:
            raise AttributeError(operation)

        def call(**params):
            recorded = self._calls[0] if self._calls else None
            problem = self._check(operation, params, recorded)
            if problem:
                self.mismatch = self.mismatch or problem
                raise FixtureMismatch(problem)
            self._calls.pop(0)
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

    def _check(self, operation, params, recorded) -> Optional[str]:
        if recorded is None:
            return f"unexpected extra Textract call: {operation}"
        if recorded["operation"] != operation:
            return f"expected {recorded['operation']}, got {operation}"
        sent = json.loads(json.dumps(params).replace(self._bucket, BUCKET_PLACEHOLDER))
        if sent != recorded["params"]:
            return f"{operation} parameters differ from the recording: {sent}"
        return None
