"""Importing the metadata extractor must not reconfigure logging.

MetadataExtractor, MetadataProcessor, S3Uploader and KBSyncTrigger import the
metadata_extractor package. Configuring structlog or the root logger at import
time changes their log output in Lambda. Each check runs in a fresh
interpreter so earlier tests cannot mask a change.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PROBE = """
import json, logging, structlog

def snapshot():
    root = logging.getLogger()
    return {
        "structlog_configured": structlog.is_configured(),
        "root_level": root.level,
        "root_handlers": [type(h).__name__ for h in root.handlers],
    }

before = snapshot()
import {module}
after = snapshot()
print(json.dumps({{"before": before, "after": after}}))
"""


@pytest.mark.parametrize(
    "module",
    [
        "metadata_extractor.metadata_extractor",
        "metadata_extractor",
        "common.bedrock_converse",
    ],
)
def test_import_leaves_logging_configuration_unchanged(module):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(SRC),
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    }
    # Windows needs SYSTEMROOT to start a Python subprocess
    if "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE.replace("{module}", module).replace("{{", "{").replace("}}", "}"),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    snapshots = json.loads(result.stdout.strip().splitlines()[-1])

    assert snapshots["after"] == snapshots["before"]
