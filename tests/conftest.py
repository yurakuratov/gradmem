import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_csv_option(raw_value: str):
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value == "":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="",
        help="Run tests only for selected memory backend(s), comma-separated. "
             "Example: --backend prefix,lora",
    )
    parser.addoption(
        "--model-family",
        action="store",
        default="",
        help="Run tests only for selected model family/families, comma-separated. "
             "Example: --model-family llama,gpt2",
    )


def pytest_collection_modifyitems(config, items):
    backend_filter = _parse_csv_option(config.getoption("backend"))
    model_filter = _parse_csv_option(config.getoption("model_family"))

    if backend_filter is None and model_filter is None:
        return

    selected = []
    deselected = []

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            selected.append(item)
            continue

        params = callspec.params
        backend_ok = True
        model_ok = True

        if backend_filter is not None and "memory_backend" in params:
            backend_ok = str(params["memory_backend"]) in backend_filter
        if model_filter is not None and "model_family" in params:
            model_ok = str(params["model_family"]) in model_filter

        if backend_ok and model_ok:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
