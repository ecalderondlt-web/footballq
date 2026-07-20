"""Reproducibility, identity, and provenance helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SplitManifest",
    "build_run_manifest",
    "ensure_unique_sample_ids",
    "load_split_manifest",
    "make_sample_id",
    "sample_ids_from_components",
    "validate_run_manifest",
    "write_run_manifest",
]

_EXPORT_MODULES = {
    "SplitManifest": "footballq.repro.splits",
    "build_run_manifest": "footballq.repro.manifest",
    "ensure_unique_sample_ids": "footballq.repro.identity",
    "load_split_manifest": "footballq.repro.splits",
    "make_sample_id": "footballq.repro.identity",
    "sample_ids_from_components": "footballq.repro.identity",
    "validate_run_manifest": "footballq.repro.manifest",
    "write_run_manifest": "footballq.repro.manifest",
}


def __getattr__(name: str) -> Any:
    """Load public helpers only when callers request them."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
