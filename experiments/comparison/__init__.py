"""Dataset access utilities for the official test pipeline.

The frozen method-comparison protocol (adapters, runner, comparison CLI) was
removed in the public release; only the manifest-bound dataset loader used by
``awf-test`` is retained.
"""

from experiments.comparison.dataset_access import (
    bound_selected_jsonl_file,
    selected_jsonl_file,
    validate_bound_dataset_source,
)

__all__ = [
    "bound_selected_jsonl_file",
    "selected_jsonl_file",
    "validate_bound_dataset_source",
]
