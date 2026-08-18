"""Safety declarations for counterfactual workflow execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


_Operator = TypeVar("_Operator", bound=Callable[..., Any])


def counterfactual_safe(operator: _Operator) -> _Operator:
    """Mark a side-effect-free operator as safe for candidate reruns.

    The optimizer executes unaccepted candidates during counterfactual
    evaluation. Operators that write external state must not use this marker;
    they require a caller-provided sandbox or transaction adapter instead.
    """
    setattr(operator, "__awf_counterfactual_safe__", True)
    return operator


def is_counterfactual_safe(operator: Callable[..., Any]) -> bool:
    """Return whether an operator carries the explicit safety marker."""
    return getattr(operator, "__awf_counterfactual_safe__", False) is True
