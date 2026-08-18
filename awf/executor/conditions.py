"""Restricted condition-expression evaluation for workflow branches."""

from __future__ import annotations

import ast
import operator
from typing import Any

from awf.executor.context import ExecutionContext


class ConditionExpressionError(ValueError):
    """Raised when a condition uses unsupported or unsafe syntax."""


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_SAFE_CONTAINER_TYPES = (dict, list, tuple, set, str, bytes)
_MAX_EXPRESSION_LENGTH = 2_000
_MAX_SEQUENCE_RESULT = 100_000
_MAX_COLLECTION_ITEMS = 1_000


def evaluate_condition_expression(
    expression: str,
    context: ExecutionContext,
) -> bool:
    """Evaluate a small expression language without Python ``eval``.

    Conditions can read ``outputs``, ``vars`` and a sanitized ``context``
    mapping. They support literals, indexing, mapping ``.get()``, boolean
    operations, comparisons and bounded basic arithmetic. Function calls,
    comprehensions and access to Python object internals are rejected.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ConditionExpressionError("condition expression must be non-empty")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ConditionExpressionError("condition expression is too long")

    safe_context = {
        "query": context.query,
        "current_node_id": context.current_node_id,
        "previous_node_id": context.previous_node_id,
        "step_count": context.step_count,
        "history": tuple(context.history),
        "outputs": context.outputs,
        "variables": context.variables,
        "cost_summary": context.cost_summary,
        "finished": context.finished,
        "success": context.success,
    }
    evaluator = _ConditionEvaluator(
        {
            "outputs": context.outputs,
            "vars": context.variables,
            "context": safe_context,
        }
    )
    result = evaluator.evaluate(expression)
    if not isinstance(
        result,
        (bool, int, float, str, bytes, list, tuple, set, dict, type(None)),
    ):
        raise ConditionExpressionError(
            f"condition produced unsupported value type: {type(result).__name__}"
        )
    return bool(result)


class _ConditionEvaluator:
    """Recursive evaluator for the explicitly allowed AST subset."""

    def __init__(self, environment: dict[str, Any]):
        self.environment = environment

    def evaluate(self, expression: str) -> Any:
        try:
            parsed = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ConditionExpressionError(
                f"invalid condition syntax: {exc}"
            ) from exc
        return self._evaluate(parsed.body)

    def _evaluate(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return self._constant(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.environment:
                raise ConditionExpressionError(f"unknown name: {node.id}")
            return self.environment[node.id]
        if isinstance(node, ast.List):
            return self._collection(node.elts, list)
        if isinstance(node, ast.Tuple):
            return self._collection(node.elts, tuple)
        if isinstance(node, ast.Set):
            return self._collection(node.elts, set)
        if isinstance(node, ast.Dict):
            if len(node.keys) > _MAX_COLLECTION_ITEMS:
                raise ConditionExpressionError("mapping literal is too large")
            return {
                self._evaluate(key): self._evaluate(value)
                for key, value in zip(node.keys, node.values)
                if key is not None
            }
        if isinstance(node, ast.Subscript):
            owner = self._evaluate(node.value)
            key = self._evaluate(node.slice)
            if type(owner) not in _SAFE_CONTAINER_TYPES:
                raise ConditionExpressionError(
                    "indexing is limited to built-in containers"
                )
            try:
                return owner[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ConditionExpressionError(
                    f"condition lookup failed: {exc}"
                ) from exc
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ConditionExpressionError("private attributes are forbidden")
            owner = self._evaluate(node.value)
            if type(owner) is not dict or node.attr not in owner:
                raise ConditionExpressionError(
                    "attribute access is limited to context mapping fields"
                )
            return owner[node.attr]
        if isinstance(node, ast.Call):
            return self._mapping_get(node)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(bool(self._evaluate(value)) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(bool(self._evaluate(value)) for value in node.values)
            raise ConditionExpressionError("unsupported boolean operator")
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand)
            if isinstance(node.op, ast.Not):
                return not bool(value)
            if (
                isinstance(node.op, (ast.UAdd, ast.USub))
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return value if isinstance(node.op, ast.UAdd) else -value
            raise ConditionExpressionError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            return self._binary_operation(node)
        if isinstance(node, ast.Compare):
            return self._comparison(node)
        if isinstance(node, ast.IfExp):
            branch = node.body if bool(self._evaluate(node.test)) else node.orelse
            return self._evaluate(branch)
        raise ConditionExpressionError(
            f"unsupported condition syntax: {type(node).__name__}"
        )

    def _constant(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > _MAX_SEQUENCE_RESULT:
            raise ConditionExpressionError("string literal is too large")
        if isinstance(value, int) and value.bit_length() > 256:
            raise ConditionExpressionError("integer literal is too large")
        if isinstance(value, (str, bytes, bool, int, float, type(None))):
            return value
        raise ConditionExpressionError(
            f"unsupported literal type: {type(value).__name__}"
        )

    def _collection(self, nodes: list[ast.AST], factory: type) -> Any:
        if len(nodes) > _MAX_COLLECTION_ITEMS:
            raise ConditionExpressionError("collection literal is too large")
        return factory(self._evaluate(item) for item in nodes)

    def _mapping_get(self, node: ast.Call) -> Any:
        if (
            not isinstance(node.func, ast.Attribute)
            or node.func.attr != "get"
            or node.keywords
            or not 1 <= len(node.args) <= 2
        ):
            raise ConditionExpressionError(
                "only mapping.get(key[, default]) calls are allowed"
            )
        owner = self._evaluate(node.func.value)
        if type(owner) is not dict:
            raise ConditionExpressionError(".get() requires a built-in mapping")
        key = self._evaluate(node.args[0])
        default = self._evaluate(node.args[1]) if len(node.args) == 2 else None
        return owner.get(key, default)

    def _binary_operation(self, node: ast.BinOp) -> Any:
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ConditionExpressionError("unsupported arithmetic operator")
        left = self._evaluate(node.left)
        right = self._evaluate(node.right)
        if isinstance(node.op, ast.Mult):
            sequence, count = (
                (left, right)
                if isinstance(left, (str, bytes, list, tuple))
                else (right, left)
            )
            if isinstance(sequence, (str, bytes, list, tuple)) and isinstance(
                count, int
            ):
                if count < 0 or len(sequence) * count > _MAX_SEQUENCE_RESULT:
                    raise ConditionExpressionError(
                        "condition sequence result is too large"
                    )
        try:
            result = operation(left, right)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ConditionExpressionError(
                f"condition arithmetic failed: {exc}"
            ) from exc
        if isinstance(result, int) and result.bit_length() > 512:
            raise ConditionExpressionError("condition integer result is too large")
        return result

    def _comparison(self, node: ast.Compare) -> bool:
        left = self._evaluate(node.left)
        for raw_operator, raw_right in zip(node.ops, node.comparators):
            operation = _COMPARISON_OPERATORS.get(type(raw_operator))
            if operation is None:
                raise ConditionExpressionError(
                    "unsupported comparison operator"
                )
            right = self._evaluate(raw_right)
            try:
                matched = operation(left, right)
            except (TypeError, ValueError) as exc:
                raise ConditionExpressionError(
                    f"condition comparison failed: {exc}"
                ) from exc
            if not matched:
                return False
            left = right
        return True
