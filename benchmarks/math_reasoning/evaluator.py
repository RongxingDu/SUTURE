"""Answer extraction and comparison for math-reasoning tasks."""

from __future__ import annotations

import ast
import multiprocessing
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isqrt
from typing import Optional


_MAX_SYMBOLIC_INPUT = 512
_MAX_AST_NODES = 128
_MAX_AST_DEPTH = 24
_MAX_POLYNOMIAL_TERMS = 256
_MAX_COLLECTION_ITEMS = 32
_MAX_INTEGER_DIGITS = 40
_MAX_POWER = 12
_SYMPY_TIMEOUT_SECONDS = 0.75
try:  # Optional: the exact polynomial/set fallback remains dependency-free.
    import sympy as _sympy_module
except Exception:  # pragma: no cover - depends on the installation profile
    _sympy_module = None

_SYMPY_AVAILABLE = _sympy_module is not None


class MathEvaluator:
    """Extract and compare common GSM8K/MATH answer formats."""

    @staticmethod
    def extract_answer(response: str) -> str:
        """Extract the semantically final answer from a model response.

        Explicit verifier/final-answer markers and balanced ``\\boxed{...}``
        expressions are preferred.  An unmarked number buried in explanatory
        prose is deliberately *not* treated as the answer.
        """
        if not response:
            return ""

        semantic_response = _normalize_semantic_markdown(response)
        cleaned_response = _clean_response_candidate(semantic_response)
        if _numeric_value(cleaned_response) is not None:
            return cleaned_response

        candidates: list[tuple[int, str]] = []

        if "####" in semantic_response:
            marker_position = semantic_response.rfind("####")
            marker_answer = semantic_response[
                marker_position + len("####"):
            ].strip()
            candidate = _semantic_candidate(
                marker_answer.splitlines()[0] if marker_answer else "",
            )
            if candidate is not None:
                candidates.append((marker_position, candidate))

        answer_patterns = [
            # The workflow's verifier contract.
            r"^\s*(?:verified|passed?)\s*:\s*([^\n]+)",
            # A verifier may reject the draft and state a corrected result.
            r"(?:the\s+)?correct\s+(?:final\s+)?"
            r"(?:answer|result|solution)\s*(?:is|=|:|should\s+be)\s*([^\n]+)",
            r"(?:the\s+)?(?:actual|right)\s+(?:final\s+)?"
            r"(?:answer|result|solution)\s*(?:is|=|:|should\s+be)\s*([^\n]+)",
            # Verifiers frequently restate domain-specific scalar answers
            # below a Markdown ``Final Answer`` heading instead of calling
            # the value itself an "answer".  Treat these as explicit answer
            # markers as well; the captured payload still passes through the
            # conservative semantic-candidate parser.
            r"(?:the\s+)?(?:correct|actual|right|final)\s+"
            r"(?:probability|value|quantity)\s*"
            r"(?:is|=|:|should\s+be)\s*([^\n]+)",
            r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*([^\n]+)",
            r"(?:final\s+)?result\s*(?:is|=|:)\s*([^\n]+)",
            r"(?:therefore|hence|thus)\s*,?\s*([^\n]+)",
        ]
        for pattern in answer_patterns:
            for match in re.finditer(
                pattern,
                semantic_response,
                re.IGNORECASE | re.MULTILINE,
            ):
                candidate = _semantic_candidate(match.group(1))
                if candidate is not None:
                    candidates.append((match.start(), candidate))

        candidates.extend(
            (position, _clean_response_candidate(value))
            for position, value in _extract_boxed_candidates(semantic_response)
        )
        if candidates:
            # Position, rather than regex order, matters when an ERROR response
            # quotes the rejected answer before supplying the corrected one.
            return max(candidates, key=lambda item: item[0])[1]

        # A bare answer, or a final line consisting only of an answer, is still
        # useful.  Do not fall back to the last numeric token of a paragraph:
        # that silently rewards arbitrary intermediate calculations.
        lines = [
            line.strip()
            for line in semantic_response.splitlines()
            if line.strip()
        ]
        for value in (
            [semantic_response] if len(lines) <= 1 else [lines[-1]]
        ):
            candidate = _semantic_candidate(value)
            if candidate is not None:
                return candidate
        return ""

    @classmethod
    def normalize_ground_truth(cls, ground_truth: object) -> str:
        """Reduce a dataset solution/rationale to its canonical final answer."""
        if ground_truth is None:
            return ""
        if isinstance(ground_truth, dict):
            if "answer" not in ground_truth:
                return ""
            return cls.normalize_ground_truth(ground_truth["answer"])
        text = str(ground_truth).strip()
        if not text:
            return ""

        cleaned_text = _clean_answer(text)
        if _numeric_value(cleaned_text) is not None:
            return cleaned_text

        boxed = _extract_last_boxed(text)
        if boxed is not None:
            return _clean_answer(boxed)

        if "####" in text:
            candidate = text.rsplit("####", 1)[1].strip()
            if candidate:
                return _clean_answer(candidate.splitlines()[0])

        # Short, single-line MATH answers such as ``2x+2`` or
        # ``\{1,2\}`` are already answers. Running the generic response
        # extractor on them would incorrectly return only their last number.
        if (
            len(text) <= _MAX_SYMBOLIC_INPUT
            and "\n" not in text
            and not re.search(
                r"\b(?:answer|therefore|hence|thus|result)\b",
                text,
                re.IGNORECASE,
            )
        ):
            return cleaned_text

        extracted = cls.extract_answer(text)
        return extracted or _clean_answer(text)

    @classmethod
    def compare_answers(cls, predicted: str, ground_truth: object) -> bool:
        """Compare numeric answers with tolerance, otherwise normalized text."""
        pred = _clean_response_candidate(predicted)
        gt = cls.normalize_ground_truth(ground_truth)
        if not pred or not gt:
            return False

        pred_num = _numeric_value(pred)
        gt_num = _numeric_value(gt)
        if pred_num is None:
            pred_num = _numeric_value_with_optional_unit(pred)
        if gt_num is None:
            gt_num = _numeric_value_with_optional_unit(gt)
        if pred_num is not None and gt_num is not None:
            difference = abs(pred_num - gt_num)
            scale = max(Decimal(1), abs(pred_num), abs(gt_num))
            return difference <= Decimal("1e-9") * scale

        if _normalize_text(pred) == _normalize_text(gt):
            return True
        if _structured_equal(pred, gt):
            return True
        return _safe_symbolic_equal(pred, gt)

    def evaluate(self, output: str, ground_truth: object) -> bool:
        """Evaluate a model response against a short or rationale-style GT."""
        extracted = self.extract_answer(output)
        return self.compare_answers(extracted, ground_truth)


def _extract_last_boxed(text: str) -> Optional[str]:
    candidates = _extract_boxed_candidates(text)
    return candidates[-1][1] if candidates else None


def _extract_boxed_candidates(text: str) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    for command in (r"\boxed{", r"\fbox{"):
        search_from = 0
        while True:
            start = text.find(command, search_from)
            if start < 0:
                break
            content_start = start + len(command)
            depth = 1
            index = content_start
            while index < len(text) and depth:
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                index += 1
            if depth == 0:
                candidates.append((start, text[content_start:index - 1]))
                search_from = index
            else:
                search_from = content_start
    return sorted(candidates, key=lambda item: item[0])


_SEMANTIC_MARKER = (
    r"(?:verified|passed?|error"
    r"|(?:the\s+)?(?:correct|actual|right)\s+(?:final\s+)?"
    r"(?:answer|result|solution)"
    r"|(?:the\s+)?(?:final\s+)?answer"
    r"|(?:final\s+)?result)"
)
_MARKDOWN_MARKER_PREFIX = re.compile(
    r"^(?P<indent>\s*)"
    r"(?:\#{1,6}\s+|(?:[-+*]|\d+[\.\)])\s+)"
    r"(?=(?:\*{1,3}|_{1,3}|`{1,3})?"
    + _SEMANTIC_MARKER
    + r"\b)",
    re.IGNORECASE,
)
_MARKDOWN_MARKER_EMPHASIS = re.compile(
    r"(?P<wrapper>\*{1,3}|_{1,3}|`{1,3})"
    r"(?P<label>"
    + _SEMANTIC_MARKER
    + r"\s*:?)"
    r"(?P=wrapper)",
    re.IGNORECASE,
)
_MARKDOWN_OUTER_WRAPPER = re.compile(
    r"^(?P<wrapper>\*{1,3}|_{1,3}|`{1,3})"
    r"(?P<body>.+)"
    r"(?P=wrapper)$",
)


def _normalize_semantic_markdown(value: str) -> str:
    """Remove Markdown wrappers only where they surround answer markers."""
    normalized: list[str] = []
    for line in value.splitlines():
        line = _MARKDOWN_MARKER_PREFIX.sub(
            lambda match: match.group("indent"),
            line,
        )
        line = _strip_outer_markdown_wrapper(line)
        line = _MARKDOWN_MARKER_EMPHASIS.sub(
            lambda match: match.group("label"),
            line,
        )
        normalized.append(line)
    return "\n".join(normalized)


def _strip_outer_markdown_wrapper(value: str) -> str:
    result = value.strip()
    while True:
        match = _MARKDOWN_OUTER_WRAPPER.fullmatch(result)
        if match is None:
            return result
        result = match.group("body").strip()


def _clean_response_candidate(value: str) -> str:
    """Clean response-only wrappers without changing dataset GT canonical form."""
    result = _clean_answer(value)
    result = _strip_outer_markdown_wrapper(result)

    # Models commonly wrap verifier payloads in inline/display TeX delimiters.
    # Remove only a pair enclosing the *entire* candidate so interval
    # parentheses and mathematical grouping remain meaningful.
    wrappers = ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$"))
    changed = True
    while changed:
        changed = False
        for opening, closing in wrappers:
            if (
                result.startswith(opening)
                and result.endswith(closing)
                and len(result) >= len(opening) + len(closing)
            ):
                result = result[len(opening):-len(closing)].strip()
                result = result.rstrip(".").strip()
                changed = True
                break
    tex_choice = re.fullmatch(
        r"\\(?:text|mathrm|mathbf)\{\s*([A-E])\s*\}",
        result,
        re.IGNORECASE,
    )
    if tex_choice:
        return tex_choice.group(1).upper()
    return result


def _semantic_candidate(value: str) -> Optional[str]:
    """Return a marker payload only when it plausibly is a complete answer."""
    candidate = _clean_response_candidate(value)
    if not candidate or len(candidate) > _MAX_SYMBOLIC_INPUT:
        return None

    boxed = _extract_last_boxed(candidate)
    if boxed is not None:
        return _clean_response_candidate(boxed)

    # Normalize explicit multiple-choice phrasing while leaving intervals such
    # as ``(A,B]`` untouched.
    choice = re.fullmatch(
        r"(?:option|choice)\s*[\(\[]?\s*([A-E])\s*[\)\]]?",
        candidate,
        re.IGNORECASE,
    )
    if choice:
        return choice.group(1).upper()
    parenthesized_choice = re.fullmatch(
        r"[\(\[]\s*([A-E])\s*[\)\]]",
        candidate,
        re.IGNORECASE,
    )
    if parenthesized_choice:
        return parenthesized_choice.group(1).upper()

    if _numeric_value(candidate) is not None:
        return candidate
    if re.fullmatch(r"[A-E]", candidate, re.IGNORECASE):
        return candidate.upper()

    # Structured TeX answers are valid mathematical payloads even when they
    # contain several alphabetic words such as ``begin``/``pmatrix`` or text
    # units.  The prose filter below must not reject these marker payloads.
    if _is_tex_expression_candidate(candidate):
        return candidate

    if re.search(
        r"(?:^|\s)(?:step|case|check|compute|intermediate|because|since)"
        r"\b",
        candidate,
        re.IGNORECASE,
    ):
        return None

    # Reject ordinary prose.  Mathematical answers may contain TeX commands,
    # digits/operators, collections, or a single symbolic identifier.
    alphabetic_words = re.findall(r"(?<!\\)\b[A-Za-z]{2,}\b", candidate)
    if len(alphabetic_words) > 1:
        return None
    if re.search(r"(?:\\[A-Za-z]+|[\d=+\-*/^_{}\[\](),])", candidate):
        return candidate
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", candidate):
        return candidate
    return None


def _is_tex_expression_candidate(value: str) -> bool:
    """Return whether ``value`` is a standalone structured TeX expression.

    This deliberately accepts only TeX syntax, commands, and their payloads;
    ordinary prose such as ``The answer is ...`` remains rejected.
    """
    environment = re.fullmatch(
        r"\\begin\{(?P<name>[A-Za-z*]+)\}.*"
        r"\\end\{(?P=name)\}",
        value,
        flags=re.DOTALL,
    )
    if environment is not None:
        return True

    # Remove command payloads whose alphabetic contents are intentionally
    # textual (for example ``\text{square units}``).  What remains must be
    # TeX commands, delimiters, operators, or scalar symbols rather than
    # natural-language prose.
    remainder = re.sub(
        r"\\(?:text|mathrm|mathbf|operatorname)\{[^{}]*\}",
        "",
        value,
    )
    remainder = re.sub(r"\\[A-Za-z]+", "", remainder)
    remainder = re.sub(r"[0-9A-Za-z\s{}\[\]()+\-*/=^_.,|&;:'<>%]", "", remainder)
    return bool(re.search(r"\\[A-Za-z]+", value)) and not remainder


def _clean_answer(value: str) -> str:
    result = value.strip()
    result = re.sub(r"^(?:is|=|:)\s*", "", result, flags=re.IGNORECASE)
    result = result.strip().strip("$")
    return result.rstrip().rstrip(".").strip()


def _numeric_value(value: str) -> Optional[Decimal]:
    normalized = value.strip().replace(",", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.strip("$").strip()

    percent = normalized.endswith("%") or normalized.endswith(r"\%")
    if percent:
        normalized = normalized.removesuffix(r"\%").removesuffix("%").strip()

    frac_match = re.fullmatch(
        r"\\(?:d?frac)\{([-+]?(?:\d+(?:\.\d+)?|\.\d+))\}"
        r"\{([-+]?(?:\d+(?:\.\d+)?|\.\d+))\}",
        normalized,
    )
    if frac_match is None:
        # MATH solutions also use the compact TeX form ``\frac{270}7``.
        frac_match = re.fullmatch(
            r"\\(?:d?frac)\{([-+]?(?:\d+(?:\.\d+)?|\.\d+))\}"
            r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))",
            normalized,
        )
    plain_fraction = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))/"
        r"([-+]?(?:\d+(?:\.\d+)?|\.\d+))",
        normalized,
    )
    match = frac_match or plain_fraction
    try:
        if match:
            denominator = Decimal(match.group(2))
            if denominator == 0:
                return None
            result = Decimal(match.group(1)) / denominator
        else:
            result = Decimal(normalized)
        if not result.is_finite():
            return None
        return result / Decimal(100) if percent else result
    except (InvalidOperation, ValueError):
        return None


def _numeric_value_with_optional_unit(value: str) -> Optional[Decimal]:
    """Parse a numeric answer with a trailing unit or degree marker.

    Units are not part of the mathematical value for MATH answers.  The
    suffix is restricted to TeX text commands, a degree marker, or words
    without digits so expressions such as ``1 and 2`` are not collapsed to
    the number ``1``.
    """
    normalized = value.strip()
    suffix_pattern = re.compile(
        r"^(?P<core>.+?)(?P<suffix>\s*(?:\^\s*\\circ|°|"
        r"\\(?:text|mathrm|mathbf)\{[^{}]*\}|"
        r"[A-Za-z][A-Za-z\s-]*))\s*$",
        flags=re.DOTALL,
    )
    match = suffix_pattern.match(normalized)
    if match is None:
        return None
    suffix = match.group("suffix")
    if re.search(r"\d", suffix):
        return None
    return _numeric_value(match.group("core").strip())


def _normalize_text(value: str) -> str:
    result = value.lower().strip()
    result = result.replace(r"\left", "").replace(r"\right", "")
    result = result.replace(r"\,", "")
    result = re.sub(
        r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}",
        r"\1",
        result,
    )
    result = result.strip("$")
    result = re.sub(r"\s+", "", result)
    choice = re.fullmatch(r"[\(\[]([a-e])[\)\]]", result)
    if choice:
        return choice.group(1)
    return result.rstrip(".")


def _structured_equal(predicted: str, reference: str, depth: int = 0) -> bool:
    """Compare finite sets, tuples/intervals, and unions conservatively."""
    if depth > 3:
        return False
    left = _strip_layout_commands(predicted)
    right = _strip_layout_commands(reference)

    left_union = _split_top_level_union(left)
    right_union = _split_top_level_union(right)
    if len(left_union) > 1 or len(right_union) > 1:
        if len(left_union) != len(right_union):
            return False
        return _unordered_items_equal(left_union, right_union, depth + 1)

    left_set = _finite_set_items(left)
    right_set = _finite_set_items(right)
    if left_set is not None or right_set is not None:
        if left_set is None or right_set is None:
            return False
        return _unordered_items_equal(left_set, right_set, depth + 1)

    left_sequence = _sequence_items(left)
    right_sequence = _sequence_items(right)
    if left_sequence is not None or right_sequence is not None:
        if left_sequence is None or right_sequence is None:
            return False
        left_brackets, left_items = left_sequence
        right_brackets, right_items = right_sequence
        return (
            left_brackets == right_brackets
            and len(left_items) == len(right_items)
            and all(
                _answer_atom_equal(a, b, depth + 1)
                for a, b in zip(left_items, right_items)
            )
        )
    return False


def _answer_atom_equal(left: str, right: str, depth: int) -> bool:
    left_num = _numeric_value(left)
    right_num = _numeric_value(right)
    if left_num is not None and right_num is not None:
        difference = abs(left_num - right_num)
        scale = max(Decimal(1), abs(left_num), abs(right_num))
        return difference <= Decimal("1e-9") * scale
    if _normalize_text(left) == _normalize_text(right):
        return True
    if _structured_equal(left, right, depth):
        return True
    return _safe_symbolic_equal(left, right)


def _unordered_items_equal(
    left: list[str],
    right: list[str],
    depth: int,
) -> bool:
    if (
        len(left) != len(right)
        or len(left) > _MAX_COLLECTION_ITEMS
    ):
        return False
    unmatched = list(right)
    for item in left:
        for index, candidate in enumerate(unmatched):
            if _answer_atom_equal(item, candidate, depth):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def _strip_layout_commands(value: str) -> str:
    return (
        _clean_answer(value)
        .replace(r"\left", "")
        .replace(r"\right", "")
        .replace(r"\,", "")
        .replace(r"\!", "")
        .strip()
    )


def _split_top_level_union(value: str) -> list[str]:
    return _split_top_level_tokens(value, (r"\cup", "∪"))


def _finite_set_items(value: str) -> Optional[list[str]]:
    stripped = value.strip()
    if stripped.startswith(r"\{") and stripped.endswith(r"\}"):
        content = stripped[2:-2]
    elif stripped.startswith("{") and stripped.endswith("}"):
        content = stripped[1:-1]
    else:
        return None
    if not content.strip():
        return []
    items = _split_top_level_tokens(content, (",",))
    if len(items) > _MAX_COLLECTION_ITEMS:
        return None
    return items


def _sequence_items(
    value: str,
) -> Optional[tuple[tuple[str, str], list[str]]]:
    stripped = value.strip()
    if (
        len(stripped) < 3
        or stripped[0] not in "(["
        or stripped[-1] not in ")]"
    ):
        return None
    items = _split_top_level_tokens(stripped[1:-1], (",",))
    if len(items) != 2:
        return None
    return (stripped[0], stripped[-1]), items


def _split_top_level_tokens(
    value: str,
    separators: tuple[str, ...],
) -> list[str]:
    """Split only outside nested TeX/Python grouping delimiters."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        if char in "([{":
            stack.append(char)
            index += 1
            continue
        if char in ")]}":
            # Mathematical intervals legitimately mix ``(`` with ``]`` (and
            # ``[`` with ``)``), so either round/square closer ends that group.
            if (
                stack
                and (
                    (char in ")]" and stack[-1] in "([")
                    or (char == "}" and stack[-1] == "{")
                )
            ):
                stack.pop()
            index += 1
            continue
        if not stack:
            separator = next(
                (
                    token
                    for token in separators
                    if value.startswith(token, index)
                ),
                None,
            )
            if separator is not None:
                part = value[start:index].strip()
                if not part:
                    return [value]
                parts.append(part)
                index += len(separator)
                start = index
                continue
        index += 1
    final = value[start:].strip()
    if not final:
        return [value]
    parts.append(final)
    return parts


def _safe_symbolic_equal(predicted: str, reference: str) -> bool:
    """Compare restricted arithmetic without evaluating either input string."""
    left = _canonical_expression(predicted)
    right = _canonical_expression(reference)
    if left is None or right is None or left[1] != right[1]:
        return False

    left_text, equations = left
    right_text, _ = right
    try:
        left_tree = _parse_restricted_ast(left_text)
        right_tree = _parse_restricted_ast(right_text)
    except (SyntaxError, ValueError):
        return False

    left_poly = _polynomial_from_ast(left_tree.body)
    right_poly = _polynomial_from_ast(right_tree.body)
    if left_poly is not None and right_poly is not None:
        return _polynomials_equal(
            left_poly,
            right_poly,
            proportional=equations,
        )

    # Rational equations can change their solution set at excluded denominator
    # roots. Keep equation comparison polynomial-only instead of allowing a
    # simplifier to cancel those domain restrictions.
    if equations or not _SYMPY_AVAILABLE:
        return False
    return _sympy_equal_with_timeout(
        left_text,
        right_text,
        equations,
    )


def _canonical_expression(value: str) -> Optional[tuple[str, bool]]:
    stripped = _strip_layout_commands(value)
    if not stripped or len(stripped) > _MAX_SYMBOLIC_INPUT:
        return None

    if stripped.count("=") == 1 and not re.search(r"[<>!]=|=>|=<", stripped):
        left, right = stripped.split("=", 1)
        left_expr = _latex_to_python_expression(left)
        right_expr = _latex_to_python_expression(right)
        if left_expr is None or right_expr is None:
            return None
        return f"({left_expr})-({right_expr})", True

    expression = _latex_to_python_expression(stripped)
    return (expression, False) if expression is not None else None


def _latex_to_python_expression(value: str) -> Optional[str]:
    text = value.strip().replace("−", "-").replace("·", "*")
    text = text.replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\dfrac", r"\frac").replace(
        r"\tfrac",
        r"\frac",
    )
    text = text.replace(r"\pi", "pi")
    text = re.sub(
        r"([A-Za-z])_\{([A-Za-z0-9]+)\}",
        r"\1_\2",
        text,
    )
    text = re.sub(
        r"([A-Za-z])_([A-Za-z0-9]+)",
        r"\1_\2",
        text,
    )

    for _ in range(32):
        position = text.rfind(r"\frac")
        if position < 0:
            break
        numerator = _read_braced_group(text, position + len(r"\frac"))
        if numerator is None:
            return None
        numerator_text, after_numerator = numerator
        denominator = _read_braced_group(text, after_numerator)
        if denominator is None:
            return None
        denominator_text, after_denominator = denominator
        text = (
            text[:position]
            + f"(({numerator_text})/({denominator_text}))"
            + text[after_denominator:]
        )
    if r"\frac" in text:
        return None

    for _ in range(32):
        position = text.rfind(r"\sqrt")
        if position < 0:
            break
        radicand = _read_braced_group(text, position + len(r"\sqrt"))
        if radicand is None:
            return None
        radicand_text, after_radicand = radicand
        text = (
            text[:position]
            + f"sqrt({radicand_text})"
            + text[after_radicand:]
        )
    if r"\sqrt" in text or "\\" in text:
        return None

    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("^", "**")
    text = re.sub(r"\s+", "", text)
    return _tokenize_and_insert_multiplication(text)


def _read_braced_group(
    value: str,
    start: int,
) -> Optional[tuple[str, int]]:
    if start >= len(value) or value[start] != "{":
        return None
    depth = 1
    index = start + 1
    while index < len(value) and depth:
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
        index += 1
    if depth:
        return None
    return value[start + 1:index - 1], index


def _tokenize_and_insert_multiplication(value: str) -> Optional[str]:
    token_pattern = re.compile(
        r"(?:\d+\.\d*|\.\d+|\d+)"
        r"|(?:[A-Za-z][A-Za-z0-9_]*)"
        r"|(?:\*\*|[+\-*/(),])"
    )
    tokens: list[str] = []
    position = 0
    while position < len(value):
        match = token_pattern.match(value, position)
        if match is None:
            return None
        tokens.append(match.group(0))
        position = match.end()
    if not tokens:
        return None

    functions = {"sqrt", "abs"}
    result: list[str] = []
    for token in tokens:
        if result:
            previous = result[-1]
            previous_ends_value = (
                previous == ")"
                or previous[0].isdigit()
                or previous[0].isalpha()
            )
            token_starts_value = (
                token == "("
                or token[0].isdigit()
                or token[0].isalpha()
            )
            is_function_call = previous in functions and token == "("
            if previous_ends_value and token_starts_value and not is_function_call:
                result.append("*")
        result.append(token)
    return "".join(result)


def _parse_restricted_ast(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_AST_NODES:
        raise ValueError("symbolic expression is too complex")
    if _ast_depth(tree) > _MAX_AST_DEPTH:
        raise ValueError("symbolic expression is too deeply nested")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Load,
    )
    if any(not isinstance(node, allowed) for node in nodes):
        raise ValueError("unsupported symbolic syntax")
    for node in nodes:
        if isinstance(node, ast.Name) and (
            len(node.id) > 24
            or node.id.startswith("_")
        ):
            raise ValueError("invalid symbolic name")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in {"sqrt", "abs"}
            or len(node.args) != 1
            or node.keywords
        ):
            raise ValueError("unsupported symbolic function")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exponent = _literal_number(node.right)
            if (
                exponent is None
                or exponent.denominator != 1
                or abs(exponent) > _MAX_POWER
            ):
                raise ValueError("symbolic exponent is unsupported or too large")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value,
                (int, float),
            ):
                raise ValueError("unsupported symbolic constant")
            digits = len(str(node.value).replace(".", "").replace("-", ""))
            if digits > _MAX_INTEGER_DIGITS:
                raise ValueError("symbolic constant is too large")
    return tree


def _literal_number(node: ast.AST) -> Optional[Fraction]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        try:
            return Fraction(str(node.value))
        except (ValueError, ZeroDivisionError):
            return None
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        value = _literal_number(node.operand)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + max((_ast_depth(child) for child in children), default=0)


Monomial = tuple[tuple[str, int], ...]
Polynomial = dict[Monomial, Fraction]


def _polynomial_from_ast(node: ast.AST) -> Optional[Polynomial]:
    if isinstance(node, ast.Constant):
        try:
            value = Fraction(str(node.value))
        except (ValueError, ZeroDivisionError):
            return None
        return {(): value} if value else {}
    if isinstance(node, ast.Name):
        if node.id in {"pi", "e"}:
            return None
        return {((node.id, 1),): Fraction(1)}
    if isinstance(node, ast.UnaryOp):
        operand = _polynomial_from_ast(node.operand)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return {term: -value for term, value in operand.items()}
        return None
    if isinstance(node, ast.Call):
        argument = _polynomial_from_ast(node.args[0])
        if argument is None or set(argument) - {()}:
            return None
        value = argument.get((), Fraction(0))
        if isinstance(node.func, ast.Name) and node.func.id == "abs":
            return {(): abs(value)} if value else {}
        if isinstance(node.func, ast.Name) and node.func.id == "sqrt":
            if value < 0:
                return None
            numerator_root = isqrt(value.numerator)
            denominator_root = isqrt(value.denominator)
            if (
                numerator_root * numerator_root != value.numerator
                or denominator_root * denominator_root != value.denominator
            ):
                return None
            result = Fraction(numerator_root, denominator_root)
            return {(): result} if result else {}
        return None
    if not isinstance(node, ast.BinOp):
        return None

    left = _polynomial_from_ast(node.left)
    right = _polynomial_from_ast(node.right)
    if left is None or right is None:
        return None
    if isinstance(node.op, ast.Add):
        return _poly_add(left, right)
    if isinstance(node.op, ast.Sub):
        return _poly_add(
            left,
            {term: -value for term, value in right.items()},
        )
    if isinstance(node.op, ast.Mult):
        return _poly_multiply(left, right)
    if isinstance(node.op, ast.Div):
        if set(right) - {()} or right.get((), Fraction(0)) == 0:
            return None
        denominator = right[()]
        return _poly_checked(
            {term: value / denominator for term, value in left.items()}
        )
    if isinstance(node.op, ast.Pow):
        if set(right) - {()}:
            return None
        exponent = right.get((), Fraction(0))
        if exponent.denominator != 1 or not 0 <= exponent <= _MAX_POWER:
            return None
        result: Polynomial = {(): Fraction(1)}
        factor = left
        power = int(exponent)
        while power:
            if power & 1:
                result = _poly_multiply(result, factor)
                if result is None:
                    return None
            power //= 2
            if power:
                factor = _poly_multiply(factor, factor)
                if factor is None:
                    return None
        return result
    return None


def _poly_add(left: Polynomial, right: Polynomial) -> Optional[Polynomial]:
    result = dict(left)
    for term, value in right.items():
        result[term] = result.get(term, Fraction(0)) + value
    return _poly_checked(result)


def _poly_multiply(
    left: Polynomial,
    right: Polynomial,
) -> Optional[Polynomial]:
    if len(left) * len(right) > _MAX_POLYNOMIAL_TERMS * 4:
        return None
    result: Polynomial = {}
    for left_term, left_value in left.items():
        for right_term, right_value in right.items():
            powers: dict[str, int] = dict(left_term)
            for name, exponent in right_term:
                powers[name] = powers.get(name, 0) + exponent
                if powers[name] > _MAX_POWER:
                    return None
            term = tuple(sorted(powers.items()))
            result[term] = (
                result.get(term, Fraction(0))
                + left_value * right_value
            )
    return _poly_checked(result)


def _poly_checked(value: Polynomial) -> Optional[Polynomial]:
    result = {
        term: coefficient
        for term, coefficient in value.items()
        if coefficient
    }
    if len(result) > _MAX_POLYNOMIAL_TERMS:
        return None
    if any(
        len(str(abs(coefficient.numerator))) > _MAX_INTEGER_DIGITS * 4
        or len(str(coefficient.denominator)) > _MAX_INTEGER_DIGITS * 4
        for coefficient in result.values()
    ):
        return None
    return result


def _polynomials_equal(
    left: Polynomial,
    right: Polynomial,
    *,
    proportional: bool,
) -> bool:
    if left == right:
        return True
    if not proportional or not left or not right or set(left) != set(right):
        return False
    first_term = next(iter(left))
    factor = left[first_term] / right[first_term]
    return factor != 0 and all(
        left[term] == factor * right[term]
        for term in left
    )


def _sympy_equal_with_timeout(
    left: str,
    right: str,
    equations: bool,
) -> bool:
    methods = multiprocessing.get_all_start_methods()
    method = "fork" if "fork" in methods else "spawn"
    context = multiprocessing.get_context(method)
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_sympy_worker,
        args=(left, right, equations, sender),
        daemon=True,
    )
    try:
        process.start()
        sender.close()
        if receiver.poll(_SYMPY_TIMEOUT_SECONDS):
            try:
                return bool(receiver.recv())
            except EOFError:
                return False
        process.kill()
        return False
    except (OSError, RuntimeError):
        return False
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(timeout=0.1)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.1)


def _sympy_worker(
    left: str,
    right: str,
    equations: bool,
    sender,
) -> None:
    """Build SymPy values from validated AST nodes, never from raw parsing."""
    try:
        sympy = _sympy_module
        if sympy is None:
            raise RuntimeError("SymPy is unavailable")

        left_tree = _parse_restricted_ast(left)
        right_tree = _parse_restricted_ast(right)
        symbols: dict[str, object] = {}

        def convert(node: ast.AST):
            if isinstance(node, ast.Constant):
                return sympy.Rational(str(node.value))
            if isinstance(node, ast.Name):
                if node.id == "pi":
                    return sympy.pi
                if node.id == "e":
                    return sympy.E
                return symbols.setdefault(node.id, sympy.Symbol(node.id))
            if isinstance(node, ast.UnaryOp):
                value = convert(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.Call):
                value = convert(node.args[0])
                if node.func.id == "sqrt":
                    return sympy.sqrt(value)
                return sympy.Abs(value)
            if isinstance(node, ast.BinOp):
                lhs = convert(node.left)
                rhs = convert(node.right)
                if isinstance(node.op, ast.Add):
                    return lhs + rhs
                if isinstance(node.op, ast.Sub):
                    return lhs - rhs
                if isinstance(node.op, ast.Mult):
                    return lhs * rhs
                if isinstance(node.op, ast.Div):
                    return lhs / rhs
                if isinstance(node.op, ast.Pow):
                    return lhs ** rhs
            raise ValueError("unsupported symbolic node")

        lhs = convert(left_tree.body)
        rhs = convert(right_tree.body)
        if equations:
            if lhs == 0 or rhs == 0:
                equal = lhs == rhs
            else:
                ratio = sympy.simplify(lhs / rhs)
                equal = bool(ratio != 0 and not ratio.free_symbols)
        else:
            equal = bool(sympy.simplify(lhs - rhs) == 0)
        sender.send(equal)
    except BaseException:
        try:
            sender.send(False)
        except BaseException:
            pass
    finally:
        sender.close()
