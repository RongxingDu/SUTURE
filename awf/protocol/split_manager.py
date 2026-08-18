"""SplitManager — reproducible opt/val/test splits."""

from __future__ import annotations

import random
from typing import Any, Optional


class SplitManager:
    """Manages reproducible data splits for optimization, validation, and testing.

    Uses seeded random shuffling for reproducibility.
    """

    def __init__(
        self,
        seed: int = 42,
        opt_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ):
        if any(r <= 0.0 or r >= 1.0 for r in (opt_ratio, val_ratio, test_ratio)):
            raise ValueError("Each split ratio must be strictly between 0 and 1")
        if abs(opt_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got "
                f"opt={opt_ratio} + val={val_ratio} + test={test_ratio}"
            )

        self.seed = seed
        self.opt_ratio = opt_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(
        self,
        items: list[Any],
        key_fn: Optional[callable] = None,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Split items into (opt, val, test) sets.

        Args:
            items: List of items to split.
            key_fn: Optional function to extract a stable key from each item.
                    If provided, uses hash-based deterministic ordering.

        Returns:
            Tuple of (opt_items, val_items, test_items).
        """
        n = len(items)
        rng = random.Random(self.seed)

        if key_fn is not None:
            # Keep equal task keys in the same split so duplicated benchmark
            # rows cannot leak from optimization into validation/test.
            items = sorted(
                items,
                key=lambda item: _stable_hash(str(key_fn(item))),
            )
            grouped: dict[str, list[Any]] = {}
            for item in items:
                grouped.setdefault(str(key_fn(item)), []).append(item)
            groups = list(grouped.values())
            rng.shuffle(groups)

            opt_target = int(n * self.opt_ratio)
            val_target = int(n * self.val_ratio)
            opt_items: list[Any] = []
            val_items: list[Any] = []
            test_items: list[Any] = []
            for group in groups:
                if len(opt_items) < opt_target:
                    opt_items.extend(group)
                elif len(val_items) < val_target:
                    val_items.extend(group)
                else:
                    test_items.extend(group)
            return opt_items, val_items, test_items

        # Shuffle item indices with fixed seed.
        indices = list(range(n))
        rng.shuffle(indices)

        opt_end = int(n * self.opt_ratio)
        val_end = opt_end + int(n * self.val_ratio)

        opt_items = [items[i] for i in indices[:opt_end]]
        val_items = [items[i] for i in indices[opt_end:val_end]]
        test_items = [items[i] for i in indices[val_end:]]

        return opt_items, val_items, test_items

    def split_source(
        self,
        items: list[Any],
        field: str,
        dev_value: str = "validate",
        test_value: str = "test",
        key_fn: Optional[callable] = None,
        reuse_dev: bool = False,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Partition items by a source-split metadata field.

        Rows whose ground-truth metadata value equals ``test_value`` form the
        held-out test split. All remaining rows form a development pool that is
        further divided into optimization/validation by the configured ratios
        (renormalized so the dev pool is fully consumed with no discard).
        When ``reuse_dev`` is True the whole development pool is returned for
        BOTH optimization and validation (opt_items == val_items), matching the
        common single-split practice in AFlow-style papers where the validate
        set is used for optimization and validation simultaneously.

        Args:
            items: List of items to split.
            field: Ground-truth metadata field that carries the split value.
            dev_value: Field value denoting development (opt/val) rows.
            test_value: Field value denoting held-out test rows.
            key_fn: Optional function to extract a stable key for splitting.
            reuse_dev: If True, opt and val both receive the full dev pool.

        Returns:
            Tuple of (opt_items, val_items, test_items).
        """
        dev_items: list[Any] = []
        test_items: list[Any] = []
        for item in items:
            field_value = None
            payload = item[1] if isinstance(item, tuple) and len(item) == 2 else item
            if isinstance(payload, tuple) and len(payload) == 2:
                ground_truth = payload[1]
                if isinstance(ground_truth, dict):
                    field_value = ground_truth.get(field)
            if field_value == test_value:
                test_items.append(item)
            else:
                dev_items.append(item)
        if not test_items:
            raise ValueError(
                f"No held-out test rows found for source-split field "
                f"{field!r} == {test_value!r}"
            )
        if not dev_items:
            raise ValueError(
                f"No development rows found for source-split field "
                f"{field!r} != {test_value!r} (expected {dev_value!r})"
            )
        if reuse_dev:
            # Single-split practice: optimization and validation share the
            # same development rows; only test is held out.
            return dev_items, dev_items, test_items
        n_dev = len(dev_items)
        opt_denom = self.opt_ratio + self.val_ratio
        opt_fraction = self.opt_ratio / opt_denom if opt_denom > 0 else 0.5
        rng = random.Random(self.seed)

        if key_fn is not None:
            dev_items = sorted(
                dev_items,
                key=lambda item: _stable_hash(str(key_fn(item))),
            )
            grouped: dict[str, list[Any]] = {}
            for item in dev_items:
                grouped.setdefault(str(key_fn(item)), []).append(item)
            groups = list(grouped.values())
            rng.shuffle(groups)
            opt_target = int(n_dev * opt_fraction)
            opt_items: list[Any] = []
            val_items: list[Any] = []
            for group in groups:
                if len(opt_items) < opt_target:
                    opt_items.extend(group)
                else:
                    val_items.extend(group)
            return opt_items, val_items, test_items

        indices = list(range(n_dev))
        rng.shuffle(indices)
        opt_end = int(n_dev * opt_fraction)
        dev_ordered = [dev_items[i] for i in indices]
        return dev_ordered[:opt_end], dev_ordered[opt_end:], test_items

    def get_split_indices(
        self,
        n: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Get the split indices for n items.

        Returns:
            Tuple of (opt_indices, val_indices, test_indices).
        """
        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)

        opt_end = int(n * self.opt_ratio)
        val_end = opt_end + int(n * self.val_ratio)

        return (
            sorted(indices[:opt_end]),
            sorted(indices[opt_end:val_end]),
            sorted(indices[val_end:]),
        )


def _stable_hash(key: str) -> int:
    """Create a stable integer hash from a string key."""
    h = 0
    for c in key:
        h = (h * 31 + ord(c)) & 0xFFFFFFFFFFFFFFFF
    return h % (2**31)
