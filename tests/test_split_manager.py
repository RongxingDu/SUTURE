"""Tests for SplitManager."""

from awf.protocol.split_manager import SplitManager


def _source_items(n_dev: int, n_test: int):
    """Build (index, (query, gt)) items carrying a ``source_split`` field."""
    items = []
    for i in range(n_dev):
        items.append((i, (f"q-dev-{i}", {"answer": "a", "source_split": "validate"})))
    for i in range(n_test):
        j = n_dev + i
        items.append((j, (f"q-test-{i}", {"answer": "b", "source_split": "test"})))
    return items


class TestSplitManager:
    """Test data splitting."""

    def test_split_proportions(self):
        items = list(range(100))
        manager = SplitManager(seed=42, opt_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        opt, val, test = manager.split(items)

        assert len(opt) == 60
        assert len(val) == 20
        assert len(test) == 20
        # All items accounted for
        all_items = set(opt) | set(val) | set(test)
        assert all_items == set(range(100))
        # No overlap
        assert len(set(opt) & set(val)) == 0
        assert len(set(opt) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_reproducibility(self):
        items = list(range(50))
        manager1 = SplitManager(seed=42)
        manager2 = SplitManager(seed=42)

        opt1, val1, test1 = manager1.split(items)
        opt2, val2, test2 = manager2.split(items)

        assert opt1 == opt2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds_different_splits(self):
        items = list(range(50))
        manager1 = SplitManager(seed=1)
        manager2 = SplitManager(seed=2)

        opt1, _, _ = manager1.split(items)
        opt2, _, _ = manager2.split(items)

        assert opt1 != opt2

    def test_invalid_ratios(self):
        import pytest
        with pytest.raises(ValueError):
            SplitManager(opt_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_key_fn_deterministic(self):
        items = ["c", "a", "b", "d"]
        manager = SplitManager(seed=42)
        opt, val, test = manager.split(items, key_fn=lambda x: x)

        all_items = opt + val + test
        # With key_fn, order is deterministic by hash
        assert len(all_items) == 4

    def test_key_fn_keeps_duplicate_tasks_in_one_split(self):
        items = [
            ("duplicate", 1),
            ("duplicate", 2),
            ("unique-a", 3),
            ("unique-b", 4),
            ("unique-c", 5),
            ("unique-d", 6),
        ]
        manager = SplitManager(seed=42, opt_ratio=0.5, val_ratio=0.25, test_ratio=0.25)

        splits = manager.split(items, key_fn=lambda item: item[0])

        containing_splits = [
            split
            for split in splits
            if any(item[0] == "duplicate" for item in split)
        ]
        assert len(containing_splits) == 1
        assert sum(
            item[0] == "duplicate" for item in containing_splits[0]
        ) == 2

    def test_split_source_partitions_by_field(self):
        items = _source_items(n_dev=119, n_test=486)
        manager = SplitManager(seed=7, opt_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        opt, val, test = manager.split_source(
            items,
            field="source_split",
            dev_value="validate",
            test_value="test",
        )
        assert len(test) == 486
        assert len(opt) + len(val) == 119
        # opt/val disjoint, cover dev rows only
        assert set(i for i, _ in opt).isdisjoint(i for i, _ in val)
        assert all(g["source_split"] == "test" for _, (_, g) in test)
        assert all(g["source_split"] == "validate" for _, (_, g) in opt + val)

    def test_split_source_reuse_dev_opt_equals_val(self):
        items = _source_items(n_dev=119, n_test=486)
        manager = SplitManager(seed=7, opt_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        opt, val, test = manager.split_source(
            items,
            field="source_split",
            dev_value="validate",
            test_value="test",
            reuse_dev=True,
        )
        # opt and val are the same development rows; test is held out.
        assert len(opt) == len(val) == 119
        assert len(test) == 486
        assert [i for i, _ in opt] == [i for i, _ in val]
        assert set(i for i, _ in test).isdisjoint(i for i, _ in opt)
