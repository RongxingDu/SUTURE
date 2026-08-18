"""SuffixReplayEngine — cache prefix, re-execute suffix.

Optimization: when a workflow edit only affects a suffix of the execution,
reuse the cached prefix results instead of re-executing everything.
"""

from __future__ import annotations

from typing import Any, Optional


class CachedPrefix:
    """Cached prefix state from a previous execution."""

    def __init__(self, node_outputs: dict[str, Any],
                 variables: dict[str, Any],
                 last_node_id: str):
        self.node_outputs = dict(node_outputs)
        self.variables = dict(variables)
        self.last_node_id = last_node_id


class SuffixReplayEngine:
    """Caches execution prefixes and replays from the edited point onward.

    When a workflow edit only modifies a suffix (later nodes), the prefix
    (earlier nodes) is unchanged and can be reused. This avoids redundant
    LLM calls for the unchanged portion.
    """

    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        # Cache: query_id -> CachedPrefix
        self._cache: dict[str, CachedPrefix] = {}

    def cache_prefix(self, query_id: str, node_outputs: dict[str, Any],
                     variables: dict[str, Any],
                     last_node_id: str) -> None:
        """Cache the prefix of an execution for later replay.

        Args:
            query_id: Identifier for the query.
            node_outputs: Outputs from executed nodes.
            variables: Shared variables at the end of the prefix.
            last_node_id: The last node executed in the prefix.
        """
        if len(self._cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[query_id] = CachedPrefix(
            node_outputs=node_outputs,
            variables=variables,
            last_node_id=last_node_id,
        )

    def get_prefix(self, query_id: str) -> Optional[CachedPrefix]:
        """Retrieve a cached prefix if available.

        Returns None if no cache entry exists for this query.
        """
        return self._cache.get(query_id)

    def is_suffix_edit(
        self,
        edit_node_id: str,
        workflow_order: list[str],
        cached_prefix: CachedPrefix,
    ) -> bool:
        """Check if the edit affects only a suffix.

        An edit is a suffix edit if the edited node appears after
        the cached prefix's last node in the workflow order.
        """
        try:
            edit_index = workflow_order.index(edit_node_id)
            prefix_index = workflow_order.index(cached_prefix.last_node_id)
            return edit_index >= prefix_index
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
