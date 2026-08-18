"""CheckpointManager — best checkpoint tracking on validation."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from awf.workflow.ir import WorkflowTemplate
from awf.workflow.serializer import dump_workflow, load_workflow
from awf.protocol.manifest import atomic_write_json_0600


class CheckpointManager:
    """Tracks the best workflow checkpoint based on validation performance.

    Saves checkpoints to disk and maintains a best-so-far record.
    """

    def __init__(
        self,
        output_dir: str | Path,
        maximize: bool = True,
        min_delta: float = 0.0,
    ):
        """
        Args:
            output_dir: Directory to save checkpoints.
            maximize: If True, higher scores are better; if False, lower is better.
            min_delta: Strict minimum score improvement required after the
                initial checkpoint.
        """
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite non-negative number")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.maximize = maximize
        self.min_delta = float(min_delta)

        self.best_score: Optional[float] = None
        self.best_workflow: Optional[WorkflowTemplate] = None
        self.best_round: int = -1
        self.checkpoint_history: list[dict] = []
        self._saved_checkpoint_count: int = 0

    def update(
        self,
        workflow: WorkflowTemplate,
        score: float,
        round_num: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update the checkpoint with a new candidate.

        Args:
            workflow: The workflow to potentially save.
            score: Validation score for this workflow.
            round_num: The optimization round number.
            metadata: Additional metadata to save.

        Returns:
            True if this is a new best (checkpoint was saved).
        """
        previous_best = self.best_score
        improvement: float | None = None
        if self.best_score is None:
            is_better = True
        elif self.maximize:
            improvement = score - self.best_score
            is_better = score > self.best_score + self.min_delta
        else:
            improvement = self.best_score - score
            is_better = score < self.best_score - self.min_delta

        entry = {
            "round": round_num,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "previous_best_score": previous_best,
            "improvement": improvement,
            "min_delta": self.min_delta,
            "accepted": is_better,
        }
        self.checkpoint_history.append(entry)

        if is_better:
            self.best_score = score
            self.best_workflow = copy.deepcopy(workflow)
            self.best_round = round_num
            self._save_checkpoint(workflow, score, round_num, metadata)
            self._saved_checkpoint_count += 1
            return True

        return False

    def _save_checkpoint(
        self,
        workflow: WorkflowTemplate,
        score: float | None,
        round_num: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint to disk."""
        workflow_path = self.output_dir / "best_workflow.yaml"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".best_workflow.",
            # Keep a recognized extension because dump_workflow chooses the
            # serializer from the path suffix.
            suffix=".yaml",
            dir=self.output_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            dump_workflow(workflow, temporary_path)
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, workflow_path)
            os.chmod(workflow_path, 0o600)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

        meta = {
            "round": round_num,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "min_delta": self.min_delta,
        }
        meta_path = self.output_dir / "checkpoint_meta.json"
        atomic_write_json_0600(meta_path, meta)

    def load_best(self) -> Optional[WorkflowTemplate]:
        """Load the best checkpoint from disk."""
        workflow_path = self.output_dir / "best_workflow.yaml"
        if workflow_path.exists():
            self.best_workflow = load_workflow(workflow_path)
            meta_path = self.output_dir / "checkpoint_meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                self.best_score = metadata.get("score")
                self.best_round = metadata.get("round", -1)
                self._saved_checkpoint_count = max(
                    self._saved_checkpoint_count,
                    1,
                )
            return self.best_workflow
        return None

    def promote_local(
        self,
        workflow: WorkflowTemplate,
        round_num: int,
        *,
        score: float | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a locally accepted workflow without a validation score.

        Aggressive counterfactual experiments deliberately remove the full
        validation promotion gate.  They still need a manifest-compatible
        ``best_workflow.yaml`` so the independent held-out evaluator can load
        the final incumbent.  This method records the local score (when one is
        available) as audit metadata; it never compares it against a previous
        validation score.
        """
        self.best_score = score
        self.best_workflow = copy.deepcopy(workflow)
        self.best_round = int(round_num)
        self.checkpoint_history.append(
            {
                "round": int(round_num),
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "previous_best_score": None,
                "improvement": None,
                "min_delta": self.min_delta,
                "accepted": True,
                "promotion_mode": "counterfactual_local_aggressive",
            }
        )
        self._save_checkpoint(workflow, score, round_num, metadata)
        self._saved_checkpoint_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of checkpoint history."""
        return {
            "best_score": self.best_score,
            "best_round": self.best_round,
            "min_delta": self.min_delta,
            # This manager intentionally keeps one validation-selected file.
            "num_checkpoints": int(self.best_workflow is not None),
            "num_checkpoint_updates": self._saved_checkpoint_count,
            "num_evaluations": len(self.checkpoint_history),
            "history": self.checkpoint_history,
        }
