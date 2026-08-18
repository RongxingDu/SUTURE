"""ParameterHierarchy: stage → block → operator → prompt.

Organizes workflow parameters in a hierarchical structure for
targeted editing during optimization.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictParameterModel(BaseModel):
    """Parameter schema base that rejects unknown hierarchy fields."""

    model_config = ConfigDict(extra="forbid")


class PromptParams(_StrictParameterModel):
    """Parameters at the prompt level (system prompt, user prompt template)."""

    system_prompt: Optional[str] = None
    user_template: Optional[str] = None
    # Additional prompt-level knobs
    extra: dict[str, Any] = Field(default_factory=dict)


class OperatorParams(_StrictParameterModel):
    """Parameters at the operator (node) level."""

    node_id: str = Field(min_length=1)
    # LLM-specific parameters
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    # Tool-specific parameters
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    # Prompt parameters for this operator
    prompt: PromptParams = Field(default_factory=PromptParams)


class BlockParams(_StrictParameterModel):
    """Parameters for a block (a subgraph of related operators)."""

    block_id: str = Field(min_length=1)
    label: str = ""
    operators: dict[str, OperatorParams] = Field(default_factory=dict)
    # Block-level metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageParams(_StrictParameterModel):
    """Parameters for a stage (a pipeline phase containing blocks)."""

    stage_id: str = Field(min_length=1)
    label: str = ""
    blocks: dict[str, BlockParams] = Field(default_factory=dict)
    # Stage-level metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParameterHierarchy(_StrictParameterModel):
    """Full hierarchical parameter structure for a workflow template.

    Hierarchy: stage → block → operator → prompt
    """

    stages: dict[str, StageParams] = Field(default_factory=dict)
    # Global parameters shared across all stages
    global_params: dict[str, Any] = Field(default_factory=dict)

    def get_operator(self, stage_id: str, block_id: str, node_id: str) -> Optional[OperatorParams]:
        """Get operator params by path."""
        stage = self.stages.get(stage_id)
        if stage is None:
            return None
        block = stage.blocks.get(block_id)
        if block is None:
            return None
        return block.operators.get(node_id)

    def set_operator(self, stage_id: str, block_id: str, operator: OperatorParams) -> None:
        """Set or update an operator in the hierarchy."""
        if stage_id not in self.stages:
            self.stages[stage_id] = StageParams(stage_id=stage_id)
        stage = self.stages[stage_id]
        if block_id not in stage.blocks:
            stage.blocks[block_id] = BlockParams(block_id=block_id)
        stage.blocks[block_id].operators[operator.node_id] = operator
