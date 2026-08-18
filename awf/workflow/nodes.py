"""Node, NodeType, and per-type NodeConfig definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class _StrictWorkflowModel(BaseModel):
    """Workflow schema base that rejects unknown serialized fields."""

    model_config = ConfigDict(extra="forbid")


class NodeType(str, Enum):
    """Types of nodes in a workflow graph."""

    LLM = "llm"          # LLM call node
    TOOL = "tool"        # Tool/function call node
    CONDITION = "condition"  # Conditional branching node
    JOIN = "join"        # Merge/synchronization node
    START = "start"      # Entry point
    END = "end"          # Exit point


class NodeConfig(_StrictWorkflowModel):
    """Configuration for a single node in the workflow.

    Different node types use different subsets of fields.
    """

    node_type: NodeType = NodeType.LLM
    # LLM nodes: system/user prompt template
    prompt_template: Optional[str] = None
    system_prompt: Optional[str] = None
    # LLM nodes: model parameters
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    # Tool nodes: tool name to invoke
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    # Condition nodes: expression for branching
    condition_expr: Optional[str] = None
    # Generic metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class Node(_StrictWorkflowModel):
    """A node in the workflow graph."""

    node_id: str = Field(min_length=1)
    node_type: NodeType
    config: NodeConfig = Field(default_factory=NodeConfig)
    # Human-readable label
    label: str = ""
    # Position for layout (optional)
    position: tuple[float, float] = (0.0, 0.0)
