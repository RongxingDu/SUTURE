"""RuntimeExecutor — main step-by-step workflow execution loop."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections import defaultdict
from typing import Any, Callable, Optional

from awf.config.schema import ExecutorConfig
from awf.executor.conditions import (
    ConditionExpressionError,
    evaluate_condition_expression,
)
from awf.executor.context import ExecutionContext
from awf.executor.safety import is_counterfactual_safe
from awf.llm.client import AsyncLLMClient
from awf.llm.cost_tracker import estimate_cost_usd, has_known_pricing
from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.trace.recorder import TraceRecorder
from awf.trace.sanitize import to_trace_value
from awf.trace.schema import LLMCallRecord, ToolCallRecord
from awf.workflow.gates import realize_selective_update
from awf.workflow.ir import WorkflowTemplate
from awf.workflow.nodes import NodeType

# Type alias for operator implementations
OperatorFn = Callable[..., Any]


class RuntimeExecutor:
    """Run a workflow under a query-specific scheduler policy."""

    def __init__(
        self,
        config: ExecutorConfig,
        operators: Optional[dict[str, OperatorFn]] = None,
        counterfactual_safe_tools: Optional[set[str]] = None,
    ):
        self.config = config
        self.operators = operators or {}
        self.counterfactual_safe_tools = set(
            counterfactual_safe_tools or ()
        )
        self.counterfactual_safe_tools.update(
            name
            for name, operator in self.operators.items()
            if is_counterfactual_safe(operator)
        )

    async def execute(
        self,
        workflow: WorkflowTemplate,
        scheduler: BaseScheduler,
        query: str,
        llm_client: Optional[AsyncLLMClient] = None,
        counterfactual: bool = False,
    ) -> tuple[Any, ExecutionContext, TraceRecorder]:
        """Execute ``workflow`` and return output, mutable context, and trace."""
        declared_workflow = workflow
        workflow, gate_decision = realize_selective_update(
            declared_workflow,
            query,
        )
        context = ExecutionContext(
            query=query,
            workflow_name=declared_workflow.name,
        )
        recorder = TraceRecorder(
            query_id=query[:50],
            query_text=query,
            # Detailed in-memory traces are part of reward and utility
            # semantics, so they must not disappear when artifact persistence
            # is disabled. ExecutorConfig.trace_enabled controls only whether
            # ExperimentRunner writes those traces to disk.
            enabled=True,
        )
        recorder.start(
            declared_workflow.name,
            declared_workflow.version,
        )
        recorder.trace.metadata["execution_mode"] = (
            "counterfactual" if counterfactual else "normal"
        )
        if gate_decision is not None:
            recorder.trace.metadata["selective_update"] = (
                gate_decision.model_dump(mode="json")
            )

        try:
            await scheduler.initialize(workflow, query)
            entry = workflow.get_entry_node()
            context.set_current_node(entry.node_id)
        except Exception as exc:
            context.mark_finished(
                success=False,
                error_message=f"Execution initialization failed: {exc}",
            )
            recorder.end(
                success=False,
                final_output=None,
                error_message=context.error_message,
            )
            return context.final_output, context, recorder

        action_limit, limit_label = self._get_action_limit(scheduler)

        while not context.finished and context.step_count < action_limit:
            context.step_count += 1
            state_before = context.get_state_snapshot()
            step_id = recorder.record_step_start(
                node_id=context.current_node_id or "unknown",
                node_type=self._get_node_type(workflow, context.current_node_id),
                state_before=state_before,
            )

            step_success = True
            step_error: str | None = None
            try:
                await asyncio.wait_for(
                    self._select_and_apply(
                        workflow,
                        scheduler,
                        context,
                        llm_client,
                        recorder,
                        step_id,
                        counterfactual,
                    ),
                    timeout=self.config.timeout_per_step,
                )
            except asyncio.TimeoutError:
                step_success = False
                step_error = (
                    f"Step timed out after {self.config.timeout_per_step} seconds"
                )
            except Exception as exc:
                step_success = False
                step_error = str(exc)
            finally:
                # A scheduler call that failed or was cancelled can still carry
                # useful latency/error diagnostics.
                self._record_scheduler_call(scheduler, recorder, step_id)

            if not step_success:
                context.mark_finished(
                    final_output=None,
                    success=False,
                    error_message=step_error,
                )
            recorder.record_step_end(
                step_id=step_id,
                state_after=context.get_state_snapshot(),
                success=step_success,
                error_message=step_error,
            )
            context.cost_summary = recorder.get_cost_summary()

        if not context.finished:
            context.mark_finished(
                final_output=context.get_last_output(),
                success=False,
                error_message=f"{limit_label} exceeded",
            )

        self._record_scheduler_telemetry(scheduler, recorder)
        recorder.end(
            success=context.success,
            final_output=context.final_output,
            error_message=context.error_message,
        )
        return context.final_output, context, recorder

    async def execute_from_checkpoint(
        self,
        checkpoint: ExecutionContext,
        resume_node_id: str,
        workflow: WorkflowTemplate,
        scheduler: BaseScheduler,
        llm_client: Optional[AsyncLLMClient] = None,
        counterfactual: bool = False,
    ) -> tuple[Any, ExecutionContext, TraceRecorder]:
        """Resume execution from a checkpoint after the prefix has run.

        The *checkpoint* carries outputs and variables from the prefix
        (reconstructed from a trace ``state_after`` snapshot).  Only the
        *suffix* nodes starting at ``resume_node_id`` are re-executed.
        """
        declared_workflow = workflow
        workflow, gate_decision = realize_selective_update(
            declared_workflow,
            checkpoint.query,
        )
        # Reconstruct a fresh context from the checkpoint snapshot.
        context = ExecutionContext(
            query=checkpoint.query,
            workflow_name=declared_workflow.name,
        )
        context.outputs = dict(checkpoint.outputs)
        context.variables = dict(checkpoint.variables)
        context.history = list(checkpoint.history)
        # Preserve the prefix artifact stream when resuming.  Without this,
        # schedulers that read ``get_last_output`` see an empty context even
        # though the checkpoint contains valid outputs, which can turn a valid
        # suffix repair into a spurious failure.
        context._output_history = [
            context.outputs[node_id]
            for node_id in context.history
            if node_id in context.outputs
        ]
        context.previous_node_id = checkpoint.previous_node_id
        context.step_count = checkpoint.step_count
        cost = checkpoint.cost_summary
        if isinstance(cost, dict):
            context.cost_summary = dict(cost)
        # Start execution with the resume node.
        context.current_node_id = resume_node_id

        recorder = TraceRecorder(
            query_id=checkpoint.query[:50],
            query_text=checkpoint.query,
            enabled=True,
        )
        recorder.start(
            declared_workflow.name,
            declared_workflow.version,
        )
        # The recorder only receives suffix steps, so seed its aggregate
        # counters with the cached prefix.  This keeps counterfactual utility
        # comparisons fair: replay saves API calls in wall-clock execution,
        # but it must not make the candidate appear cheaper merely because
        # unchanged prefix work is absent from the new trace.
        recorder.seed_cost_summary(cost if isinstance(cost, dict) else {})
        recorder.trace.metadata["execution_mode"] = (
            "counterfactual" if counterfactual else "normal"
        )
        recorder.trace.metadata["suffix_replay"] = True
        if gate_decision is not None:
            recorder.trace.metadata["selective_update"] = (
                gate_decision.model_dump(mode="json")
            )

        try:
            await scheduler.initialize(workflow, checkpoint.query)
            await scheduler.skip_to_node(workflow, resume_node_id)
        except Exception as exc:
            context.mark_finished(
                success=False,
                error_message=f"Checkpoint initialization failed: {exc}",
            )
            recorder.end(
                success=False,
                final_output=None,
                error_message=context.error_message,
            )
            return context.final_output, context, recorder

        action_limit, limit_label = self._get_action_limit(scheduler)

        while not context.finished and context.step_count < action_limit:
            context.step_count += 1
            state_before = context.get_state_snapshot()
            step_id = recorder.record_step_start(
                node_id=context.current_node_id or "unknown",
                node_type=self._get_node_type(workflow, context.current_node_id),
                state_before=state_before,
            )

            step_success = True
            step_error: str | None = None
            try:
                await asyncio.wait_for(
                    self._select_and_apply(
                        workflow,
                        scheduler,
                        context,
                        llm_client,
                        recorder,
                        step_id,
                        counterfactual,
                    ),
                    timeout=self.config.timeout_per_step,
                )
            except asyncio.TimeoutError:
                step_success = False
                step_error = (
                    f"Step timed out after {self.config.timeout_per_step} seconds"
                )
            except Exception as exc:
                step_success = False
                step_error = str(exc)
            finally:
                self._record_scheduler_call(scheduler, recorder, step_id)

            if not step_success:
                context.mark_finished(
                    final_output=None,
                    success=False,
                    error_message=step_error,
                )
            recorder.record_step_end(
                step_id=step_id,
                state_after=context.get_state_snapshot(),
                success=step_success,
                error_message=step_error,
            )
            context.cost_summary = recorder.get_cost_summary()

        if not context.finished:
            context.mark_finished(
                final_output=context.get_last_output(),
                success=False,
                error_message=f"{limit_label} exceeded",
            )

        self._record_scheduler_telemetry(scheduler, recorder)
        recorder.end(
            success=context.success,
            final_output=context.final_output,
            error_message=context.error_message,
        )
        return context.final_output, context, recorder

    async def _select_and_apply(
        self,
        workflow: WorkflowTemplate,
        scheduler: BaseScheduler,
        context: ExecutionContext,
        llm_client: Optional[AsyncLLMClient],
        recorder: TraceRecorder,
        step_id: str,
        counterfactual: bool,
    ) -> None:
        decision_node_id = context.current_node_id
        action_value, params = await scheduler.select_action(workflow, context)
        self._record_scheduler_call(scheduler, recorder, step_id)

        action = SchedulerAction.coerce(action_value)
        params = params if isinstance(params, dict) else {}
        executes_node = action in {
            SchedulerAction.CONTINUE,
            SchedulerAction.EXECUTE,
            SchedulerAction.VERIFY,
            SchedulerAction.REPAIR,
            SchedulerAction.REROUTE,
            SchedulerAction.FALLBACK,
        }
        recorder.update_step(
            step_id,
            action=action.value,
            metadata={
                "scheduler_params": params,
                "node_executed": False,
                "scheduler_action_validated": False,
                "decision_node_id": decision_node_id,
            },
        )
        allow_deviation = bool(
            getattr(
                getattr(scheduler, "config", None),
                "allow_deviation",
                True,
            )
        )
        if (
            not allow_deviation
            and action
            in {
                SchedulerAction.REROUTE,
                SchedulerAction.SKIP,
                SchedulerAction.RETRY,
                SchedulerAction.BRANCH,
                SchedulerAction.DEVIATE,
            }
        ):
            raise ValueError(
                f"Scheduler deviation is disabled; action "
                f"{action.value!r} is not allowed"
            )
        self._prepare_action_target(
            action,
            params,
            workflow,
            context,
            allow_deviation=allow_deviation,
        )

        recorder.update_step(
            step_id,
            action=action.value,
            node_id=context.current_node_id or "unknown",
            node_type=self._get_node_type(workflow, context.current_node_id),
            metadata={
                "scheduler_params": params,
                "node_executed": executes_node,
                "scheduler_action_validated": True,
                "decision_node_id": decision_node_id,
            },
            state_before=context.get_state_snapshot(),
        )
        await self._apply_action(
            action,
            params,
            workflow,
            context,
            llm_client,
            recorder,
            step_id,
            counterfactual,
        )

    def _prepare_action_target(
        self,
        action: SchedulerAction,
        params: dict[str, Any],
        workflow: WorkflowTemplate,
        context: ExecutionContext,
        *,
        allow_deviation: bool,
    ) -> None:
        """Resolve the unit selected by a canonical execution action."""
        execution_actions = {
            SchedulerAction.CONTINUE,
            SchedulerAction.EXECUTE,
            SchedulerAction.VERIFY,
            SchedulerAction.REPAIR,
            SchedulerAction.REROUTE,
            SchedulerAction.FALLBACK,
        }
        if action not in execution_actions:
            return

        # CONTINUE has deliberately non-routing semantics: execute the current
        # unit and then follow the workflow's normal edge.  Treating an
        # optional ``next_unit_id`` as a jump made a superficially harmless
        # response such as ``continue -> end`` skip the current completion
        # node (for example ``finalize``), leaving an intermediate verifier
        # message as the task output.  Explicit routing actions below are the
        # only LLM-scheduler actions allowed to select a different unit.
        if action == SchedulerAction.CONTINUE:
            return

        target = (
            params.get("next_unit_id")
            or params.get("target_node")
            or params.get("node_id")
        )
        if target is None and action in {
            SchedulerAction.VERIFY,
            SchedulerAction.REPAIR,
            SchedulerAction.REROUTE,
            SchedulerAction.FALLBACK,
        }:
            target = self._infer_action_target(action, workflow, context)

        if target is None:
            if action in {
                SchedulerAction.VERIFY,
                SchedulerAction.REPAIR,
                SchedulerAction.REROUTE,
                SchedulerAction.FALLBACK,
            }:
                raise ValueError(
                    f"{action.value} action requires a matching valid "
                    "next_unit_id"
                )
            return
        if target not in workflow.nodes:
            raise ValueError(f"Unknown scheduler target node: {target}")
        if (
            not allow_deviation
            and target != context.current_node_id
        ):
            raise ValueError(
                "Scheduler deviation is disabled; target must be the current "
                f"node ({context.current_node_id!r}), got {target!r}"
            )
        if target != context.current_node_id:
            context.set_current_node(target)

    @staticmethod
    def _infer_action_target(
        action: SchedulerAction,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
    ) -> str | None:
        """Best-effort target inference for concise scheduler responses."""
        keyword = {
            SchedulerAction.VERIFY: "verif",
            SchedulerAction.REPAIR: "repair",
            SchedulerAction.REROUTE: "route",
            SchedulerAction.FALLBACK: "fallback",
        }[action]
        current = context.current_node_id
        candidates = workflow.get_successors(current) if current else []
        candidates.extend(
            node_id for node_id in workflow.nodes if node_id not in candidates
        )
        for node_id in candidates:
            node = workflow.nodes[node_id]
            metadata_role = str(node.config.metadata.get("role", ""))
            haystack = f"{node_id} {node.label} {metadata_role}".lower()
            if keyword in haystack:
                return node_id
        return None

    async def _apply_action(
        self,
        action: SchedulerAction,
        params: dict[str, Any],
        workflow: WorkflowTemplate,
        context: ExecutionContext,
        llm_client: Optional[AsyncLLMClient],
        recorder: TraceRecorder,
        step_id: str,
        counterfactual: bool,
    ) -> None:
        """Apply a scheduler action to the current execution state."""
        if action in {
            SchedulerAction.CONTINUE,
            SchedulerAction.EXECUTE,
            SchedulerAction.VERIFY,
            SchedulerAction.REPAIR,
            SchedulerAction.REROUTE,
            SchedulerAction.FALLBACK,
        }:
            await self._execute_node(
                workflow,
                context,
                llm_client,
                recorder,
                step_id,
                counterfactual,
            )
            self._advance_node(workflow, context)

        elif action == SchedulerAction.SKIP:
            self._advance_node(workflow, context)

        elif action == SchedulerAction.RETRY:
            # Legacy behavior: move back; the next scheduler decision executes.
            if context.previous_node_id:
                context.set_current_node(context.previous_node_id)

        elif action == SchedulerAction.BRANCH:
            target = params.get("target_node") or params.get("next_unit_id")
            if target is not None and target not in workflow.nodes:
                raise ValueError(f"Unknown branch target node: {target}")
            if target:
                context.set_current_node(target)
            else:
                self._advance_node(workflow, context)

        elif action == SchedulerAction.DEVIATE:
            custom_output = params.get("output")
            if context.current_node_id:
                context.record_output(context.current_node_id, custom_output)
            self._advance_node(workflow, context)

        elif action in {SchedulerAction.STOP, SchedulerAction.EARLY_EXIT}:
            context.mark_finished(
                final_output=context.get_last_output(),
                success=True,
            )

    async def _execute_node(
        self,
        workflow: WorkflowTemplate,
        context: ExecutionContext,
        llm_client: Optional[AsyncLLMClient],
        recorder: TraceRecorder,
        step_id: str,
        counterfactual: bool,
    ) -> None:
        """Execute the current workflow node."""
        node_id = context.current_node_id
        if not node_id or node_id not in workflow.nodes:
            raise RuntimeError(f"Cannot execute unknown workflow node: {node_id}")

        node = workflow.nodes[node_id]

        if node.node_type == NodeType.START:
            context.record_output(node_id, {"status": "started"})

        elif node.node_type == NodeType.END:
            context.mark_finished(
                final_output=context.get_last_output(),
                success=True,
            )

        elif node.node_type == NodeType.LLM:
            await self._execute_llm_node(
                node,
                node_id,
                context,
                llm_client,
                recorder,
                step_id,
                counterfactual,
            )

        elif node.node_type == NodeType.TOOL:
            await self._execute_tool_node(
                node,
                node_id,
                context,
                recorder,
                step_id,
                counterfactual,
            )

        elif node.node_type == NodeType.CONDITION:
            self._execute_condition_node(node, node_id, context)

        elif node.node_type == NodeType.JOIN:
            context.record_output(node_id, {"status": "merged"})

        else:
            raise RuntimeError(f"Unsupported node type: {node.node_type}")

    async def _execute_llm_node(
        self,
        node: Any,
        node_id: str,
        context: ExecutionContext,
        llm_client: Optional[AsyncLLMClient],
        recorder: TraceRecorder,
        step_id: str,
        counterfactual: bool,
    ) -> None:
        """Execute a custom LLM operator or an API-backed LLM node."""
        if node_id in self.operators:
            self._require_counterfactual_safe(
                node_id,
                counterfactual,
            )
            output = await self._invoke_callable(
                self.operators[node_id],
                context,
            )
            context.record_output(node_id, output)
            return

        if llm_client is None:
            raise RuntimeError(
                f"LLM node '{node_id}' requires an llm_client or custom operator"
            )

        system_prompt = node.config.system_prompt or ""
        user_template = node.config.prompt_template or "{query}"
        prompt_values = defaultdict(str, context.variables)
        prompt_values["query"] = context.query
        user_prompt = user_template.format_map(prompt_values)
        configured_model = getattr(node.config, "model", None)
        client_config = getattr(llm_client, "config", None)
        client_model = getattr(client_config, "model", "")
        client_seed = getattr(client_config, "seed", None)
        client_provider = getattr(client_config, "provider", None)
        provider_name = getattr(client_provider, "value", client_provider)
        effective_seed = (
            None if provider_name == "deepseek" else client_seed
        )
        client_temperature = getattr(client_config, "temperature", None)
        client_max_tokens = getattr(client_config, "max_tokens", None)
        model_name = configured_model or client_model or ""
        request_metadata = {
            "request": {
                "model": str(model_name),
                "temperature": (
                    node.config.temperature
                    if node.config.temperature is not None
                    else client_temperature
                ),
                "max_tokens": (
                    node.config.max_tokens
                    if node.config.max_tokens is not None
                    else client_max_tokens
                ),
                "seed": effective_seed,
            }
        }

        generate_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": node.config.temperature,
            "max_tokens": node.config.max_tokens,
        }
        # Only require custom clients to accept ``model`` when a node actually
        # requests a per-node override.
        if configured_model is not None:
            generate_kwargs["model"] = configured_model

        call_start = time.perf_counter()
        try:
            response, usage = await llm_client.generate(**generate_kwargs)
        except BaseException as exc:
            recorder.record_llm_call(
                step_id,
                LLMCallRecord(
                    call_id=f"{step_id}_llm",
                    model=str(model_name),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    latency_seconds=time.perf_counter() - call_start,
                    cost_estimate_available=False,
                    success=False,
                    error_message=str(exc),
                    call_type="workflow",
                    metadata=request_metadata,
                ),
            )
            raise

        usage = usage or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        explicit_cost = (
            "cost_usd" in usage and usage.get("cost_usd") is not None
        )
        call_metadata = {
            **request_metadata,
            "usage": dict(usage),
        }
        recorder.record_llm_call(
            step_id,
            LLMCallRecord(
                call_id=f"{step_id}_llm",
                model=str(model_name),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=int(
                    usage.get("total_tokens", prompt_tokens + completion_tokens)
                ),
                latency_seconds=time.perf_counter() - call_start,
                cost_usd=float(
                    usage.get(
                        "cost_usd",
                        estimate_cost_usd(
                            str(model_name),
                            prompt_tokens,
                            completion_tokens,
                            prompt_cache_hit_tokens=usage.get(
                                "prompt_cache_hit_tokens"
                            ),
                            prompt_cache_miss_tokens=usage.get(
                                "prompt_cache_miss_tokens"
                            ),
                        ),
                    )
                ),
                cost_estimate_available=bool(
                    usage.get(
                        "cost_estimate_available",
                        explicit_cost or has_known_pricing(str(model_name)),
                    )
                ),
                success=True,
                call_type="workflow",
                metadata=call_metadata,
            ),
        )
        context.record_output(node_id, response)

    async def _execute_tool_node(
        self,
        node: Any,
        node_id: str,
        context: ExecutionContext,
        recorder: TraceRecorder,
        step_id: str,
        counterfactual: bool,
    ) -> None:
        """Invoke a configured tool by ``tool_name`` (or node-id fallback)."""
        configured_tool_name = node.config.tool_name
        tool_name = configured_tool_name or node_id
        operator = self.operators.get(tool_name)
        legacy_node_lookup = configured_tool_name is None and operator is not None
        if operator is None:
            raise RuntimeError(
                f"Tool node '{node_id}' requires registered tool '{tool_name}'"
            )
        self._require_counterfactual_safe(tool_name, counterfactual)

        resolved_args = self._resolve_tool_args(node.config.tool_args, context)
        call_args, call_kwargs = self._bind_tool_invocation(
            operator,
            resolved_args,
            context,
            prefer_context=legacy_node_lookup,
        )
        call_start = time.perf_counter()
        try:
            output = await self._invoke_callable(
                operator,
                *call_args,
                **call_kwargs,
            )
        except BaseException as exc:
            recorder.record_tool_call(
                step_id,
                ToolCallRecord(
                    tool_call_id=f"{step_id}_tool",
                    tool_name=tool_name,
                    tool_args=to_trace_value(resolved_args),
                    latency_seconds=time.perf_counter() - call_start,
                    success=False,
                    error_message=str(exc),
                ),
            )
            raise

        recorder.record_tool_call(
            step_id,
            ToolCallRecord(
                tool_call_id=f"{step_id}_tool",
                tool_name=tool_name,
                tool_args=to_trace_value(resolved_args),
                tool_result=to_trace_value(output),
                latency_seconds=time.perf_counter() - call_start,
                success=True,
            ),
        )
        context.record_output(node_id, output)

    def _require_counterfactual_safe(
        self,
        operator_name: str,
        counterfactual: bool,
    ) -> None:
        if (
            counterfactual
            and operator_name not in self.counterfactual_safe_tools
        ):
            raise RuntimeError(
                "Counterfactual execution blocked operator "
                f"'{operator_name}': mark a side-effect-free operator with "
                "@counterfactual_safe, add it to counterfactual_safe_tools, "
                "or use a sandbox/transaction adapter"
            )

    @staticmethod
    async def _invoke_callable(
        operator: OperatorFn,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke synchronous and asynchronous operator implementations."""
        if inspect.iscoroutinefunction(operator):
            return await operator(*args, **kwargs)
        result = await asyncio.to_thread(operator, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _bind_tool_invocation(
        operator: OperatorFn,
        tool_args: dict[str, Any],
        context: ExecutionContext,
        *,
        prefer_context: bool,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Choose a compatible calling convention without executing twice."""
        plain = ((), dict(tool_args))
        context_kw = ((), {"context": context, **tool_args})
        context_pos = ((context,), dict(tool_args))
        candidates = (
            [context_pos, context_kw, plain]
            if prefer_context and not tool_args
            else [plain, context_kw, context_pos]
        )
        try:
            signature = inspect.signature(operator)
        except (TypeError, ValueError):
            return plain
        for args, kwargs in candidates:
            try:
                signature.bind(*args, **kwargs)
                return args, kwargs
            except TypeError:
                continue
        raise TypeError(
            f"Configured tool arguments do not match {tool_args!r}"
        )

    @classmethod
    def _resolve_tool_args(
        cls,
        value: Any,
        context: ExecutionContext,
    ) -> Any:
        values = dict(context.variables)
        values["query"] = context.query
        if isinstance(value, dict):
            return {
                key: cls._resolve_tool_args(item, context)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._resolve_tool_args(item, context) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._resolve_tool_args(item, context) for item in value)
        if isinstance(value, str):
            if (
                value.startswith("{")
                and value.endswith("}")
                and value.count("{") == 1
                and value.count("}") == 1
                and value[1:-1] in values
            ):
                return values[value[1:-1]]
            placeholder = re.compile(
                r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})"
            )
            return placeholder.sub(
                lambda match: (
                    str(values[match.group(1)])
                    if match.group(1) in values
                    else match.group(0)
                ),
                value,
            )
        return value

    def _execute_condition_node(
        self, node: Any, node_id: str, context: ExecutionContext
    ) -> None:
        """Evaluate a condition node with the restricted expression engine."""
        expr = node.config.condition_expr
        if expr:
            try:
                result = evaluate_condition_expression(expr, context)
            except ConditionExpressionError as exc:
                raise RuntimeError(
                    f"Invalid condition expression for '{node_id}': {exc}"
                ) from exc
        else:
            result = True
        context.record_output(node_id, {"condition_result": result})

    def _advance_node(
        self, workflow: WorkflowTemplate, context: ExecutionContext
    ) -> None:
        """Move to the template successor after a successful execution."""
        if context.finished or not context.current_node_id:
            return

        successors = workflow.get_successors(context.current_node_id)
        if not successors:
            context.mark_finished(
                final_output=context.get_last_output(),
                success=True,
            )
            return

        node = workflow.nodes.get(context.current_node_id)
        if node and node.node_type == NodeType.CONDITION:
            condition_output = context.get_output(context.current_node_id)
            if isinstance(condition_output, dict):
                branch = condition_output.get("condition_result", True)
                if not branch and len(successors) > 1:
                    context.set_current_node(successors[1])
                else:
                    context.set_current_node(successors[0])
            else:
                context.set_current_node(successors[0])
        else:
            context.set_current_node(successors[0])

    def _get_action_limit(self, scheduler: BaseScheduler) -> tuple[int, str]:
        """Apply both executor and scheduler action budgets."""
        executor_limit = max(int(self.config.max_steps), 0)
        scheduler_config = getattr(scheduler, "config", None)
        scheduler_limit = getattr(
            scheduler_config,
            "max_actions_per_query",
            None,
        )
        if scheduler_limit is None:
            return executor_limit, "Max steps"
        scheduler_limit = max(int(scheduler_limit), 0)
        if scheduler_limit < executor_limit:
            return scheduler_limit, "Max actions"
        return executor_limit, "Max steps"

    def _record_scheduler_call(
        self,
        scheduler: BaseScheduler,
        recorder: TraceRecorder,
        step_id: str,
    ) -> None:
        pop_call = getattr(scheduler, "pop_last_llm_call", None)
        if not callable(pop_call):
            return
        call = pop_call()
        if call is None:
            return
        if not isinstance(call, LLMCallRecord):
            call = LLMCallRecord.model_validate(call)
        call.call_id = f"{step_id}_scheduler"
        call.call_type = "scheduler"
        recorder.record_llm_call(step_id, call)

    @staticmethod
    def _record_scheduler_telemetry(
        scheduler: BaseScheduler,
        recorder: TraceRecorder,
    ) -> None:
        telemetry = getattr(scheduler, "telemetry", None)
        if not callable(telemetry):
            return
        try:
            payload = telemetry()
        except Exception:
            return
        if isinstance(payload, dict) and payload:
            recorder.trace.metadata["scheduler_telemetry"] = to_trace_value(
                payload
            )

    @staticmethod
    def _get_node_type(
        workflow: WorkflowTemplate,
        node_id: Optional[str],
    ) -> str:
        if node_id and node_id in workflow.nodes:
            return workflow.nodes[node_id].node_type.value
        return "unknown"
