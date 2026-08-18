"""Execute every bundled workflow through its fixed full-workflow baseline."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from awf.config.schema import ExecutorConfig
from awf.executor.context import ExecutionContext
from awf.executor.runtime import RuntimeExecutor
from awf.scheduler.base import BaseScheduler, SchedulerAction
from awf.scheduler.fixed_scheduler import FixedScheduler
from awf.workflow.serializer import load_workflow
from benchmarks.code_generation.evaluator import CodeEvaluator
from benchmarks.code_generation.reward import CodeGenReward
from experiments.workflows.agent.operators import execute_action
from experiments.workflows.code_gen.operators import extract_final_code
from experiments.workflows.math.operators import extract_final_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeWorkflowLLM:
    config = SimpleNamespace(model="fake")

    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, system_prompt="", user_prompt="", **kwargs):
        self.prompts.append(user_prompt)
        system = system_prompt.lower()
        if "write clean" in system:
            output = "def answer():\n    return 42"
        elif "code reviewer" in system:
            output = "PASS"
        elif "solve problems" in system:
            output = r"Step 1: compute. \boxed{42}"
        elif "math reviewer" in system:
            output = "VERIFIED: 42"
        elif "provide a clear final" in system:
            output = "Final grounded answer"
        else:
            output = "intermediate result"
        return output, {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
        }


class ContinueWithSuccessorScheduler(BaseScheduler):
    """Reproduce an LLM returning a successor with ``continue``."""

    def __init__(self, targeted_node):
        self.targeted_node = targeted_node

    async def initialize(self, workflow, query):
        return None

    async def select_action(self, workflow, context):
        if context.current_node_id == self.targeted_node:
            return SchedulerAction.CONTINUE, {"next_unit_id": "end"}
        return SchedulerAction.CONTINUE, {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "operators", "expected_output", "expected_steps"),
    [
        (
            "code_gen",
            {
                "finalize": extract_final_code,
                "extract_final_code": extract_final_code,
            },
            "def answer():\n    return 42",
            6,
        ),
        (
            "math",
            {
                "finalize": extract_final_answer,
                "extract_final_answer": extract_final_answer,
            },
            "VERIFIED: 42",
            6,
        ),
        (
            "agent",
            {
                "execute": execute_action,
                "execute_action": execute_action,
            },
            "Final grounded answer",
            7,
        ),
    ],
)
async def test_bundled_workflow_reaches_end(
    name,
    operators,
    expected_output,
    expected_steps,
):
    workflow = load_workflow(
        PROJECT_ROOT
        / "experiments"
        / "workflows"
        / name
        / "default_workflow.yaml"
    )
    llm = FakeWorkflowLLM()
    output, context, recorder = await RuntimeExecutor(
        ExecutorConfig(),
        operators,
    ).execute(workflow, FixedScheduler(), "demo query", llm)

    assert context.success is True
    assert context.error_message is None
    assert output == expected_output
    assert recorder.trace.steps[-1].node_type == "end"
    assert recorder.trace.total_steps == expected_steps
    assert any("intermediate result" in prompt for prompt in llm.prompts[1:])


def test_math_finalizer_recovers_unparseable_positive_verifier_payload():
    context = ExecutionContext("demo query", "math_reasoning_v1")
    context.record_output("solve", r"Work gives \boxed{42}.")
    context.record_output("verify", "VERIFIED: the answer is represented below")

    assert extract_final_answer(context) == r"Work gives \boxed{42}."


@pytest.mark.asyncio
async def test_continue_target_cannot_skip_required_completion_node():
    workflow = load_workflow(
        PROJECT_ROOT
        / "experiments"
        / "workflows"
        / "code_gen"
        / "default_workflow.yaml"
    )
    output, _, recorder = await RuntimeExecutor(
        ExecutorConfig(),
        {
            "finalize": extract_final_code,
            "extract_final_code": extract_final_code,
        },
    ).execute(
        workflow,
        ContinueWithSuccessorScheduler("finalize"),
        "write answer",
        FakeWorkflowLLM(),
    )
    trace = recorder.trace
    ground_truth = {
        "test": "def check(candidate): assert candidate() == 42",
        "entry_point": "answer",
    }
    reward = CodeGenReward(
        CodeEvaluator(timeout_seconds=2, allow_local_execution=True)
    )

    assert output == "def answer():\n    return 42"
    assert any(
        step.node_id == "finalize"
        and step.metadata.get("node_executed", True)
        for step in trace.steps
    )
    assert reward.hard_reward("write answer", ground_truth, output, trace) == 1.0
    assert reward.process_reward(
        "write answer",
        ground_truth,
        output,
        trace,
    ) == 1.0
