"""Tests for trace schema and recorder."""

from awf.trace.recorder import TraceRecorder
from awf.trace.schema import ExecutionTrace, LLMCallRecord, ToolCallRecord, TraceStep


class TestTraceSchema:
    """Test trace Pydantic models."""

    def test_create_llm_call_record(self):
        record = LLMCallRecord(
            call_id="call_1",
            model="gpt-4o",
            system_prompt="You are helpful.",
            user_prompt="Hello",
            response_text="Hi!",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_seconds=0.5,
        )
        assert record.prompt_tokens == 10
        assert record.success is True

    def test_create_tool_call_record(self):
        record = ToolCallRecord(
            tool_call_id="tool_1",
            tool_name="calculator",
            tool_args={"expr": "2+2"},
            tool_result="4",
        )
        assert record.tool_name == "calculator"
        assert record.tool_result == "4"

    def test_create_trace_step(self):
        step = TraceStep(
            step_id="step_0",
            step_index=0,
            node_id="llm_1",
            node_type="llm",
            action="execute",
            state_before={"history": []},
            state_after={"history": ["llm_1"]},
        )
        assert step.step_index == 0
        assert step.node_id == "llm_1"

    def test_execution_trace_properties(self):
        trace = ExecutionTrace(
            trace_id="trace_1",
            query_text="Test query",
            success=True,
            total_prompt_tokens=100,
            total_completion_tokens=50,
        )
        assert trace.total_steps == 0


class TestTraceRecorder:
    """Test TraceRecorder functionality."""

    def test_record_full_execution(self):
        recorder = TraceRecorder(query_id="q1", query_text="What is 2+2?")
        recorder.start("test_workflow", "1.0")

        # Record step 1
        step_id = recorder.record_step_start("llm_1", "llm", "execute")
        llm_call = LLMCallRecord(
            call_id="c1",
            model="gpt-4o",
            response_text="4",
            prompt_tokens=10,
            completion_tokens=3,
        )
        recorder.record_llm_call(step_id, llm_call)
        recorder.record_step_end(step_id, {"output": "4"}, success=True)

        # End trace
        trace = recorder.end(success=True, final_output="4", hard_reward=1.0)

        assert trace.trace_id is not None
        assert trace.total_steps == 1
        assert trace.success is True
        assert trace.hard_reward == 1.0
        assert trace.total_prompt_tokens == 10
        assert trace.total_completion_tokens == 3

    def test_record_multiple_steps(self):
        recorder = TraceRecorder(query_id="q2", query_text="Test")
        recorder.start()

        for i in range(3):
            step_id = recorder.record_step_start(f"node_{i}", "llm", "execute")
            recorder.record_step_end(step_id, {f"result_{i}": i})

        trace = recorder.end(success=True)
        assert trace.total_steps == 3
