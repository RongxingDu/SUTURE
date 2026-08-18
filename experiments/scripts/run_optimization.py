#!/usr/bin/env python3
"""Main optimization loop script.

Usage:
    python experiments/scripts/run_optimization.py --config experiments/configs/code_gen.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from awf.config.loader import load_config
from awf.protocol.experiment import ExperimentRunner
from awf.protocol.manifest import file_sha256
from awf.workflow.serializer import load_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run workflow optimization experiment")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--workflow", "-w",
        help="Path to initial workflow YAML file",
    )
    parser.add_argument(
        "--benchmark",
        default="code_gen",
        choices=[
            "code_gen",
            "math",
            "agent",
            "gpqa",
            "mmlu",
            "scicode",
        ],
        help="Benchmark type",
    )
    parser.add_argument(
        "--data",
        help="Path to dataset file (JSONL)",
    )
    parser.add_argument(
        "--allow-local-code-execution",
        action="store_true",
        help=(
            "Explicitly enable the restricted local subprocess runner for "
            "trusted code fixtures. It is not a production sandbox."
        ),
    )
    parser.add_argument(
        "--use-bubblewrap-code-sandbox",
        action="store_true",
        help=(
            "Execute generated code in the Linux bubblewrap sandbox with "
            "network and project/home filesystem isolation. Required for "
            "SciCode."
        ),
    )
    parser.add_argument(
        "--scicode-hdf5",
        help=(
            "Path to SciCode's official test_data.h5. Required only for "
            "--benchmark scicode and fingerprinted in the run manifest."
        ),
    )
    parser.add_argument(
        "--scicode-protocol",
        choices=["first_subproblem", "independent_subproblems"],
        default="first_subproblem",
        help=(
            "SciCode evaluation protocol. The pilot uses first_subproblem; "
            "independent_subproblems is explicitly non-official."
        ),
    )
    parser.add_argument(
        "--allow-zero-hard-reward",
        action="store_true",
        help=(
            "Explicitly allow a code experiment with execution disabled. "
            "Hard code reward will be unavailable and remain zero."
        ),
    )
    args = parser.parse_args()
    if args.allow_zero_hard_reward and args.benchmark != "code_gen":
        parser.error("--allow-zero-hard-reward applies only to code_gen")
    if (
        args.allow_local_code_execution
        and args.benchmark != "code_gen"
    ):
        parser.error("--allow-local-code-execution applies only to code_gen")
    if (
        args.use_bubblewrap_code_sandbox
        and args.benchmark not in {"code_gen", "scicode"}
    ):
        parser.error(
            "--use-bubblewrap-code-sandbox applies only to code_gen or scicode"
        )
    if args.scicode_hdf5 and args.benchmark != "scicode":
        parser.error("--scicode-hdf5 applies only to scicode")
    if sum(
        (
            args.allow_zero_hard_reward,
            args.allow_local_code_execution,
            args.use_bubblewrap_code_sandbox,
        )
    ) > 1:
        parser.error(
            "--allow-zero-hard-reward, --allow-local-code-execution, and "
            "--use-bubblewrap-code-sandbox are mutually exclusive"
        )
    if (
        args.benchmark == "code_gen"
        and not args.allow_local_code_execution
        and not args.use_bubblewrap_code_sandbox
        and not args.allow_zero_hard_reward
    ):
        parser.error(
            "code_gen needs --use-bubblewrap-code-sandbox, a library-provided "
            "sandbox runner, or the trusted --allow-local-code-execution "
            "option; use "
            "--allow-zero-hard-reward only for process-reward experiments"
        )
    if args.benchmark in {"gpqa", "mmlu", "scicode"} and not args.data:
        parser.error(f"{args.benchmark} requires --data")
    if args.benchmark == "scicode":
        if not args.scicode_hdf5:
            parser.error("scicode requires --scicode-hdf5")
        if not args.use_bubblewrap_code_sandbox:
            parser.error(
                "scicode requires --use-bubblewrap-code-sandbox; generated "
                "scientific code must not run in the host process"
            )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    logger.info(f"Loaded config: {config.name}")

    # Load workflow
    if args.workflow:
        workflow_path = Path(args.workflow)
    else:
        workflow_path = _default_workflow_path(args.benchmark)

    if not workflow_path.exists():
        logger.error(f"Workflow file not found: {workflow_path}")
        sys.exit(1)

    workflow = load_workflow(workflow_path)
    logger.info(f"Loaded workflow: {workflow.name} v{workflow.version}")

    effective_data_path = _effective_data_path(args.benchmark, args.data)

    # Load benchmark-specific components
    reward_evaluator, data, operators = _load_benchmark(
        args.benchmark,
        str(effective_data_path),
        allow_local_code_execution=args.allow_local_code_execution,
        use_bubblewrap_code_sandbox=args.use_bubblewrap_code_sandbox,
        scicode_hdf5_path=args.scicode_hdf5,
        scicode_protocol=args.scicode_protocol,
    )
    logger.info(f"Loaded benchmark: {args.benchmark}")

    # Create experiment runner
    runner = ExperimentRunner(
        config=config,
        workflow=workflow,
        reward_evaluator=reward_evaluator,
        operators=operators,
        run_metadata=_run_metadata(
            args.benchmark,
            args.allow_local_code_execution,
            args.allow_zero_hard_reward,
            use_bubblewrap_code_sandbox=args.use_bubblewrap_code_sandbox,
            scicode_hdf5_path=args.scicode_hdf5,
            scicode_protocol=args.scicode_protocol,
        ),
    )

    # Load and split data
    runner.load_data(data, dataset_source_path=effective_data_path)

    # Run experiment
    results = asyncio.run(runner.run())

    logger.info(f"Experiment complete. Results saved to {config.output_dir}/{config.name}/")
    logger.info(f"Best val score: {results.get('best_val_score')}")
    logger.info(
        "Held-out test was not evaluated. Run awf-test once after reviewing "
        "the validation-selected checkpoint."
    )


def _run_metadata(
    benchmark: str,
    allow_local_code_execution: bool,
    allow_zero_hard_reward: bool = False,
    use_bubblewrap_code_sandbox: bool = False,
    scicode_hdf5_path: str | Path | None = None,
    scicode_protocol: str = "first_subproblem",
) -> dict[str, str | bool | int]:
    """Public evaluation mode recorded in the reproducibility manifest."""
    if benchmark == "code_gen":
        if use_bubblewrap_code_sandbox:
            execution_mode = "bubblewrap"
        elif allow_local_code_execution:
            execution_mode = "restricted_local_subprocess"
        else:
            execution_mode = "disabled"
    elif benchmark == "scicode":
        execution_mode = (
            "scientific_bubblewrap"
            if use_bubblewrap_code_sandbox
            else "disabled"
        )
    else:
        execution_mode = "not_applicable"
    metadata: dict[str, str | bool | int] = {
        "benchmark": benchmark,
        "code_execution_mode": execution_mode,
    }
    if benchmark == "code_gen":
        metadata["zero_hard_reward_opt_in"] = bool(
            allow_zero_hard_reward
        )
    elif benchmark == "scicode":
        if scicode_protocol not in {
            "first_subproblem",
            "independent_subproblems",
        }:
            raise ValueError(
                f"Unsupported SciCode protocol: {scicode_protocol}"
            )
        if scicode_hdf5_path is None:
            raise ValueError(
                "SciCode metadata requires an official HDF5 target path"
            )
        hdf5_path = Path(scicode_hdf5_path)
        if not hdf5_path.is_file():
            raise FileNotFoundError(
                f"SciCode HDF5 target file not found: {hdf5_path}"
            )
        metadata.update(
            {
                "scicode_protocol": scicode_protocol,
                "scicode_hdf5_sha256": file_sha256(hdf5_path),
                "scicode_hdf5_size_bytes": hdf5_path.stat().st_size,
            }
        )
    return metadata


def _effective_data_path(
    benchmark: str,
    supplied_path: str | Path | None,
) -> Path:
    """Resolve the exact source that must be sealed in every new manifest."""
    if supplied_path is not None:
        path = Path(supplied_path).expanduser().resolve()
    else:
        defaults = {
            "code_gen": PROJECT_ROOT / "data" / "humaneval.jsonl",
            "math": PROJECT_ROOT / "data" / "gsm8k.jsonl",
            "agent": PROJECT_ROOT / "data" / "agent_tasks.jsonl",
        }
        path = defaults.get(benchmark)
        if path is None:
            raise FileNotFoundError(
                f"No default dataset exists for {benchmark}; provide --data"
            )
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset source file not found: {path}")
    return path


def _load_benchmark(
    benchmark: str,
    data_path: str | None,
    allow_local_code_execution: bool = False,
    use_bubblewrap_code_sandbox: bool = False,
    scicode_hdf5_path: str | Path | None = None,
    scicode_protocol: str = "first_subproblem",
):
    """Load benchmark-specific components."""
    if benchmark == "code_gen":
        from benchmarks.code_generation.reward import CodeGenReward
        from benchmarks.code_generation.evaluator import CodeEvaluator
        from benchmarks.code_generation.dataset import load_humaneval, load_mbpp
        from experiments.workflows.code_gen.operators import extract_final_code

        reward = CodeGenReward(
            CodeEvaluator(
                allow_local_execution=allow_local_code_execution,
                use_bubblewrap=use_bubblewrap_code_sandbox,
            )
        )
        if not (
            allow_local_code_execution
            or use_bubblewrap_code_sandbox
        ):
            logger.warning(
                "Code execution is disabled. Hard code rewards will remain zero "
                "unless a sandbox runner is configured or the explicit local "
                "test-only flag is used."
            )
        if data_path:
            with open(data_path, "r") as f:
                first = next((line for line in f if line.strip()), "")
            sample = json.loads(first) if first else {}
            data = (
                load_mbpp(data_path)
                if "test_list" in sample
                else load_humaneval(data_path)
            )
        else:
            default_path = PROJECT_ROOT / "data" / "humaneval.jsonl"
            if not default_path.exists():
                raise FileNotFoundError(
                    "No code dataset found; provide --data explicitly"
                )
            data = load_humaneval(default_path)
        operators = {
            "finalize": extract_final_code,
            "extract_final_code": extract_final_code,
        }

    elif benchmark == "math":
        from benchmarks.math_reasoning.reward import MathReward
        from benchmarks.math_reasoning.dataset import load_gsm8k, load_math
        from experiments.workflows.math.operators import extract_final_answer

        reward = MathReward()
        if data_path:
            with open(data_path, "r") as f:
                first = next((line for line in f if line.strip()), "")
            sample = json.loads(first) if first else {}
            if "problem" in sample and "solution" in sample:
                data = load_math(data_path)
            elif "question" in sample and "answer" in sample:
                data = load_gsm8k(data_path)
            else:
                raise ValueError(
                    "Unrecognized math dataset schema: expected GSM8K "
                    "(question/answer) or MATH (problem/solution)"
                )
        else:
            default_path = PROJECT_ROOT / "data" / "gsm8k.jsonl"
            if not default_path.exists():
                raise FileNotFoundError(
                    "No math dataset found; provide --data explicitly"
                )
            data = load_gsm8k(default_path)
        operators = {
            "finalize": extract_final_answer,
            "extract_final_answer": extract_final_answer,
        }

    elif benchmark == "agent":
        from benchmarks.agent_tasks.reward import AgentReward
        from benchmarks.agent_tasks.dataset import load_agent_tasks
        from experiments.workflows.agent.operators import execute_action

        reward = AgentReward()
        if data_path:
            data = load_agent_tasks(data_path)
        else:
            default_path = PROJECT_ROOT / "data" / "agent_tasks.jsonl"
            if not default_path.exists():
                raise FileNotFoundError(
                    "No agent dataset found; provide --data explicitly"
                )
            data = load_agent_tasks(default_path)
        operators = {
            "execute": execute_action,
            "execute_action": execute_action,
        }

    elif benchmark in {"gpqa", "mmlu"}:
        from benchmarks.multiple_choice import (
            MultipleChoiceReward,
            load_gpqa,
            load_mmlu,
        )

        if not data_path:
            raise FileNotFoundError(
                f"No {benchmark.upper()} dataset found; provide --data "
                "explicitly"
            )
        reward = MultipleChoiceReward()
        data = (
            load_gpqa(data_path)
            if benchmark == "gpqa"
            else load_mmlu(data_path)
        )
        # The canonical dataset prompt already declares the strict terminal
        # answer contract, so the minimal research workflow needs no tool.
        operators = {}

    elif benchmark == "scicode":
        from benchmarks.scicode import (
            SciCodeEvaluator,
            SciCodeReward,
            ScientificBubblewrapRunner,
            load_scicode,
        )
        from experiments.workflows.scicode.operators import (
            extract_scicode_code,
        )

        if not data_path:
            raise FileNotFoundError(
                "No SciCode dataset found; provide --data explicitly"
            )
        if scicode_hdf5_path is None:
            raise ValueError(
                "SciCode requires the official --scicode-hdf5 target file"
            )
        if not use_bubblewrap_code_sandbox:
            raise ValueError(
                "SciCode requires the scientific bubblewrap runner"
            )
        data, private_test_specs = load_scicode(
            data_path,
            protocol=scicode_protocol,
        )
        evaluator = SciCodeEvaluator(
            private_test_specs=private_test_specs,
            hdf5_path=scicode_hdf5_path,
            runner=ScientificBubblewrapRunner(),
        )
        # Validate target availability and its optional reader dependency
        # before any paid model call. The hash is also manifest-bound below.
        evaluator.validate_assets()
        reward = SciCodeReward(evaluator)
        operators = {
            "finalize": extract_scicode_code,
            "extract_scicode_code": extract_scicode_code,
        }

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    return reward, data, operators


def _default_workflow_path(benchmark: str) -> Path:
    """Return the canonical initial artifact for one benchmark family."""
    workflow_family = (
        "multiple_choice"
        if benchmark in {"gpqa", "mmlu"}
        else benchmark
    )
    return (
        PROJECT_ROOT
        / "experiments"
        / "workflows"
        / workflow_family
        / "default_workflow.yaml"
    )


if __name__ == "__main__":
    main()
