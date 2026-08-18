# Running experiments

This guide explains how to run a full AWF experiment: workflow optimization on
the development split, then the one-shot official evaluation of the
manifest-bound held-out test split. It documents the one-click pipeline script,
the underlying Python entrypoints, and the key configuration parameters.

## Prerequisites

Install the project and dev dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The temporary research configs currently contain the supplied Aliyun and
DeepSeek keys directly. For a deployment or a different key, replace the
`api_key` values or use the script override. Aliyun DashScope configs
(`aliyun_qwen3.yaml`, `aliyun_qwen35.yaml`) can still be run with:

```bash
export ALIYUN_API_KEY=sk-...
```

OpenAI-style configs (e.g. `default.yaml`) expect `OPENAI_API_KEY` instead.
The public result snapshot redacts API-key fields.

## One-click pipeline (recommended)

`experiments/scripts/run_optimize_and_test.sh` runs the full optimization, then
the official held-out test against the exact same config, ordered dataset, and
benchmark. It derives the results directory from the config, logs every step to
`pipeline.log`, and prints a summary of the best validation round and the
held-out test metrics.

```bash
bash experiments/scripts/run_optimize_and_test.sh \
    --config experiments/configs/math_aliyun_qwen35_full.yaml \
    --benchmark math \
    --data /home/rongxing/Benchmark/AFlow/math_level5_four_domains_aflow_full.jsonl \
    --api-key "$ALIYUN_API_KEY"
```

### Pipeline options

| Option | Description |
| ------ | ----------- |
| `--config PATH` | Experiment YAML config (required). |
| `--benchmark NAME` | `code_gen` \| `math` \| `agent` \| `gpqa` \| `mmlu` \| `scicode` (required). |
| `--data PATH` | Full ordered benchmark JSONL. If omitted, each sub-script falls back to its own default. |
| `--api-key KEY` | API key exported as `ALIYUN_API_KEY`. Falls back to the existing environment variable. |
| `--workflow PATH` | Initial workflow YAML to optimize from (optional; uses the config default otherwise). |
| `--skip-optimize` | Skip optimization; only run the held-out test on an existing `results.json`. |
| `--skip-test` | Run optimization only; do not run the held-out test. |
| `--dry-run` | Print the exact commands that would run and exit, without making API calls. |
| `--allow-local-code-execution` | Restricted local subprocess runner for trusted code fixtures (code_gen only). |
| `--use-bubblewrap-code-sandbox` | Linux bubblewrap sandbox with filesystem/network isolation (code_gen or scicode; required for scicode). |
| `--allow-zero-hard-reward` | Acknowledge hard code reward will be unavailable (code_gen only). |
| `--scicode-hdf5 PATH` | SciCode official `test_data.h5` (required for scicode). |
| `--scicode-protocol NAME` | `first_subproblem` (default) \| `independent_subproblems` (non-official). |

The pipeline is resume-safe: if `results.json` already contains test metrics
it skips optimization, and `run_test` refuses to evaluate an already-recorded
official test a second time.

Before spending API budget, always preview the plan:

```bash
bash experiments/scripts/run_optimize_and_test.sh \
    --config experiments/configs/math_aliyun_qwen35_full.yaml \
    --benchmark math \
    --data /path/to/benchmark.jsonl \
    --api-key "$ALIYUN_API_KEY" \
    --dry-run
```

## Manual two-step run

Running the steps separately gives more control (e.g. debugging optimization
before touching the held-out test).

### Step 1 — Optimization

```bash
python experiments/scripts/run_optimization.py \
    --config experiments/configs/math_aliyun_qwen35_full.yaml \
    --benchmark math \
    --data /path/to/benchmark.jsonl
```

The optimizer creates reproducible optimization/validation/test splits,
validates the initial workflow as round 0, applies validation-based early
stopping, and writes a manifest binding the ordered dataset, exact split
indices, scientific config, and validation-selected checkpoint. It does **not**
evaluate the held-out test split.

`run_optimization.py` arguments:

| Option | Description |
| ------ | ----------- |
| `--config`, `-c PATH` | Experiment YAML config (required). |
| `--workflow`, `-w PATH` | Initial workflow YAML (optional). |
| `--benchmark NAME` | Benchmark type (default `code_gen`). |
| `--data PATH` | Dataset JSONL path. |
| `--allow-local-code-execution` | Enable restricted local code execution (code_gen only). |
| `--use-bubblewrap-code-sandbox` | Use the bubblewrap sandbox (code_gen/scicode; scicode requires it). |
| `--scicode-hdf5 PATH` | SciCode `test_data.h5` (scicode only). |
| `--scicode-protocol NAME` | SciCode protocol (default `first_subproblem`). |
| `--allow-zero-hard-reward` | Allow zero hard reward with execution disabled (code_gen only). |

### Step 2 — Official held-out test

```bash
python experiments/scripts/run_test.py \
    --results experiments/results/aliyun_qwen35/math_aliyun_qwen35_full/results.json \
    --config experiments/configs/math_aliyun_qwen35_full.yaml \
    --benchmark math \
    --data /path/to/benchmark.jsonl
```

`run_test.py` verifies the manifest (config snapshot, split seed/ratios, dataset
row count, checkpoint hashes, execution mode) and the exact test indices, then
evaluates the checkpoint once and atomically records `test_metrics` and
`test_evaluated_at` in `results.json`. It raises on manifest mismatches or a
second official evaluation.

`run_test.py` arguments:

| Option | Description |
| ------ | ----------- |
| `--results`, `-r PATH` | Manifest-bearing `results.json` from optimization (required). |
| `--config`, `-c PATH` | The exact config YAML used for optimization (required). |
| `--benchmark NAME` | The exact benchmark used for optimization (required). |
| `--data PATH` | The exact ordered benchmark JSONL used for optimization (required). |
| `--allow-local-code-execution` | Match an optimization manifest that used local code execution. |
| `--use-bubblewrap-code-sandbox` | Match a manifest that used the bubblewrap sandbox (required for scicode). |
| `--scicode-hdf5 PATH` | The exact SciCode `test_data.h5` used during optimization (scicode only). |
| `--scicode-protocol NAME` | The exact SciCode protocol used during optimization. |
| `--allow-zero-hard-reward` | Match a manifest that disabled code execution (code_gen only). |

The execution-mode flags (`--allow-local-code-execution`,
`--use-bubblewrap-code-sandbox`, `--allow-zero-hard-reward`) are mutually
exclusive and must match what the optimization manifest recorded.

## Configuration parameters

Configs are YAML and inherit recursively via `extends:` (e.g.
`math_aliyun_qwen35_full.yaml` extends `aliyun_qwen35.yaml` extends
`default.yaml`). API keys use `${ENV_VAR}` placeholders resolved at load time.

### Top-level experiment control

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `name` | `default` | Experiment name; forms the results sub-directory `output_dir/name`. |
| `output_dir` | `experiments/results` | Parent directory for this experiment's results. |
| `seed` | `42` | Seed for reproducible splits and shuffling. |
| `early_stopping_patience` | `3` | Rounds without a promoted best before stopping early. |
| `early_stopping_min_delta` | `0.0` | Minimum improvement to reset the early-stopping counter. |
| `validation_min_delta` | `0.0` | Minimum gain required to accept a candidate (see promotion gate). |
| `validation_hard_regression_tolerance` | `0.0` | Allowed hard-reward regression on acceptance. |
| `hard_success_priority` | `false` | Promote on accuracy first: hard reward must improve by `validation_min_delta`, while utility only needs to non-regress within tolerance. |
| `confirm_on_opt` | `false` | Re-validate candidates on the optimization set before promotion. |
| `confirm_repeats` | `1` | Confirmation repeats when `confirm_on_opt` is true. |

### Data splitting

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `opt_split_ratio` | `0.6` | Fraction of rows for optimization. |
| `val_split_ratio` | `0.2` | Fraction of rows for validation. |
| `test_split_ratio` | `0.2` | Fraction of rows held out for the official test. |
| `split_source_field` | *(empty)* | Ground-truth metadata field that carries the split value. When set, rows are partitioned by field value instead of by ratio. |
| `split_dev_value` | `validate` | Field value denoting development (opt/val) rows. |
| `split_test_value` | `test` | Field value denoting held-out test rows. |
| `split_validate_reuse` | `false` | AFlow-style single split: the full development pool is returned for **both** optimization and validation (`opt == val`); only test is held out. Requires `split_source_field`. |

With `split_validate_reuse: true` the ratios only validate the split contract
(must sum to 1.0); opt and val both receive the full validate pool. This is
what the full MATH config uses: AFlow `validate` rows (119) are reused for
optimization and validation, AFlow `test` rows (486) are held out.

### Promotion gate (validation)

Acceptance of a candidate requires passing the validation gate. The gate roles
depend on `hard_success_priority`:

`runtime_utility` is the mean hard reward. Token and edit terms affect the
inner candidate gain, not this validation metric.

- `hard_success_priority: false` (default): utility must improve by
  `validation_min_delta`; hard reward may non-regress within
  `validation_hard_regression_tolerance`.
- `hard_success_priority: true`: hard reward must improve by
  `validation_min_delta`; runtime utility may non-regress within
  `validation_hard_regression_tolerance`.

### `reward`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `alpha_process` | `0.8` | Weight of the process reward in the composite: `R = R_hard + alpha_process * R_process`. |

### `scheduler`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `scheduler_type` | `fixed` | Scheduler family (`fixed`, `graph`, or `cascade`). `cascade` is the dual-layer outer policy: a zero-call gate handles obvious continue/early-exit cases and DeepSeek handles only borderline routing. |
| `max_actions_per_query` | `50` | Maximum scheduling actions per query. |
| `allow_deviation` | `false` | Whether the scheduler may deviate from the planned route. |
| `system_prompt` | *(empty)* | Scheduler system prompt. |
| `gate_*` | paper-inspired defaults | Gate weights and thresholds for the deterministic cascade (`spec`, `lite`, local agreement, historical reliability). |
| `scheduler_max_tokens` | `384` | Compact JSON decision budget for the DeepSeek outer scheduler. |
| `llm` | — | Provider/model block: `provider`, `model`, `api_key`, `api_base`, `temperature`, `max_tokens`, `max_retries`, `timeout_seconds`, `extra_kwargs`. Aliyun configs set `extra_kwargs.extra_body.enable_thinking: false` to keep reasoning tokens off every call. |

`workflow_llm` is an optional top-level backend for workflow-node execution.
When omitted, workflow nodes reuse `scheduler.llm` for backwards compatibility;
the dual Qwen/DeepSeek config sets it explicitly.

### `optimizer`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `workflow_content_only` | `true` | Active research mode. Limits updates to prompt, operator, and graph-path edits (the internal graph-path scope is named `block`), disables selective execution gates/high-cost-success triggers, and keeps suffix replay as the failure-repair correctness gate. Set `false` only for a legacy outer-layer ablation. |
| `max_rounds` | `10` | Number of optimization rounds. |
| `failure_buffer_capacity` | `100` | Max failures retained for candidate generation. |
| `success_guard_fraction` | `0.2` | Fraction of successful rows required as guards. |
| `min_success_guards` | `0` | Minimum number of success guards. |
| `hard_success_threshold` | `1.0` | Hard-reward level treated as full success. |
| `efficiency_optimization_enabled` | `false` | Legacy high-cost-success trigger; ignored in workflow-content-only mode. |
| `selective_update_enabled` | `false` | Legacy query-gated outer update; ignored in workflow-content-only mode. |
| `candidates_per_round` | `5` | Candidate edits proposed per round. |
| `max_edit_distance` | `0.5` | Maximum allowed edit distance for a candidate. |
| `lambda_cost` | `0.0001` | Token-delta coefficient in candidate gain: `G = delta_hard - lambda_cost * delta_tokens - mu_edit * D_edit`. |
| `lambda_latency` | `0.0` | Optional latency coefficient for the separate high-cost-success/efficiency-anchor signal; not part of utility. |
| `lambda_api_cost` | `0.0` | Optional known API-cost coefficient for the separate efficiency signal; not part of utility. |
| `rho_omega` | `0.001` | Optional runtime-complexity coefficient for the separate efficiency signal; not part of utility. |
| `mu_edit` | `0.02` | Small edit-distance coefficient in candidate gain; after the suffix-replay correctness gate, lighter valid repairs are preferred. |
| `epsilon_stat` | `0.01` | Statistical margin in the acceptance rule. |
| `use_suffix_replay` | `false` | Enables suffix-replay evaluation. Inner-only failure repairs instantiate it even when this legacy flag is unset, and require it when `require_failure_suffix_replay=true`. |
| `require_failure_suffix_replay` | `null` | If unset, defaults to `true` for inner-only mode. A failure update is admissible only when every triggered failure succeeds via suffix replay. |
| `max_suffix_replay_cache` | `1000` | Max cached suffix-replay contexts. |
| `llm` | — | Provider/model block for the optimizer (same shape as scheduler `llm`). |

### `executor`

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `max_steps` | `100` | Maximum execution steps per workflow run. |
| `timeout_per_step` | `300.0` | Timeout (seconds) per step. |
| `trace_enabled` | `true` | Record execution traces. |

## Results layout

Optimization writes under `output_dir/name/`:

```
results.json               # manifest + per-round summary + (after test) test_metrics
pipeline.log               # one-click pipeline log
checkpoints/best_workflow.yaml
checkpoints/checkpoint_meta.json
traces/                    # per-split execution traces
official_test/             # held-out test run artifacts
final_test_metrics.json    # copy of the recorded test metrics
```

The one-click script prints a summary from `results.json`: best validation
score, per-round acceptance, early-stop status, and the held-out test metrics
(`hard_success_rate`, `hard_reward`, `composite_reward`, token counts, and
LLM call count).
