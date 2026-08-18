# SUTURE

**SUTURE** — **Su**ffix-replayed **T**race-localized **U**pdates with
**R**untime-adaptive **E**xecution — implements optimization and runtime
scheduling for API-based LLM workflows.

The code follows a two-stage **repair, then route** design:

1. repair workflow prompts, operators, and local graph paths from failed
   execution traces; then
2. freeze the workflow and calibrate a cascade scheduler that decides whether
   to follow the graph, ask an LLM scheduler, or stop early.

The Python package remains importable as `awf` for compatibility. The preferred
command names are `suture-*`; the original `awf-*` commands remain aliases.

See [docs/SUTURE_METHOD.md](docs/SUTURE_METHOD.md) for the implementation-level
algorithm description.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Set the API key required by the selected OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY=...
```

YAML configuration files support recursive inheritance through `extends:` and
environment-variable interpolation.

## Run

Optimize a workflow:

```bash
suture-optimize \
  --config experiments/configs/math.yaml \
  --benchmark math \
  --data /path/to/dataset.jsonl
```

Run the manifest-bound evaluation command for a saved workflow checkpoint:

```bash
suture-test \
  --results /path/to/run/results.json \
  --config experiments/configs/math.yaml \
  --benchmark math \
  --data /path/to/dataset.jsonl
```

The combined shell entrypoint runs both commands with one configuration:

```bash
bash experiments/scripts/run_optimize_and_test.sh \
  --config experiments/configs/math.yaml \
  --benchmark math \
  --data /path/to/dataset.jsonl
```

See [experiments/README.md](experiments/README.md) for CLI arguments and the
configuration reference.

## How the code works

### 1. Trace-localized workflow repair

The executor records node inputs, outputs, costs, state checkpoints, and
rewards in an `ExecutionTrace`. The optimizer groups current-workflow failures
with a bounded sample of successful traces used as guards.

An LLM localizer ranks suspicious executed nodes. Candidate generation searches
four edit scopes in increasing order of invasiveness:

- prompt content;
- operator parameters;
- local graph paths (`block` in the internal schema); and
- optional multi-level edits.

Candidates must preserve graph reachability, dependencies, output contracts,
tool boundaries, and acyclicity where applicable.

### 2. Paired suffix replay

For each candidate, the evaluator finds the earliest changed node. When a
compatible checkpoint exists immediately before it, the executor restores the
unchanged state and reruns only the modified suffix. A candidate must repair
the configured fraction of failures without breaking the successful guard
traces before scalar ranking is applied.

The ranking function combines hard-reward change, recorded token-cost change,
and edit distance:

```text
G = delta_hard_reward
    - lambda_cost * delta_tokens
    - mu_edit * edit_distance
```

### 3. Runtime-adaptive execution

After workflow repair, the workflow is frozen. The cascade scheduler scores
the current artifact using specification adherence, lightweight artifact
quality, agreement with earlier artifacts, and historical node reliability.

```text
low score                 -> follow the frozen workflow graph
borderline score          -> invoke the LLM scheduler
high-confidence terminal -> early exit
```

The runtime validates scheduler actions against dependencies, the configured
deviation policy, and the step budget. Supported actions are `continue`,
`early_exit`, `verify`, `reroute`, `repair`, and `fallback`.

Enable the combined path with:

```yaml
scheduler:
  scheduler_type: cascade
  calibration_enabled: true

scheduler_calibration_round: 3
```

See `experiments/configs/suture_math40.yaml` for a complete configuration.

## Code layout

```text
awf/workflow/       workflow graph schema, validation, and serialization
awf/executor/       graph execution, checkpoints, and trace recording
awf/optimizer/      failure localization, candidate edits, and suffix replay
awf/scheduler/      fixed, graph, and calibrated cascade schedulers
awf/protocol/       split manifests and checkpoint binding
awf/reward/         hard and process reward interfaces
benchmarks/         benchmark adapters and evaluators
experiments/        configs and command-line entrypoints
tests/              unit and integration tests
```

## Execution safety

LLM-generated code is not executed by default. Production use must provide a
`CodeRunner` backed by an isolated container or VM. The
`--allow-local-code-execution` option is intended only for trusted fixtures and
does not isolate the filesystem or network.

Counterfactual replay permits only tools explicitly marked or allowlisted as
safe. Side-effecting tools require a sandbox or transaction adapter that can
isolate and roll back their effects.

## Test

```bash
.venv/bin/python -m pytest -q
```
