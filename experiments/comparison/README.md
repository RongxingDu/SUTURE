# Frozen method comparison protocol

> Historical comparison note: the default AWF optimizer now uses
> `workflow_content_only=true`, so new runs do not deploy S-CWU selective
> policies. This comparison protocol remains available only for previously
> frozen legacy artifacts or an explicit `workflow_content_only=false` ablation.

The current three-method builder compares three already-frozen GPQA artifacts:
the unoptimized one-call workflow (`vanilla`), one genuine AFlow graph update
(`aflow`), and the validation-selected Selective Counterfactual Workflow
Update (`scwu`). Search and inference telemetry are reported separately.

MMLU and SciCode are implemented as AWF benchmarks and can use the generic
runner for built-in Vanilla/S-CWU studies. The checked-out AFlow extension has
no MMLU or SciCode evaluator/template/search pipeline, so the formal builder
fails closed outside GPQA instead of presenting an unimplemented AFlow
baseline.

The formal CLI accepts only the audited built-in AWF and AFlow adapter
factories. New research baselines must first receive an explicit built-in
adapter and identity policy; an arbitrary same-process Python factory is not a
valid label-isolation boundary.

## Prepare the specification

Run the offline preparation command after both searches finish and before any
held-out test is accessed:

```bash
python experiments/scripts/prepare_method_comparison.py \
  --benchmark gpqa \
  --awf-results /path/to/awf/results.json \
  --initial-workflow experiments/workflows/multiple_choice/default_workflow.yaml \
  --aflow-graph /path/to/run/frozen_artifacts/workspace/GPQA/workflows/round_2/graph.py \
  --aflow-search-telemetry /path/to/run/search_telemetry_summary.json \
  --config experiments/configs/gpqa_deepseek_pilot.yaml \
  --data /home/rongxing/Benchmark/QA/GPQA/gpqa_pilot20.jsonl \
  --aflow-root /path/to/AFlow \
  --output /new/path/gpqa_comparison.json
```

The command refuses an existing output, inline credentials, a missing search
record, changed dataset/config/workflow/checkpoint hashes, and changed AFlow
artifacts. It requires the S-CWU checkpoint to contain a deployed selective
update. Credentials remain environment-only and no API client is created.

AFlow must be an actual `round_2` produced by one search update. New
schema-v2 runs archive a cache-free snapshot under
`frozen_artifacts/workspace/GPQA/workflows/round_2`; the comparison must point
to that snapshot, not the mutable AFlow checkout. The envelope binds the AWF
ordered-dataset fingerprint, canonical SHA-256 of the manifest
optimization-index list, and the whole frozen `round_2` directory. It also
binds the search-time role models/config hashes, AFlow commit/runtime-tree
hash, and raw logical-call telemetry hash/count. The directory hash is the
comparison-adapter hash: recursively sorted files,
excluding `__pycache__` and `.pyc`, represented as
`{relative_path, sha256}` records and hashed with
`awf.protocol.manifest.json_sha256`. The summary file must live outside that
directory. Both `workflow_optimizer` and `execution_round_2` must have calls.
Copying or renaming `round_1` is not a searched AFlow baseline. Generated
`graph.py`/`prompt.py` also pass a best-effort static policy that rejects
direct filesystem, environment, network, dynamic-code, and dunder
introspection surfaces before import. This is an integrity guard, not a VM.
Historical schema-v1 envelopes remain readable so existing results are not
rewritten, but future comparisons should use the archived schema-v2 snapshot.

For GPQA, the frozen AFlow adapter matches the upstream full-graph retry
policy: at most five attempts with a fixed one-second wait after each failed
non-final attempt. This policy is part of the method identity and row metadata.
The wait contributes to workflow wall latency, not provider-call latency;
provider `max_retries` remains a separate SDK transport policy.

## Validation first, test once

Freeze the roster using only validation:

```bash
python experiments/scripts/run_method_comparison.py \
  --spec /path/to/gpqa_comparison.json \
  --phase selection \
  --output /new/path/gpqa_validation_comparison.json
```

Before the **test command** deserializes held-out labels, it atomically creates the shared
`comparison_test_ledger.json` beside the bound AWF result. The legacy filename
is now the single claim used by both comparison and `awf-test`: a claim by
either command permanently rejects the other, even if a different output path
is supplied. Failures after the claim are still consumed attempts.

New optimization manifests seal the exact raw JSONL SHA-256 and size.
Preparation and selection verify that seal but deserialize only their requested
split rows. A legacy manifest without this source-file seal is rejected for a
new comparison instead of falling back to whole-dataset loading. The formal
loader hashes and extracts rows from one open file descriptor in a single pass,
so a pathname replacement cannot move unverified bytes into the scorer.

Selection holds a shared protocol lock through durable output publication;
either test entry point holds the exclusive lock through claim and completion.
Claims and completions are atomic concurrent state transitions. Long-running
commands also reserve a new output path with `O_EXCL` before making API calls.

Only after all choices are frozen, execute the exact same roster once on test:

```bash
python experiments/scripts/run_method_comparison.py \
  --spec /path/to/gpqa_comparison.json \
  --phase test \
  --validation-result /path/to/gpqa_validation_comparison.json \
  --output /new/path/gpqa_test_comparison.json
```

For an AWF-only SciCode comparison, use `first_subproblem` and pass the same
manifest-bound HDF5 file plus
`--use-bubblewrap-code-sandbox --scicode-hdf5 /path/to/test_data.h5` to both
commands. The runner pins one verified HDF5 inode through the entire phase and
rechecks it before publishing results. `independent_subproblems` is rejected
because raw JSONL row indices do not identify its expanded sample sequence.

The serial reference-paired, counterbalanced ABBA/BAAB
`inference.pair_summary` is the primary comparison. Adapters receive blind
samples (`ground_truth=None`); scoring labels stay inside the runner. Unpaired
method totals are descriptive, especially with three methods, and test outcomes
must never be used to change the roster or select a winner.

The completed GPQA mechanism pilot and its negative efficiency result are
reported in
`experiments/results/deepseek_v4_flash/GPQA_S_CWU_PILOT_REPORT.md`.
