# SUTURE method

**SUTURE** stands for **Su**ffix-replayed **T**race-localized **U**pdates with
**R**untime-adaptive **E**xecution.

Its guiding idea is **repair, then route**:

- use development failures to learn what the workflow should do; and
- after freezing that workflow, learn how much of it each query should execute.

The two stages are separated deliberately. During workflow repair the scheduler
is fixed, so candidate effects can be attributed to workflow changes. During
scheduler calibration the workflow is fixed, so routing effects can be
attributed to the execution policy.

## Problem formulation

Let \(W\) be a workflow graph containing LLM, tool, condition, join, start, and
end nodes. Let \(\pi_\theta\) be a runtime scheduling policy over the valid
actions of (W). The deployed system is

\[
\Pi(x; W, \theta).
\]

SUTURE learns the pair sequentially:

\[
W^* = \operatorname{Repair}(W_0; D_{\mathrm{opt}}),
\qquad
\theta^* = \operatorname{Calibrate}(W^*; D_{\mathrm{val}}).
\]

This is a sequential two-stage or bilevel protocol, not joint gradient
training: the second stage does not update \(W^*\), and scheduler measurements
are not fed back into the workflow candidate search.

## Stage 1: trace-localized workflow repair

### 1. Collect a failure-heavy batch

The incumbent workflow runs on the optimization split. SUTURE retains
current-version hard failures and samples a bounded set of hard-success traces
as regression guards. Held-out examples are rejected by the optimizer context.

### 2. Localize suspicious workflow units

For each failed trace, an LLM judge ranks at most five executed nodes using the
query, bounded evaluator diagnostics, step outputs, and workflow hierarchy. A
node appearing at rank \(r_i(v)\) in failure \(i\) receives weight

\[
s(v)=\sum_i 2^{-(r_i(v)-1)}.
\]

The highest-scoring units become repair anchors. Localization selects where to
look; it does not select the edit scope or propose the repair.

### 3. Generate local candidates from shallow to deep

For each anchor, SUTURE searches the following intervention levels:

1. **prompt** — system and user prompt content;
2. **operator** — model, temperature, token budget, tool, or condition
   parameters;
3. **graph path** — atomic additions/removals of local nodes and edges; and
4. **multi-level** — an optional coherent composition of the preceding edits.

Candidate graphs must preserve entry/terminal nodes, reachability, dependency
and output contracts, existing tool boundaries, and DAG structure when the
incumbent is acyclic. In the conservative protocol, once a shallower level
contains a valid repair, SUTURE finishes that level and does not generate a
more invasive one.

### 4. Evaluate with paired suffix replay

For a candidate edit \(\delta\), SUTURE finds the earliest changed executable
node. If the incumbent trace contains a successful, non-terminal checkpoint
immediately before that node, it restores the exact state and re-executes only
the candidate suffix. The unchanged prefix, including its recorded token and
latency cost, remains part of the candidate outcome.

This gives paired observations

\[
(h_i^0,t_i^0,l_i^0),\qquad
(h_i^\delta,t_i^\delta,l_i^\delta)
\]

for the same query and prefix. A full rerun is recorded as a distinct
evaluation mode and is promotion-ineligible under the strict replay protocol.

### 5. Apply hard gates before scalar ranking

A repair must satisfy the configured failure coverage

\[
\frac{|\{i\in F:h_i^\delta\ge\tau_h\}|}{|F|}\ge\rho
\]

and must not regress any sampled success guard. Optional repeated executions
replace a single stochastic hard label with a minimum-success confirmation
rule.

Only eligible repairs are ranked by

\[
G(\delta)
=\overline{\Delta h}
-\lambda_T\overline{\Delta\text{tokens}}
-\mu D(\delta).
\]

Hard correctness is therefore a constraint, not a soft term that token savings
can offset. The selected patch is materialized as a new workflow version.

## Stage 2: calibrated runtime scheduling

Stage 2 freezes \(W^*\) and calibrates a cascade scheduler. The scheduler sees
only runtime-observable information; it never sees reference answers or reward
labels during deployment.

### 1. Zero-generation artifact gate

After an executable node produces an artifact, a deterministic gate extracts:

- \(s\): specification adherence;
- \(l\): a lightweight local artifact-quality score;
- \(a\): agreement with earlier artifacts; and
- \(r\): exponentially updated reliability of the artifact-producing node.

In LAS-compatible mode the score is

\[
q_\theta(z)=
\frac{w_s s+w_l l+w_a(a-1)+w_h(1-r)}
     {w_s+w_l+w_a+w_h}.
\]

Query length, numeric density, and proof-like terms assign a risk band. A
high-risk query raises the threshold required for direct early exit.

### 2. Three-way cascade policy

The gate chooses among three execution modes:

\[
\pi_\theta(z)=
\begin{cases}
\text{follow the frozen graph}, & q<\tau_{\mathrm{sched}},\\
\text{invoke the LLM scheduler}, &
  \tau_{\mathrm{sched}}\le q<\tau_{\mathrm{exit}},\\
\text{early exit}, & q\ge\tau_{\mathrm{exit}}
  \land\text{terminal}(z).
\end{cases}
\]

The compact LLM scheduler may choose `continue`, `early_exit`, `verify`,
`reroute`, `repair`, or `fallback`. The runtime validates the action against
the workflow graph, dependency state, deviation policy, and step budget.

### 3. Correctness-constrained calibration

Gate weights and thresholds are grid-selected on complete validation traces.
For each candidate, recorded terminal artifacts simulate possible early exits.
Candidates that violate the allowed hard-reward regression are discarded;
accuracy ranks the feasible set first, followed by normalized token and latency
savings.

The selected policy is then executed for real on the validation set. It is
deployed only if the audit satisfies

\[
R_{\mathrm{hard}}(W^*,\theta)
\ge R_{\mathrm{hard}}(W^*,\theta_0)-\delta,
\]

\[
\Delta\text{tokens}\le0,qquad
\Delta\text{latency}\le0,
\]

with at least one strict efficiency improvement. Otherwise SUTURE restores the
passive frozen-workflow scheduler.

## Experimental protocol

The optimization split supplies failures and candidate evidence. Validation
selects workflow checkpoints and scheduler parameters. The manifest binds the
ordered dataset, split indices, scientific configuration, checkpoint, and
execution mode. The official held-out split can be claimed only once and is
never used to revise the workflow or scheduler.

The aggressive local-promotion mode, full-rerun-admissible repair, and legacy
Selective Counterfactual Workflow Update (S-CWU) are explicit ablations rather
than the conservative SUTURE definition.

## Paper-facing description

Recommended title:

> **SUTURE: Trace-Localized Workflow Repair and Calibrated Runtime Scheduling
> for LLM Agents**

Short description:

> SUTURE first repairs an LLM workflow with failure-localized edits validated
> by paired suffix replay, then freezes the repaired workflow and calibrates a
> correctness-constrained cascade scheduler that decides whether to continue,
> route, verify, repair, or stop at runtime.
