# Branch A Gate MVP

Frozen GPT-2 hidden states as predictive-state proxies, with a learned
controller that configures a minimal gate chain to decide when and how to
reuse predictive evidence from similar past hidden states.

## Gate chain

```
MATCH → SELECT → PREDICT → MIX
```

For each query hidden state `h_t` (final GPT-2 layer at position `t`):

| Stage   | Operation |
|---------|-----------|
| MATCH   | Cosine similarity of `h_t` against a datastore of training hidden states |
| SELECT  | Take the top-`k` nearest neighbors |
| PREDICT | `weights = softmax(sims / tau)` → `P_local(y) = Σ weights_i [y_i == y]` |
| MIX     | `P_final(y) = α · P_GPT(y) + (1-α) · P_local(y)` |

The controller learns to choose `(k, tau, alpha)` per query.  If `k=0`, only
GPT is used.

**Success condition:** learned controller val NLL < best fixed-action val NLL,
or same NLL at lower average `k`.

## Project structure

```
branch_a_gate_mvp/
  configs/default.yaml        — all hyperparameters
  src/
    utils.py                  — config, device, seeding, action grid
    gate_chain.py             — MATCH→SELECT→PREDICT→MIX logic
    extract_states.py         — Step 1: extract GPT-2 hidden states
    build_neighbors.py        — Step 2: top-64 cosine neighbor index
    baselines.py              — Step 3: fixed (k,tau,alpha) grid search
    train_controller.py       — Step 4: MLP controller + entropy heuristic
    evaluate.py               — Step 5: final comparison table
    inspect_results.py        — Step 6: human-readable cases
  outputs/
    states/                   — datastore.pt, controller_train.pt, val.pt
    neighbors/                — *_top64.pt
    reports/                  — CSVs + JSONs
    inspection/               — txt files
    models/                   — controller.pt
```

## Installation

```bash
cd branch_a_gate_mvp
pip install -r requirements.txt
```

GPU is recommended (RTX 3060+ or better).  All scripts fall back to CPU if
CUDA is unavailable.

## Running

Run the steps in order.  Each step saves its outputs to `outputs/`.

### Pipeline v1 (original)

```bash
# Step 1 — extract hidden states (downloads TinyStories + GPT-2 once; ~10 min GPU)
python src/extract_states.py --config configs/default.yaml

# Step 2 — build top-64 cosine neighbor index (~2-5 min GPU)
python src/build_neighbors.py --config configs/default.yaml

# Step 3 — evaluate all fixed (k, tau, alpha) actions
python src/baselines.py --config configs/default.yaml

# Step 4 — train classifier MLP controller + entropy heuristic
python src/train_controller.py --config configs/default.yaml

# Step 5 — comparison table (v1)
python src/evaluate.py --config configs/default.yaml

# Step 6 — inspection cases (v1)
python src/inspect_results.py --config configs/default.yaml
```

### Pipeline v2 (Q-controllers + extended heuristics)

Steps 1–3 are the same.  Run these after completing pipeline v1.

```bash
# Step 7 — Q-MLP controller (full + small action sets A/B/C)
#          Also computes and caches Q base data for steps 8-9.
python src/train_q_controller.py --config configs/default.yaml

# Step 8 — Scikit-learn Q-controllers (DecisionTree / RandomForest / GradientBoosting)
python src/train_sklearn_controllers.py --config configs/default.yaml

# Step 9 — Extended threshold heuristics + cost-lambda sweep
python src/run_heuristics.py --config configs/default.yaml

# Step 10 — Unified comparison table (all methods)
python src/evaluate_v2.py --config configs/default.yaml

# Step 11 — Per-method inspection reports
python src/inspect_results_v2.py --config configs/default.yaml
```

Force-recompute Q base data cache (if you change lambda_cost or action grid):

```bash
python src/train_q_controller.py --config configs/default.yaml --force-recompute
```

### Pipeline v3 — Clean story-split audit

Runs on `configs/clean_storysplit.yaml`, saves everything under `outputs_clean_storysplit/`.
No story appears in more than one of {datastore, controller_train, val}.

**One-command option (runs all 12 steps sequentially):**

```bash
python src/run_clean_audit_pipeline.py --config configs/clean_storysplit.yaml
```

**Step-by-step:**

```bash
# Step 1 — story-level state extraction (~15 min GPU)
python src/extract_states.py --config configs/clean_storysplit.yaml

# Step 2 — neighbors + story_id metadata
python src/build_neighbors.py --config configs/clean_storysplit.yaml

# Step 3 — split integrity audit (story overlap, snippet overlap, neighbor sources)
python src/audit_splits.py --config configs/clean_storysplit.yaml

# Step 4 — fixed kNN baselines
python src/baselines.py --config configs/clean_storysplit.yaml

# Step 5 — old MLP classifier controller
python src/train_controller.py --config configs/clean_storysplit.yaml

# Step 6 — Q-MLP controllers (full + A/B/C)
python src/train_q_controller.py --config configs/clean_storysplit.yaml

# Step 7 — sklearn Q-controllers
python src/train_sklearn_controllers.py --config configs/clean_storysplit.yaml

# Step 8 — extended heuristics + cost sweep (NaN bug fixed)
python src/run_heuristics.py --config configs/clean_storysplit.yaml

# Step 9 — similarity distribution diagnostics
python src/similarity_diagnostics.py --config configs/clean_storysplit.yaml

# Step 10 — unified evaluation table v2
python src/evaluate_v2.py --config configs/clean_storysplit.yaml

# Step 11 — per-method inspection files
python src/inspect_results_v2.py --config configs/clean_storysplit.yaml

# Step 12 — old vs clean comparison + AUDIT_REPORT.md
python src/compare_split_results.py --config configs/clean_storysplit.yaml
```

Resume from a specific step after a failure:

```bash
python src/run_clean_audit_pipeline.py --config configs/clean_storysplit.yaml --start-step 6
```

## Configuration

Edit `configs/default.yaml` to change dataset size, model, or hyperparameters.

Key settings:

| Key | Default | Meaning |
|-----|---------|---------|
| `datastore_positions` | 50 000 | Retrieval memory size |
| `controller_train_positions` | 10 000 | Training queries for controller |
| `val_positions` | 10 000 | Evaluation queries (never used for selection) |
| `max_k` | 64 | Maximum neighbors retrieved |
| `k_values` | [4,8,16,32,64] | Action grid k values |
| `tau_values` | [0.05,0.1,0.2] | Temperature values |
| `alpha_values` | [0.25,0.5,0.75] | Mix weights |
| `lambda_cost` | 0.0 | Retrieval cost penalty (0 = pure NLL) |
| `max_q_samples` | 300 000 | Cap on Q training samples (N×A = 460k; subsample if over) |
| `q_controller_hidden` | 128 | Q-MLP hidden dimension |
| `cost_lambda_values` | [0.0, 0.001, …] | Cost-efficiency sweep values |

## Expected outputs

After pipeline v1, `outputs/reports/` contains:

```
fixed_baselines.csv          — every fixed action on train + val
fixed_baselines.json         — summary of best fixed actions
controller_results.json      — old MLP controller + heuristic val metrics
action_distribution.csv      — old controller action counts
final_metrics.csv            — v1 comparison table
```

After pipeline v2, `outputs/reports/` additionally contains:

```
q_controller_metrics.json    — Q-MLP variants (full, A, B, C)
q_dist_Q-MLP-*.csv           — action distributions per Q-MLP variant
sklearn_controller_metrics.json — sklearn Q-controllers
heuristic_baselines.csv      — all (feature, direction, action) combos
best_heuristics.json         — best heuristic + top-10
cost_sweep.csv               — fixed-action NLL vs lambda tradeoff
final_metrics_v2.csv         — unified comparison table (all methods)
final_metrics_v2.json        — same as JSON
```

`outputs/inspection_v2/<method>/` contains per-method inspection files:

```
retrieval_used.txt           — examples where controller used retrieval
gpt_only.txt                 — examples where controller skipped retrieval
biggest_help_vs_gpt.txt      — retrieval helped most vs GPT
biggest_hurt_vs_gpt.txt      — retrieval hurt most vs GPT
wins_vs_best_fixed.txt       — controller beat best fixed action
fails_vs_best_fixed.txt      — controller was worse than best fixed
high_confidence_wrong.txt    — high GPT confidence but wrong; retrieval targets
```

## Design constraints (MVP)

- No regions, modules, or semantic abstractions
- No WRITE or PRUNE gates
- No hierarchy
- No fine-tuning of GPT-2
- Controller observations never use the true next-token label
- Datastore is strictly separate from controller_train and val queries
- Best fixed action chosen on controller_train, not validation
