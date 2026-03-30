# Dynamic Swarm Generation Implementation Plan

## Goal
Build a robust architect-driven system that generates request-specific swarm workflows while remaining budget-aware, measurable against single-shot baselines, and stable in production.

## Scope
- In scope:
  - Dynamic workflow generation from user request intent.
  - Workflow constraints and policy enforcement.
  - Budget-aware model routing and execution.
  - A/B evaluation against single-shot baseline.
  - Observability, artifact consistency, and quality metrics.
- Out of scope:
  - Full RL-based optimization.
  - Human preference model training.

## Current Baseline
- Architect already produces DAG plans and budget checks.
- Executor already runs nodes with retries and cost tracking.
- A/B mode exists with swarm vs single-shot execution.
- Swarm config selector exists with pinned and auto modes.

## Phase 1: Workflow Contracting (Pinned + Dynamic)
### Objective
Define clear workflow contracts so generated plans are comparable and testable.

### Deliverables
- Workflow template registry:
  - `single_writer`
  - `research_analyze_write`
  - `single_coder`
  - `dynamic` (architect free-form)
- Strict validator for:
  - Node count
  - Dependency order
  - Allowed agent types
  - Search/tool capability requirements
- Repair-on-violation behavior:
  - If generated plan violates selected workflow contract, re-plan once with explicit violation feedback.

### Acceptance Criteria
- 100% of pinned-config runs satisfy contract shape.
- Violation reports appear in run artifacts.

## Phase 2: Dynamic Planner Policy Layer
### Objective
Make dynamic generation reliable, not purely prompt-driven.

### Deliverables
- Intent router that maps request classes to candidate workflow families.
- Policy guardrails:
  - Max node count by task class and budget.
  - Search-required gating for live-data tasks.
  - Approval-required auto-tagging for sensitive actions.
- Dynamic fallback policy:
  - If budget too tight, collapse to minimal viable workflow.

### Acceptance Criteria
- Dynamic planner picks valid family with >= 90% consistency on repeated prompts.
- Budget feasibility fallback always returns a valid executable plan.

## Phase 3: Quality Control Pipeline
### Objective
Increase output quality reliability for multi-node swarms.

### Deliverables
- Optional `qa` reviewer node policy for long-form/coding workflows.
- Structured quality checks:
  - Coherence
  - Factual risk flags
  - Formatting completeness
- Self-correction strategy:
  - Targeted rewrite or node rerun based on failure category.

### Acceptance Criteria
- Reduced retry/failure rate for long-form tasks.
- Improved quality score variance across runs.

## Phase 4: Fair Evaluation Harness (A/B)
### Objective
Produce trustworthy evidence that swarm adds value.

### Deliverables
- Pinned-config-only A/B enforcement.
- Cost-normalized comparison policy:
  - Allow cheaper baseline (efficiency win tracking).
  - Also support optional strict cost band mode for parity experiments.
- Standardized result schema:
  - Cost, latency, retries, failures
  - Quality score bundle
  - Win/loss decision with reason labels

### Acceptance Criteria
- Every A/B run produces complete comparable artifacts.
- No artifact path mismatch between reported and actual location.

## Phase 5: Observability and Data Products
### Objective
Turn experiments into decisions.

### Deliverables
- Unified artifact folder for each experiment with both arms.
- Experiment index file (or DB table) with summary rows.
- Dashboard-ready aggregates:
  - Win rate by config
  - Cost per quality point
  - Failure categories

### Acceptance Criteria
- Team can rank configs by quality-cost frontier over N>=20 runs/config.

## Data Model (Recommended)
- `swarm_configs`
  - `id`, `name`, `workflow`, `policy`, `version`, `enabled`
- `experiments`
  - `id`, `prompt_hash`, `config_id`, `baseline_model`, `budget`, `created_at`
- `experiment_runs`
  - `experiment_id`, `arm`, `cost`, `latency_ms`, `retry_count`, `quality_score`, `status`

## Rollout Plan
1. Internal alpha with 3 pinned configs + dynamic mode.
2. Enable A/B for pinned configs only.
3. Collect 1 week of experiment data.
4. Promote top-performing configs; demote unstable ones.

## Risks and Mitigations
- Risk: Dynamic planner drifts in shape.
  - Mitigation: Contract validator + re-plan feedback.
- Risk: Swarm quality appears worse than single-shot at higher cost.
  - Mitigation: Add QA/rewrite policy only where ROI is positive.
- Risk: Evaluation noise from prompt variance.
  - Mitigation: Fixed prompt sets and repeated trials.

## Suggested Milestones
- Milestone A (2-3 days): Contract validator + pinned workflow hard enforcement.
- Milestone B (3-5 days): Dynamic policy layer + fallback rules.
- Milestone C (3-5 days): Quality checks + A/B scoring + experiment index.

## Definition of Done
- Dynamic generation is stable under budget constraints.
- Pinned configs are deterministic in workflow structure.
- A/B outputs are complete, unified, and decision-grade.
- Team can identify when swarm truly beats single-shot for a given task family.
