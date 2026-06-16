# Roadmap

## Phase 1: Data Foundation And Visualization

Ingest open tracking/event datasets, normalize coordinates, compute basic features, export model
windows, render clips, and provide a constant-velocity sanity baseline.

## Phase 2: Baseline Trajectory Prediction

Add stronger non-neural and lightweight learned baselines. Evaluate displacement error, ball-aware
movement, missingness, and split quality.

## Phase 3: Probabilistic Trajectory Model

Train stochastic multi-agent trajectory models over canonical windows. Add uncertainty-aware
rollouts and held-out match evaluation.

## Phase 4: Counterfactual Tactical Analysis

Model alternative defensive and attacking movements under different game states. Compare observed
and generated outcomes around shots, entries, and transitions.

## Phase 5: Pattern Discovery And Validation

Cluster generated and observed tactical patterns, link them to chance quality and goals, and
validate against held-out competitions and providers.

