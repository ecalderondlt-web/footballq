# Literature And Design References

## TacticAI

TacticAI motivates representing football situations geometrically and relationally. Future phases
can turn canonical frame-agent tables into graph inputs with players, ball, teams, and event
context.

## FootBots

FootBots-style multi-agent motion prediction motivates fixed-length trajectory windows and
agent masks. Phase 1 prepares `[time, agent, feature]` tensors without adding transformer or deep
learning dependencies.

## Data-Driven Ghosting

Ghosting work motivates counterfactual defensive movement: where defenders might have moved under
an expected policy. Phase 1 keeps velocities, ball distances, team shape, and stable agent ordering
so later ghost models can be trained.

## GenTac

GenTac motivates stochastic tactical trajectory rollouts. Phase 1 only includes a deterministic
constant-velocity baseline, leaving probabilistic generative modeling for later phases.

