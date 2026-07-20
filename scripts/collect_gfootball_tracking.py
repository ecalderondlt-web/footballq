"""Collect Google Research Football raw observations as JSONL tracking source.

This is a lightweight bridge: it records GRF raw observations, then
``GFootballAdapter`` converts those observations into footballq's canonical
tracking rows. The script requires an installed ``gfootball`` package, but the
core footballq package does not.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-name", default="11_vs_11_easy_stochastic")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--action-policy",
        choices=["random", "idle", "builtin_ai", "builtin_ai_perturbed"],
        default="random",
    )
    parser.add_argument("--perturbation-rate", type=float, default=0.05)
    parser.add_argument("--action-set", choices=["default", "v2", "full"], default="full")
    parser.add_argument("--extra-player", action="append", default=[])
    parser.add_argument("--match-prefix", default=None)
    parser.add_argument("--collection-job-id", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _constant_action(space: Any, value: int) -> Any:
    if hasattr(space, "n"):
        return min(int(value), int(space.n) - 1)
    if hasattr(space, "nvec"):
        return [min(int(value), int(n) - 1) for n in space.nvec]
    return value


def _sample_action(space: Any, rng: np.random.Generator) -> Any:
    if hasattr(space, "n"):
        return int(rng.integers(0, int(space.n)))
    if hasattr(space, "nvec"):
        return [int(rng.integers(0, int(n))) for n in space.nvec]
    if hasattr(space, "sample"):
        return space.sample()
    return 0


def _action(
    policy: str,
    space: Any,
    rng: np.random.Generator,
    perturbation_rate: float = 0.05,
) -> Any:
    if policy == "idle":
        return _constant_action(space, 0)
    if policy == "builtin_ai":
        return _constant_action(space, 19)
    if policy == "builtin_ai_perturbed":
        if rng.random() < perturbation_rate:
            return _sample_action(space, rng)
        return _constant_action(space, 19)
    return _sample_action(space, rng)


def _seed_environment(env: Any, seed: int) -> None:
    seed_fn = getattr(env, "seed", None)
    if not callable(seed_fn):
        raise RuntimeError("The GRF environment does not expose the required seed() API.")
    seed_fn(int(seed))
    action_seed_fn = getattr(getattr(env, "action_space", None), "seed", None)
    if callable(action_seed_fn):
        action_seed_fn(int(seed))


def main() -> None:
    args = parse_args()
    try:
        import gfootball.env as football_env
    except ImportError as exc:
        raise SystemExit(
            "The gfootball package is not installed. Install Google Research Football first, "
            "then rerun this collector. The footballq adapter can still consume saved JSONL "
            "observations without this optional dependency."
        ) from exc

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if not 0.0 <= float(args.perturbation_rate) <= 1.0:
        raise ValueError("perturbation-rate must lie in [0, 1].")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = football_env.create_environment(
        env_name=args.env_name,
        representation="raw",
        stacked=False,
        render=False,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        extra_players=args.extra_player or None,
        other_config_options={"action_set": args.action_set},
    )
    _seed_environment(env, args.seed)
    rows_written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        try:
            for episode_id in range(args.episodes):
                observation = env.reset()
                for frame_id in range(args.max_steps):
                    record = {
                        "provider": "google_research_football",
                        "env_name": args.env_name,
                        "episode_id": episode_id,
                        "frame_id": frame_id,
                        "time_s": frame_id / args.fps,
                        "fps": args.fps,
                        "collection_seed": args.seed,
                        "environment_seed": args.seed,
                        "action_policy": args.action_policy,
                        "perturbation_rate": args.perturbation_rate,
                        "collection_job_id": args.collection_job_id,
                        "split": args.split,
                        "observation": _jsonable(observation),
                    }
                    if args.match_prefix:
                        record["match_id"] = f"{args.match_prefix}_episode_{episode_id}"
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    rows_written += 1
                    action = _action(
                        args.action_policy,
                        env.action_space,
                        rng,
                        perturbation_rate=float(args.perturbation_rate),
                    )
                    observation, _, done, _ = env.step(action)
                    if done:
                        break
        finally:
            env.close()
    print(f"wrote {rows_written:,} GFootball observation frames to {args.out}")


if __name__ == "__main__":
    main()
