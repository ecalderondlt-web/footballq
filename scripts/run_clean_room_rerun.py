"""Clean-room rerun driver for the v2 candidate representation.

Reproduces the full integrity-gate pipeline (train -> falsification -> probes ->
discovery -> combined gate) for the current candidate config on an independent
machine, using the canonical artifact names from docs/NEXT_WEEK_RUNBOOK.md so the
resulting gate summaries are directly comparable with the ones reported in
docs/INTEGRITY_SPRINT_README.md.

This script adds no scientific logic. Every step shells out to the existing
scientific entry points with --split-manifest / --scientific-mode semantics
inherited from configs and CLI flags documented in the runbook.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "v2_context_w0p05_slot_recon_margin"
CONFIG = "configs/td_jepa_nonoverlap_gap1p0_context_w0p05_slot_recon_margin_skillcorner.yaml"
TD_DATA = "data/processed/skillcorner_td_jepa_nonoverlap_gap1p0.pt"
WINDOWS = "data/processed/skillcorner_windows_h2s.pt"
SPLIT = "splits/skillcorner_10match_inductive_v1.json"
SEEDS = [7, 11, 23]
BASELINE_FEATURES = [
    "raw_delta_z",
    "pca_delta_z",
    "random_encoder_delta_z",
    "handcrafted_structure_metrics",
    "pca_handcrafted_structure_metrics",
]
FALSIFICATION_GATE_DIR = (
    "runs/td_jepa/v2_nonoverlap_geometry_gap1p0_context_w0p05_slot_recon_margin"
    "_falsification_gate_extended"
)
PROBE_SUMMARY_DIR = f"runs/probe_suite/{CANDIDATE}_h2s_linear_incremental_summary"
DISCOVERY_SUMMARY_DIR = f"runs/discovery/{CANDIDATE}_control_summary"
GATE_OUT = f"runs/integrity/{CANDIDATE}_gate_summary.json"
STATE_PATH = ROOT / "runs" / "clean_room_state.json"
LOG_DIR = ROOT / "runs" / "logs"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"run_dirs": {}, "embeddings": {}, "completed": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_step(name: str, cmd: list[str], state: dict) -> str:
    if name in state["completed"]:
        print(f"[skip] {name} (already completed)")
        return ""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    print(f"[run ] {name}: {' '.join(cmd)}")
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log.write(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        raise SystemExit(f"step {name} failed with exit code {proc.returncode}")
    state["completed"].append(name)
    save_state(state)
    return proc.stdout


def stage_train(state: dict) -> None:
    for seed in SEEDS:
        name = f"train_seed{seed}"
        if name in state["completed"]:
            continue
        stdout = run_step(
            name,
            [sys.executable, "scripts/train_td_jepa.py", "--config", CONFIG, "--seed", str(seed)],
            state,
        )
        match = re.search(r"run_dir: (\S+)", stdout)
        if not match:
            raise SystemExit(f"could not parse run_dir from train output for seed {seed}")
        state["run_dirs"][str(seed)] = match.group(1)
        save_state(state)


def stage_falsification(state: dict) -> None:
    for seed in SEEDS:
        run_dir = state["run_dirs"][str(seed)]
        run_step(
            f"falsification_seed{seed}",
            [
                sys.executable,
                "scripts/run_td_falsification_controls.py",
                "--checkpoint",
                f"{run_dir}/best.pt",
                "--data",
                TD_DATA,
                "--out",
                f"{run_dir}/falsification_val",
                "--split",
                "val",
            ],
            state,
        )
    summary_args = []
    for seed in SEEDS:
        run_dir = state["run_dirs"][str(seed)]
        summary_args += [
            "--summary",
            f"{seed}:{run_dir}/falsification_val/td_falsification_summary.json",
        ]
    run_step(
        "falsification_gate",
        [
            sys.executable,
            "scripts/summarize_td_falsification.py",
            *summary_args,
            "--metric",
            "total_loss",
            "--out",
            FALSIFICATION_GATE_DIR,
        ],
        state,
    )


def stage_embeddings(state: dict) -> None:
    for seed in SEEDS:
        run_dir = state["run_dirs"][str(seed)]
        out = f"data/processed/skillcorner_td_embeddings_{CANDIDATE}_seed{seed}_all.pt"
        run_step(
            f"export_embeddings_seed{seed}",
            [
                sys.executable,
                "scripts/export_td_embeddings.py",
                "--checkpoint",
                f"{run_dir}/best.pt",
                "--data",
                TD_DATA,
                "--out",
                out,
                "--split",
                "all",
            ],
            state,
        )
        state["embeddings"][str(seed)] = out
        save_state(state)


def stage_probes(state: dict) -> None:
    for seed in SEEDS:
        embeddings = state["embeddings"][str(seed)]
        dataset = f"data/processed/skillcorner_probe_dataset_{CANDIDATE}_seed{seed}.pt"
        run_step(
            f"probe_dataset_seed{seed}",
            [
                sys.executable,
                "scripts/build_probe_dataset.py",
                "--embeddings",
                embeddings,
                "--windows",
                WINDOWS,
                "--out",
                dataset,
                "--targets",
                "future_ball_global_x_bucket",
                "future_ball_displacement_m",
                "team_shape_change_bucket",
                "--split-manifest",
                SPLIT,
                "--scientific-mode",
            ],
            state,
        )
        run_step(
            f"probe_suite_seed{seed}",
            [
                sys.executable,
                "scripts/run_probe_suite.py",
                "--dataset",
                dataset,
                "--out",
                f"runs/probe_suite/{CANDIDATE}_seed{seed}_h2s_linear",
                "--linear-only",
            ],
            state,
        )
    suite_args = []
    for seed in SEEDS:
        suite_args += [
            "--suite",
            f"{seed}:runs/probe_suite/{CANDIDATE}_seed{seed}_h2s_linear/results.json",
        ]
    run_step(
        "probe_incremental_summary",
        [
            sys.executable,
            "scripts/summarize_probe_incremental.py",
            *suite_args,
            "--out",
            PROBE_SUMMARY_DIR,
        ],
        state,
    )


def _locate_cluster_summary(base: str) -> str:
    base_path = ROOT / base
    direct = base_path / "cluster_summary.json"
    if direct.exists():
        return str(direct.relative_to(ROOT))
    matches = sorted(base_path.glob("delta_0p2s/cluster_summary.json")) or sorted(
        base_path.glob("**/cluster_summary.json")
    )
    if not matches:
        raise SystemExit(f"no cluster_summary.json found under {base}")
    return str(matches[0].relative_to(ROOT))


def stage_discovery(state: dict) -> None:
    for seed in SEEDS:
        embeddings = state["embeddings"][str(seed)]
        latent_dir = f"runs/discovery/{CANDIDATE}_seed{seed}"
        run_step(
            f"discovery_latent_seed{seed}",
            [
                sys.executable,
                "scripts/run_discovery_suite.py",
                "--config",
                "configs/discovery_suite_skillcorner.yaml",
                "--embeddings",
                embeddings,
                "--windows",
                WINDOWS,
                "--out",
                latent_dir,
                "--seed",
                str(seed),
                "--split-manifest",
                SPLIT,
                "--scientific-mode",
            ],
            state,
        )
        for feature in BASELINE_FEATURES:
            run_step(
                f"discovery_{feature}_seed{seed}",
                [
                    sys.executable,
                    "scripts/cluster_latent_transitions.py",
                    "--dataset",
                    f"{latent_dir}/transition_dataset.pt",
                    "--out",
                    f"runs/discovery/{CANDIDATE}_seed{seed}_baselines/{feature}",
                    "--feature",
                    feature,
                    "--seed",
                    str(seed),
                    "--delta-seconds",
                    "0.2",
                    "--k",
                    "8",
                    "16",
                    "32",
                    "64",
                    "--max-iter",
                    "12",
                    "--fit-sample-size",
                    "20000",
                ],
                state,
            )
    summary_args = []
    for seed in SEEDS:
        latent_dir = f"runs/discovery/{CANDIDATE}_seed{seed}"
        latent_summary = _locate_cluster_summary(latent_dir)
        summary_args += ["--cluster-summary", f"normalized_delta_z:{seed}:{latent_summary}"]
        for feature in BASELINE_FEATURES:
            base = f"runs/discovery/{CANDIDATE}_seed{seed}_baselines/{feature}"
            located = _locate_cluster_summary(base)
            summary_args += ["--cluster-summary", f"{feature}:{seed}:{located}"]
    run_step(
        "discovery_control_summary",
        [
            sys.executable,
            "scripts/summarize_discovery_controls.py",
            *summary_args,
            "--out",
            DISCOVERY_SUMMARY_DIR,
        ],
        state,
    )


def stage_gate(state: dict) -> None:
    run_step(
        "integrity_gate",
        [
            sys.executable,
            "scripts/summarize_integrity_gates.py",
            "--falsification",
            f"{FALSIFICATION_GATE_DIR}/td_falsification_gate_summary.json",
            "--probe",
            f"{PROBE_SUMMARY_DIR}/probe_incremental_summary.json",
            "--discovery",
            f"{DISCOVERY_SUMMARY_DIR}/discovery_control_summary.json",
            "--out",
            GATE_OUT,
        ],
        state,
    )


STAGES = {
    "train": stage_train,
    "falsification": stage_falsification,
    "embeddings": stage_embeddings,
    "probes": stage_probes,
    "discovery": stage_discovery,
    "gate": stage_gate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    args = parser.parse_args()
    state = load_state()
    for stage in args.stages:
        print(f"=== stage: {stage} ===")
        STAGES[stage](state)
    print("clean-room rerun stages complete:", ", ".join(args.stages))


if __name__ == "__main__":
    main()
