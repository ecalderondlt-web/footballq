"""Run a blinded model-annotator panel over a rendered annotation package.

Each configured annotator is an independent headless CLI call (fresh context)
that receives only the annotation guide, the blind ids, and static contact
sheets rendered from the blinded GIFs. Raw CLI outputs are saved verbatim, the
controlled-vocabulary rows are parsed out mechanically, per-annotator filled
CSVs are produced, and the repository's own analyzers are invoked for
enrichment (which reads the private key) and panel agreement (which does not).

The orchestrator never passes key material to annotators; annotators never see
residual scores, cluster ids, or positive/control status.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ["tactical_pattern", "routine_motion", "tracking_artifact", "ambiguous"]
ROW_RE = re.compile(
    r"([A-Za-z0-9_\-]+)\s*,\s*(tactical_pattern|routine_motion|tracking_artifact|ambiguous)"
)

BASE_PROMPT = """You are a blinded annotator for football (soccer) tracking clips. Label only
what is visible.

Each clip is a static contact sheet PNG: 12 frames sampled in temporal order
(left-to-right, top-to-bottom; each tile labeled with its frame index) from a
~4 second top-down tracking animation. Markers are tracked player positions of
two teams plus the ball on a pitch diagram. There is no broadcast video and no
identities - only tracked positions. Players or the ball may be missing from
tracking in some frames; that is part of what you judge.

For each clip, choose exactly one label:
- tactical_pattern: coordinated movement, spacing, pressure, transition, or
  ball/player interaction that appears football-meaningful from the clip alone
- routine_motion: ordinary smooth movement with no clear motif
- tracking_artifact: missing players/ball, identity jump, impossible motion,
  rendering issue, or provider/tracking artifact
- ambiguous: not enough visual evidence to decide

Rules:
- Judge only what is visible in the sheet.
- Do not infer from file order, blind id, or file names.
- Mark tracking or rendering problems as tracking_artifact even if the clip
  also looks interesting.
- Use ambiguous when a clip could plausibly fit multiple labels.
- Any label outside the four above invalidates the submission.

Output STRICT CSV only: first line exactly `blind_id,annotation`, then one row
per clip in the order listed, no commentary, no code fences."""


def read_template(annotator_csv: Path) -> list[dict[str, str]]:
    with annotator_csv.open() as handle:
        return list(csv.DictReader(handle))


def build_prompt(items: list[tuple[str, str]], read_instruction: str) -> str:
    listing = "\n".join(f"- {blind_id}: {sheet}" for blind_id, sheet in items)
    return (
        f"{BASE_PROMPT}\n\n{read_instruction}\n\n"
        f"Clips to annotate ({len(items)}):\n{listing}\n"
    )


def run_cli(cmd: list[str], cwd: Path, timeout: int, log_path: Path) -> str:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        output = (partial or "") + "\n[TIMEOUT]"
    except FileNotFoundError:
        output = "[CLI NOT FOUND]"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("\n===== CMD: " + " ".join(cmd) + "\n")
        handle.write(output)
    return output


def parse_labels(raw: str) -> dict[str, str]:
    cleaned = raw.replace("\\n", "\n")
    labels: dict[str, str] = {}
    for match in ROW_RE.finditer(cleaned):
        blind_id, label = match.group(1), match.group(2)
        if blind_id.lower() != "blind_id":
            labels.setdefault(blind_id, label)
    return labels


def annotate_claude(
    model: str, sandbox: Path, items: list[tuple[str, str]], log_path: Path, timeout: int
) -> dict[str, str]:
    prompt = build_prompt(
        items,
        "Use your Read tool on each sheet PNG listed below (paths are relative to"
        " the current directory). Read every sheet before answering.",
    )
    output = run_cli(
        ["claude", "-p", prompt, "--model", model, "--allowedTools", "Read"],
        sandbox,
        timeout,
        log_path,
    )
    return parse_labels(output)


def annotate_codex(
    sandbox: Path,
    items: list[tuple[str, str]],
    log_path: Path,
    timeout: int,
    batch_size: int,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompt = build_prompt(
            batch,
            "The sheets for this batch are attached as images, in the same order"
            " as the list below.",
        )
        cmd = ["codex", "exec", "-s", "read-only", "-C", str(sandbox), "--skip-git-repo-check"]
        for _, sheet in batch:
            cmd += ["-i", str(sandbox / sheet)]
        cmd += ["--", prompt]
        output = run_cli(cmd, sandbox, timeout, log_path)
        labels.update(parse_labels(output))
    return labels


def annotate_kimi(
    sandbox: Path,
    items: list[tuple[str, str]],
    log_path: Path,
    timeout: int,
    batch_size: int,
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompt = build_prompt(
            batch,
            "View each sheet PNG listed below (paths are relative to your working"
            " directory). View every sheet in this batch before answering.",
        )
        cmd = [
            "kimi",
            "-w",
            str(sandbox),
            "--print",
            "--output-format",
            "text",
            "-p",
            prompt,
        ]
        output = run_cli(cmd, sandbox, timeout, log_path)
        labels.update(parse_labels(output))
    return labels


ANNOTATORS = {
    "fable": lambda sandbox, items, log, timeout, batch: annotate_claude(
        "claude-fable-5", sandbox, items, log, timeout
    ),
    "opus": lambda sandbox, items, log, timeout, batch: annotate_claude(
        "claude-opus-4-8", sandbox, items, log, timeout
    ),
    "sonnet": lambda sandbox, items, log, timeout, batch: annotate_claude(
        "claude-sonnet-5", sandbox, items, log, timeout
    ),
    "codex": annotate_codex,
    "kimi": annotate_kimi,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--annotators", nargs="+", default=list(ANNOTATORS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--sheets-dir", type=Path, default=None)
    args = parser.parse_args()

    package = args.package_dir.resolve()
    annotator_csv = package / "annotator" / "annotations.csv"
    sheets_dir = args.sheets_dir or (package / "annotator" / "sheets")
    out_dir = args.out_dir or (package / "panel")
    out_dir.mkdir(parents=True, exist_ok=True)

    template = read_template(annotator_csv)
    items = []
    for row in template:
        blind_id = row["blind_id"]
        sheet = sheets_dir / f"{blind_id}_sheet.png"
        if not sheet.exists():
            raise SystemExit(f"missing contact sheet for {blind_id}: {sheet}")
        items.append((blind_id, f"sheets/{sheet.name}"))

    (out_dir / "PROMPT.md").write_text(
        build_prompt(items, "(read instruction varies per annotator CLI)")
    )

    status: dict[str, dict[str, object]] = {}
    for name in args.annotators:
        if name not in ANNOTATORS:
            raise SystemExit(f"unknown annotator {name!r}; known: {sorted(ANNOTATORS)}")
        annotator_dir = out_dir / name
        sandbox = annotator_dir / "sandbox"
        sandbox_sheets = sandbox / "sheets"
        sandbox_sheets.mkdir(parents=True, exist_ok=True)
        for _, sheet_rel in items:
            source = sheets_dir / Path(sheet_rel).name
            target = sandbox_sheets / Path(sheet_rel).name
            if not target.exists():
                shutil.copy2(source, target)
        print(f"[panel] running annotator: {name}", flush=True)
        labels = ANNOTATORS[name](
            sandbox, items, annotator_dir / "raw_output.log", args.timeout, args.batch_size
        )
        filled = 0
        out_rows = []
        for row in template:
            new_row = dict(row)
            label = labels.get(row["blind_id"], "")
            if label in ALLOWED:
                new_row["annotation"] = label
                filled += 1
            out_rows.append(new_row)
        filled_csv = annotator_dir / "annotations.csv"
        with filled_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(template[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        status[name] = {"filled": filled, "total": len(template)}
        print(f"[panel] {name}: {filled}/{len(template)} labels parsed", flush=True)

    (out_dir / "panel_run_status.json").write_text(json.dumps(status, indent=2))
    complete = [n for n, s in status.items() if s["filled"] == len(template)]
    print(
        json.dumps(
            {"annotators_run": list(status), "complete": complete, "out_dir": str(out_dir)}
        )
    )
    if len(complete) < 2:
        sys.exit("fewer than two complete annotators; panel statistics not meaningful")


if __name__ == "__main__":
    main()
