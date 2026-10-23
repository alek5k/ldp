#!/usr/bin/env python3
"""Retain one scored checkpoint and remove ``latest.ckpt`` for each LDP run."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


SCORED_CHECKPOINT_PATTERN = re.compile(
    r"^epoch=(?P<epoch>\d+)-(?P<metric>test_mean_score|val_loss)="
    r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.ckpt$"
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str
    retained_path: Path | None


def select_checkpoint(paths: list[Path]) -> Path:
    """Select the best scored checkpoint from one run's checkpoint folder."""
    parsed = []
    for path in paths:
        match = SCORED_CHECKPOINT_PATTERN.match(path.name)
        if match is None:
            raise ValueError(f"Unrecognised scored checkpoint name: {path}")
        parsed.append((
            path,
            match.group("metric"),
            float(match.group("value")),
            int(match.group("epoch")),
        ))

    metrics = {metric for _, metric, _, _ in parsed}
    if len(metrics) != 1:
        raise ValueError(f"Mixed checkpoint metrics in {paths[0].parent}: {metrics}")
    metric = metrics.pop()
    if metric == "test_mean_score":
        return max(parsed, key=lambda item: (item[2], item[3], item[0].stat().st_mtime))[0]
    return min(parsed, key=lambda item: (item[2], -item[3], -item[0].stat().st_mtime))[0]


def iter_checkpoint_directories(root: Path):
    for directory, _, filenames in os.walk(root, followlinks=True):
        if Path(directory).name == "checkpoints":
            yield Path(directory), filenames


def iter_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for directory, filenames in iter_checkpoint_directories(root):
        checkpoint_paths = [directory / name for name in filenames if name.endswith(".ckpt")]
        scored_paths = [path for path in checkpoint_paths if path.name != "latest.ckpt"]
        if scored_paths:
            retained_path = select_checkpoint(scored_paths)
            candidates.extend(
                Candidate(path, "superseded_scored_checkpoint", retained_path)
                for path in scored_paths
                if path != retained_path
            )
        candidates.extend(
            Candidate(directory / name, "resume_checkpoint", None)
            for name in filenames
            if name == "latest.ckpt"
        )
    return sorted(candidates, key=lambda candidate: candidate.path)


def write_manifest(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("path\tbytes\treason\tretained_checkpoint\n")
        for candidate in candidates:
            retained = str(candidate.retained_path) if candidate.retained_path else ""
            file.write(
                f"{candidate.path}\t{candidate.path.stat().st_size}\t"
                f"{candidate.reason}\t{retained}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="delete without prompting")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    candidates = iter_candidates(root)
    total_bytes = sum(candidate.path.stat().st_size for candidate in candidates)
    print(f"Found {len(candidates)} checkpoints to delete, reclaiming {total_bytes} bytes.")
    if args.manifest is not None:
        write_manifest(candidates, args.manifest.expanduser())
    if args.dry_run:
        for candidate in candidates:
            print(f"[dry-run] {candidate.reason}: {candidate.path}")
        return
    if not args.yes:
        answer = input("Delete these checkpoints? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("No checkpoints deleted.")
            return
    for candidate in candidates:
        candidate.path.unlink()
    print(f"Deleted {len(candidates)} checkpoints.")


if __name__ == "__main__":
    main()
