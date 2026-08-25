"""Upload reconstructible completed-run artifacts to Weights & Biases."""

from __future__ import annotations

from pathlib import Path
def sync_checkpoints_to_wandb(
    wandb_module: object,
    wandb_run: object,
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    epoch: int,
) -> int:
    """Log every checkpoint file as one W&B model artifact and return its count."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(path for path in checkpoint_dir.glob("*.ckpt") if path.is_file())
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints were written in: {checkpoint_dir}")

    run_name = Path(output_dir).name
    artifact = wandb_module.Artifact(
        name=f"{run_name}-checkpoints",
        type="model",
        metadata={
            "final_epoch": epoch,
            "checkpoint_count": len(checkpoints),
        },
    )
    for checkpoint in checkpoints:
        artifact.add_file(str(checkpoint), name=f"checkpoints/{checkpoint.name}")
    wandb_run.log_artifact(artifact, aliases=["latest"])
    return len(checkpoints)


def sync_run_folder_to_wandb(
    wandb_module: object,
    wandb_run: object,
    *,
    output_dir: str | Path,
    epoch: int,
) -> int:
    """Upload every regular file in a completed training folder as one artifact.

    Artifacts retain the relative file paths, so downloading the artifact
    recreates the run directory's configs, logs, media, checkpoints, and W&B
    local metadata. Symlinks are intentionally skipped because W&B artifacts
    store file content rather than portable link metadata.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Training output directory does not exist: {output_dir}")

    artifact = wandb_module.Artifact(
        name=f"{output_dir.name}-run-folder",
        type="run-folder",
        metadata={
            "final_epoch": epoch,
            "output_dir_name": output_dir.name,
        },
    )
    file_count = 0
    byte_count = 0
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative_path = path.relative_to(output_dir)
        artifact.add_file(str(path), name=str(relative_path))
        file_count += 1
        byte_count += path.stat().st_size

    if not file_count:
        raise FileNotFoundError(f"No regular files found in: {output_dir}")

    artifact.metadata["file_count"] = file_count
    artifact.metadata["byte_count"] = byte_count
    logged_artifact = wandb_run.log_artifact(artifact, aliases=["latest"])
    if hasattr(logged_artifact, "wait"):
        logged_artifact.wait()
    return file_count
