#!/usr/bin/env python3
"""Launch LTE-IMG-NoT training and evaluation runs in detached screen sessions."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train_lte_img_not.sh"
TRAIN_PY = REPO_ROOT / "train.py"
EVAL_SCRIPT = REPO_ROOT / "eval.py"
TASKS = (
    "square",
    "tool_hang",
    "transport",
    "lh-aloha",
    "lh-square",
    "waitatgoal",
    "liftqa",
)
UNET_ONLY_TASKS = frozenset(("lh-square", "waitatgoal", "liftqa"))
# ``sim_max_reward`` is the maximum per-step reward for the standard
# Robomimic tasks, but the temporal runner records the episode reward sum.
# WaitAtGoal and LiftQA award 0.3, 0.6, then 1.0 for complete trajectories.
SUCCESS_REWARD_THRESHOLDS = {
    "square": 1.0,
    "tool_hang": 1.0,
    "transport": 1.0,
    "lh-aloha": 4.0,
    "lh-square": 1.0,
    "waitatgoal": 1.9,
    "liftqa": 1.9,
}
ZARR_RUNNER_MODULES = (
    "temporal_image_runner",
    "robomimic_image_runner",
    "robomimic_longhist_image_runner",
    "aloha_image_runner",
)
CLI_LOG_DIR = REPO_ROOT / "data" / "cli_logs"
LOCAL_OUTPUT_ROOT = REPO_ROOT / "data" / "outputs"
EXTERNAL_OUTPUT_ROOT = REPO_ROOT / "data" / "outputs_extdrive"
# New training runs live on the external drive by default. Local and external
# roots are both still searched by the progress and analysis tools.
DEFAULT_OUTPUT_ROOT = EXTERNAL_OUTPUT_ROOT
LAUNCHER_EPOCH_CONFIGS = {
    "square": "square/lte_img_not.yaml",
    "square-unet": "square/lte_img_not.yaml",
    "tool_hang": "tool/lte_img_not.yaml",
    "tool_hang-unet": "tool/lte_img_not.yaml",
    "transport": "transport/lte_img_not.yaml",
    "transport-unet": "transport/lte_img_not.yaml",
    "lh-aloha": "aloha/lte_img_not.yaml",
    "lh-aloha-unet": "aloha/lte_img_not.yaml",
    "lh-square": "longhist/lte_img_not.yaml",
    "lh-square-unet": "longhist/lte_img_not.yaml",
    "waitatgoal": "waitatgoal/lte_img_not.yaml",
    "liftqa": "liftqa/lte_img_not.yaml",
}
PTP_TASK_CONFIGS = {
    "lh-aloha": REPO_ROOT / "experiment_configs" / "aloha" / "transformer_aloha.yaml",
    "lh-square": REPO_ROOT / "experiment_configs" / "longhist" / "transformer_longhist.yaml",
}
STALE_PROGRESS_SECONDS = 15 * 60
RUN_NOTE_FILENAME = "note.txt"
RUN_NOTE_MAX_LENGTH = 40
ANSI_COLORS = {
    "complete": "\033[32m",  # green
    "active": "\033[34m",    # blue
    "stale": "\033[31m",     # red
    "failed": "\033[31m",    # red
}
ANSI_RESET = "\033[0m"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _prompt_menu(prompt: str, choices: list[str]) -> str:
    while True:
        print()
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice}")
        answer = input(prompt).strip()
        try:
            return choices[int(answer) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(choices)}.")


def _prompt_int(prompt: str, default: int, minimum: int = 0) -> int:
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
            if value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Enter an integer greater than or equal to {minimum}.")


def _prompt_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        answer = input(f"{prompt} [{default:g}]: ").strip()
        if not answer:
            return default
        try:
            value = float(answer)
            if not math.isfinite(value) or value < minimum:
                raise ValueError
            return value
        except ValueError:
            print(f"Enter a finite number greater than or equal to {minimum:g}.")


def _prompt_text(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Enter y or n.")


def _print_gpu_usage() -> None:
    """Print memory and compute utilisation before choosing a GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("GPU usage unavailable: nvidia-smi was not found on PATH.")
        return
    if result.returncode:
        print("GPU usage unavailable: nvidia-smi could not query the GPUs.")
        return

    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        index, used_text, total_text, utilisation = fields
        try:
            used = float(used_text)
            total = float(total_text)
            memory_percent = 100.0 * used / total if total else 0.0
            memory_label = f"{used:g}/{total:g} MiB"
            percent_label = f"{memory_percent:.0f}%"
        except ValueError:
            memory_label = f"{used_text}/{total_text} MiB"
            percent_label = "?"
        try:
            utilisation_label = f"{float(utilisation):g}%"
        except ValueError:
            utilisation_label = utilisation
        rows.append((index, memory_label, percent_label, utilisation_label))
    if not rows:
        print("GPU usage unavailable: nvidia-smi returned no GPU records.")
        return

    index_width = max(len("GPU"), *(len(index) for index, _, _, _ in rows))
    memory_width = max(len("Memory"), *(len(memory) for _, memory, _, _ in rows))
    percent_width = max(len("Mem %"), *(len(percent) for _, _, percent, _ in rows))
    utilisation_width = max(len("GPU util"), *(len(utilisation) for _, _, _, utilisation in rows))
    print("\nCurrent GPU usage")
    print(
        f"  {'GPU':>{index_width}} | {'Memory':>{memory_width}} | "
        f"{'Mem %':>{percent_width}} | {'GPU util':>{utilisation_width}}"
    )
    print(
        f"  {'-' * index_width}-|-{'-' * memory_width}-|-"
        f"{'-' * percent_width}-|-{'-' * utilisation_width}"
    )
    for index, memory, percent, utilisation in rows:
        print(
            f"  {index:>{index_width}} | {memory:>{memory_width}} | "
            f"{percent:>{percent_width}} | {utilisation:>{utilisation_width}}"
        )


def _prompt_gpu() -> str:
    _print_gpu_usage()
    while True:
        answer = input("GPU index (CUDA_VISIBLE_DEVICES) [0]: ").strip()
        if not answer:
            return "0"
        if answer.isdigit():
            return answer
        print("Enter one numeric GPU index, for example 0 or 1.")


def _make_screen_session(label: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-_") or "job"
    # The uniqueness suffix below already contains a timestamp; remove any
    # timestamp inherited from a run/checkpoint label to keep sessions readable.
    safe_label = re.sub(r"(?:^|[-_])\d{8}[-_]\d{6}(?=$|[-_])", "", safe_label)
    safe_label = re.sub(r"[-_]{2,}", "-", safe_label).strip("-_") or "job"
    suffix = f"-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    # GNU screen imposes a short session-name limit.  Evaluation labels can
    # include both a timestamped run name and a checkpoint filename, so retain
    # a descriptive prefix while reserving space for the unique suffix.
    max_session_length = 50
    label_length = max_session_length - len("ldp-") - len(suffix)
    safe_label = safe_label[:label_length].rstrip("-_") or "job"
    return f"ldp-{safe_label}{suffix}"


def _start_screen_session(session: str, command: str) -> None:
    screen = shutil.which("screen")
    if screen is None:
        raise RuntimeError("screen was not found on PATH; cannot launch the experiment.")

    script_path = Path("/tmp") / f"{session}.sh"
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CLI_LOG_DIR / f"{session}.log"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "source ~/.bashrc 2>/dev/null || true\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"exec > >(tee -a {shlex.quote(str(log_path))}) 2>&1\n"
        f"{command}\n"
        "exit_code=$?\n"
        "if [ \"$exit_code\" -eq 0 ]; then\n"
        "    exit 0\n"
        "fi\n"
        "echo \"Command failed with status $exit_code; leaving this screen session open.\"\n",
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    print(f"\nScreen session: {session}")
    print(f"Attach with: screen -r {session}")
    print(f"Command: {command}")
    print(f"Log: {log_path.relative_to(REPO_ROOT)}")
    created = subprocess.run(
        [screen, "-T", "screen-256color", "-S", session, "-dm", "bash", "-l", "-i"],
        cwd=REPO_ROOT,
        check=False,
    )
    if created.returncode:
        raise RuntimeError(f"Could not create screen session (status {created.returncode}).")
    sent = subprocess.run(
        [screen, "-S", session, "-p", "0", "-X", "stuff", f"source {script_path}\r"],
        cwd=REPO_ROOT,
        check=False,
    )
    if sent.returncode:
        raise RuntimeError(f"Could not send the command to screen (status {sent.returncode}).")


def _environment_prefix(gpu: str) -> str:
    # train_lte_img_not.sh calls `python`; prioritise the interpreter that
    # started this CLI so the screen process uses the same Conda environment.
    python_bin = Path(sys.executable).resolve().parent
    return " ".join((
        "MUJOCO_GL=egl",
        "MPLCONFIGDIR=/tmp/ldp_mpl",
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}",
        f"ROBOMIMIC_RENDER_GPU_DEVICE={shlex.quote(gpu)}",
        f"PATH={shlex.quote(str(python_bin))}:$PATH",
    ))


def _run_name(
    task: str,
    decoder: str,
    seed: int,
    *,
    temporal_embedding_cache: bool = False,
    cache_start_epoch: int = 5,
    cache_warmup_epochs: int = 20,
    cache_refresh_epochs: int = 5,
) -> str:
    """Return a timestamped LTE run name without encoding hyperparameters."""
    del seed, temporal_embedding_cache, cache_start_epoch
    del cache_warmup_epochs, cache_refresh_epochs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{task.replace('-', '_')}_lte_{decoder}_{timestamp}"


def _ptp_run_name(task: str) -> str:
    """Return the timestamped run name used by past-token-prediction training."""
    return f"{task.replace('-', '_')}_ptp_{time.strftime('%Y%m%d_%H%M%S')}"


def _next_available_path(path: Path) -> Path:
    """Keep a readable run name while avoiding collisions with prior runs."""
    if not _path_is_occupied(path):
        return path
    attempt = 2
    while True:
        candidate = path.with_name(f"{path.name}-r{attempt}")
        if not _path_is_occupied(candidate):
            return candidate
        attempt += 1


def _path_is_occupied(path: Path) -> bool:
    """Treat a dangling symlink as occupied too, so a launch cannot overwrite it."""
    return path.exists() or path.is_symlink()


def _training_output_roots() -> tuple[Path, ...]:
    """Return output roots once each, preferring the external-drive archive."""
    roots = []
    seen = set()
    for root in (EXTERNAL_OUTPUT_ROOT, LOCAL_OUTPUT_ROOT):
        identity = root.resolve(strict=False)
        if identity in seen:
            continue
        seen.add(identity)
        roots.append(root)
    return tuple(roots)


def _training_run_directories() -> list[Path]:
    """Discover direct run directories across local and external output roots."""
    runs = {}
    for output_root in _training_output_roots():
        if not output_root.is_dir():
            continue
        for run_dir in sorted(output_root.iterdir(), key=lambda path: path.name):
            if run_dir.is_dir() and not run_dir.is_symlink():
                runs.setdefault(run_dir.name, run_dir)
    return sorted(runs.values(), key=lambda path: path.name)


def _next_available_training_paths(output_root: Path, name: str) -> tuple[Path, Path]:
    """Find matching train/eval directories whose shared run name is unused."""
    inference_root = REPO_ROOT / "data" / "inference"
    output_roots = (output_root, *_training_output_roots())
    attempt = 1
    while True:
        run_name = name if attempt == 1 else f"{name}-r{attempt}"
        output_dir = output_root / run_name
        inference_dir = inference_root / run_name
        if (
            not any(_path_is_occupied(root / run_name) for root in output_roots)
            and not _path_is_occupied(inference_dir)
        ):
            return output_dir, inference_dir
        attempt += 1


def _task_name(decoder: str) -> tuple[str, str, str]:
    """Return the task name, launcher argument, and supported decoder."""
    task = _prompt_menu("Select task: ", list(TASKS))
    if task in UNET_ONLY_TASKS:
        if decoder != "unet":
            print(f"{task} currently has an LTE U-Net preset; using U-Net.")
        return task, task, "unet"
    launcher_task = task if decoder == "transformer" else f"{task}-unet"
    return task, launcher_task, decoder


def _planned_runs(
    task: str,
    decoder: str,
    *,
    temporal_embedding_cache: bool = False,
    cache_start_epoch: int = 5,
    cache_warmup_epochs: int = 20,
    cache_refresh_epochs: int = 5,
) -> list[tuple[int, Path, Path]]:
    run_count = _prompt_int("Sequential runs", default=1, minimum=1)
    first_seed = _prompt_int("First training seed", default=42, minimum=0)
    default_output_root = DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)
    output_root = Path(
        input(f"Training output root [{default_output_root}]: ").strip()
        or default_output_root
    )
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    runs = []
    for offset in range(run_count):
        seed = first_seed + offset
        name = _run_name(
            task,
            decoder,
            seed,
            temporal_embedding_cache=temporal_embedding_cache,
            cache_start_epoch=cache_start_epoch,
            cache_warmup_epochs=cache_warmup_epochs,
            cache_refresh_epochs=cache_refresh_epochs,
        )
        output_dir, inference_dir = _next_available_training_paths(output_root, name)
        runs.append((seed, output_dir, inference_dir))
    return runs


def _planned_ptp_runs(task: str) -> list[tuple[int, Path, Path]]:
    """Plan PTP runs with the same output and collision rules as LTE runs."""
    run_count = _prompt_int("Sequential runs", default=1, minimum=1)
    first_seed = _prompt_int("First training seed", default=42, minimum=0)
    default_output_root = DEFAULT_OUTPUT_ROOT.relative_to(REPO_ROOT)
    output_root = Path(
        input(f"Training output root [{default_output_root}]: ").strip()
        or default_output_root
    )
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    runs = []
    for offset in range(run_count):
        seed = first_seed + offset
        output_dir, inference_dir = _next_available_training_paths(
            output_root, _ptp_run_name(task)
        )
        runs.append((seed, output_dir, inference_dir))
    return runs


def _config_default_epochs(launcher_task: str) -> int:
    """Read the selected task preset rather than duplicating its epoch count."""
    config_relpath = LAUNCHER_EPOCH_CONFIGS.get(launcher_task)
    if config_relpath is None:
        raise ValueError(f"No LTE epoch config registered for {launcher_task!r}")
    config_path = REPO_ROOT / "experiment_configs" / config_relpath
    return _config_path_default_epochs(config_path)


def _config_path_default_epochs(config_path: Path) -> int:
    """Read training.num_epochs from a standalone experiment config."""
    match = re.search(
        r"(?m)^training:\n(?:  [^\n]*\n)*?  num_epochs:\s*(\d+)\s*$",
        config_path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError(f"Could not find training.num_epochs in {config_path}")
    return int(match.group(1))


def _prompt_optimization_parameters(
    config_path: Path,
) -> tuple[int, int | None, float, str | None, int | None]:
    """Prompt for batch and learning-rate settings defined by a task config."""
    config = OmegaConf.load(config_path)
    batch_size = _prompt_int(
        "Training batch size", default=int(config.dataloader.batch_size), minimum=1
    )
    val_batch_size = None
    if "val_dataloader" in config and "batch_size" in config.val_dataloader:
        val_batch_size = batch_size
    learning_rate = _prompt_float(
        "Learning rate", default=float(config.optimizer.learning_rate), minimum=0.0
    )

    lr_scheduler = None
    lr_warmup_steps = None
    if "lr_scheduler" in config.training:
        lr_scheduler = _prompt_text(
            "Learning-rate scheduler", default=str(config.training.lr_scheduler)
        )
    if "lr_warmup_steps" in config.training:
        lr_warmup_steps = _prompt_int(
            "Learning-rate warm-up steps",
            default=int(config.training.lr_warmup_steps),
            minimum=0,
        )
    return batch_size, val_batch_size, learning_rate, lr_scheduler, lr_warmup_steps


def _lte_rgb_keys(launcher_task: str) -> tuple[str, list[str]]:
    """Resolve the primary and available LTE camera keys from the task preset."""
    config_relpath = LAUNCHER_EPOCH_CONFIGS.get(launcher_task)
    if config_relpath is None:
        raise ValueError(f"No LTE task config registered for {launcher_task!r}")
    config = OmegaConf.load(REPO_ROOT / "experiment_configs" / config_relpath)
    task = config.task
    primary_key = str(task.lte_temporal_rgb_key)
    rgb_keys = [
        str(key)
        for key, metadata in task.shape_meta.obs.items()
        if metadata.get("type", "low_dim") == "rgb"
    ]
    if primary_key not in rgb_keys:
        raise ValueError(
            f"Task {launcher_task!r} has non-RGB LTE key {primary_key!r}."
        )
    return primary_key, rgb_keys


def _training_command(
    task: str,
    output_dir: Path,
    seed: int,
    gpu: str,
    epochs: int,
    *,
    history_decoder_samples: int,
    temporal_latent_dim: int,
    temporal_encoder_hidden_dim: int,
    temporal_encoder_hidden_layers: int,
    history_decoder_hidden_dim: int,
    history_decoder_hidden_layers: int,
    temporal_embedding_cache: bool,
    cache_start_epoch: int,
    cache_warmup_epochs: int,
    cache_refresh_epochs: int,
    temporal_multi_image_fusion: bool,
    temporal_rgb_keys: list[str] | None,
    batch_size: int,
    val_batch_size: int | None,
    learning_rate: float,
    lr_scheduler: str | None,
    lr_warmup_steps: int | None,
    image_augmentation: bool,
) -> str:
    command = [
        str(TRAIN_SCRIPT),
        task,
        str(output_dir),
        f"training.seed={seed}",
        "training.device=cuda:0",
        f"training.num_epochs={epochs}",
        f"dataloader.batch_size={batch_size}",
        f"optimizer.learning_rate={learning_rate:g}",
        "+task.dataset.image_augmentation="
        f"{str(image_augmentation).lower()}",
        "policy.temporal_embedding_cache_enabled="
        f"{str(temporal_embedding_cache).lower()}",
        f"policy.temporal_embedding_cache_start_epoch={cache_start_epoch}",
        f"policy.temporal_embedding_cache_warmup_epochs={cache_warmup_epochs}",
        f"policy.temporal_embedding_cache_refresh_epochs={cache_refresh_epochs}",
        "policy.history_reconstruction.num_history_queries="
        f"{history_decoder_samples}",
        f"policy.temporal_latent_dim={temporal_latent_dim}",
        f"policy.temporal_hidden_dim={temporal_encoder_hidden_dim}",
        f"policy.temporal_num_hidden_layers={temporal_encoder_hidden_layers}",
        "policy.history_reconstruction.hidden_dim="
        f"{history_decoder_hidden_dim}",
        "policy.history_reconstruction.num_hidden_layers="
        f"{history_decoder_hidden_layers}",
        "policy.temporal_multi_image_fusion_enabled="
        f"{str(temporal_multi_image_fusion).lower()}",
    ]
    if val_batch_size is not None:
        command.append(f"val_dataloader.batch_size={val_batch_size}")
    if lr_scheduler is not None:
        command.append(f"training.lr_scheduler={lr_scheduler}")
    if lr_warmup_steps is not None:
        command.append(f"training.lr_warmup_steps={lr_warmup_steps}")
    if temporal_rgb_keys:
        command.append("policy.temporal_rgb_keys=[" + ",".join(temporal_rgb_keys) + "]")
    return f"{_environment_prefix(gpu)} {shlex.join(command)}"


def _ptp_training_command(
    config_path: Path,
    output_dir: Path,
    seed: int,
    gpu: str,
    epochs: int,
    batch_size: int,
    val_batch_size: int | None,
    learning_rate: float,
    lr_scheduler: str | None,
    lr_warmup_steps: int | None,
    image_augmentation: bool,
) -> str:
    """Build a PTP transformer launch using its direct Hydra config."""
    command = [
        sys.executable,
        str(TRAIN_PY),
        "--config-dir",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        f"hydra.run.dir={output_dir}",
        f"logging.name={output_dir.name}",
        "logging.id=null",
        f"training.seed={seed}",
        "training.device=cuda:0",
        f"training.num_epochs={epochs}",
        f"dataloader.batch_size={batch_size}",
        f"optimizer.learning_rate={learning_rate:g}",
        "+task.dataset.image_augmentation="
        f"{str(image_augmentation).lower()}",
    ]
    if val_batch_size is not None:
        command.append(f"val_dataloader.batch_size={val_batch_size}")
    if lr_scheduler is not None:
        command.append(f"training.lr_scheduler={lr_scheduler}")
    if lr_warmup_steps is not None:
        command.append(f"training.lr_warmup_steps={lr_warmup_steps}")
    return f"{_environment_prefix(gpu)} {shlex.join(command)}"


def _evaluation_command(
    checkpoint: Path,
    output_dir: Path,
    test_start_seed: int,
    gpu: str,
    n_test: int,
    *,
    record_zarr: bool = False,
) -> str:
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--checkpoint", str(checkpoint),
        "--output_dir", str(output_dir),
        "--device", "cuda:0",
        "--n_test", str(n_test),
        "--n_train", "0",
        "--n_test_vis", "0",
        "--test_start_seed", str(test_start_seed),
    ]
    if record_zarr:
        command.extend(("--zarr_path", str(output_dir / "rollouts.zarr")))
    return f"{_environment_prefix(gpu)} {shlex.join(command)}"


def _checkpoint_runner_target(checkpoint: Path) -> str:
    """Read the runner target without importing a checkpoint or PyTorch."""
    config_path = checkpoint.parents[1] / ".hydra" / "config.yaml"
    if config_path.is_file():
        return config_path.read_text(encoding="utf-8")
    return ""


def _checkpoint_supports_zarr_export(checkpoint: Path) -> bool:
    target = _checkpoint_runner_target(checkpoint)
    if target:
        return any(module in target for module in ZARR_RUNNER_MODULES)
    # Older or manually moved runs may not retain Hydra metadata. All tasks
    # managed by this LTE CLI use an image runner with Zarr export support.
    return any(task.replace("-", "_") in checkpoint.parents[1].name for task in TASKS)


def _start_training(with_evaluation: bool) -> None:
    decoder = _prompt_menu("Select decoder: ", ["transformer (default)", "unet"])
    decoder_name = "transformer" if decoder.startswith("transformer") else "unet"
    base_task, task, decoder_name = _task_name(decoder_name)
    gpu = _prompt_gpu()
    epochs = _prompt_int(
        "Training epochs", default=_config_default_epochs(task), minimum=1)
    config_path = REPO_ROOT / "experiment_configs" / LAUNCHER_EPOCH_CONFIGS[task]
    (
        batch_size,
        val_batch_size,
        learning_rate,
        lr_scheduler,
        lr_warmup_steps,
    ) = _prompt_optimization_parameters(config_path)
    history_decoder_samples = _prompt_int(
        "History samples per LTE decoder reconstruction", default=16, minimum=1
    )
    temporal_latent_dim = _prompt_int(
        "LTE temporal encoder latent dimension", default=64, minimum=1
    )
    temporal_encoder_hidden_dim = _prompt_int(
        "LTE temporal encoder hidden dimension", default=256, minimum=1
    )
    temporal_encoder_hidden_layers = _prompt_int(
        "LTE temporal encoder hidden layers", default=1, minimum=1
    )
    history_decoder_hidden_dim = _prompt_int(
        "LTE history decoder hidden dimension", default=256, minimum=1
    )
    history_decoder_hidden_layers = _prompt_int(
        "LTE history decoder hidden layers", default=1, minimum=1
    )
    temporal_multi_image_fusion = _prompt_bool(
        "Fuse multiple RGB cameras into LTE", default=False
    )
    primary_rgb_key, available_rgb_keys = _lte_rgb_keys(task)
    print(
        "LTE RGB cameras: "
        + ", ".join(available_rgb_keys)
        + f" (single-camera default: {primary_rgb_key})"
    )
    temporal_rgb_keys: list[str] = [primary_rgb_key]
    if temporal_multi_image_fusion:
        raw_keys = input(
            "LTE RGB keys (comma-separated; blank uses every RGB key): "
        ).strip()
        if raw_keys:
            temporal_rgb_keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        else:
            temporal_rgb_keys = available_rgb_keys
        if len(temporal_rgb_keys) < 2:
            raise ValueError("Multi-image LTE requires at least two RGB keys.")
        unknown_keys = set(temporal_rgb_keys) - set(available_rgb_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown LTE RGB key(s): " + ", ".join(sorted(unknown_keys))
            )
    temporal_embedding_cache = _prompt_bool(
        "Cache detached LTE ResNet embeddings", default=base_task == "lh-square"
    )
    image_augmentation = _prompt_bool(
        "Use ColorJitter image augmentation (can slow training)", default=False
    )
    cache_start_epoch = 5
    cache_warmup_epochs = 20
    cache_refresh_epochs = 5
    if temporal_embedding_cache:
        cache_start_epoch = _prompt_int(
            "Start caching detached LTE ResNet embeddings at epoch",
            default=cache_start_epoch,
            minimum=0,
        )
        cache_warmup_epochs = _prompt_int(
            "Refresh cache every epoch for the first N cached epochs",
            default=cache_warmup_epochs,
            minimum=0,
        )
        cache_refresh_epochs = _prompt_int(
            "Refresh cache every N epochs afterwards",
            default=cache_refresh_epochs,
            minimum=1,
        )
    # The suffix is assigned only while planning a new launch. Existing and
    # active LDP training or inference directories are never renamed.
    runs = _planned_runs(
        base_task,
        decoder_name,
        temporal_embedding_cache=temporal_embedding_cache,
        cache_start_epoch=cache_start_epoch,
        cache_warmup_epochs=cache_warmup_epochs,
        cache_refresh_epochs=cache_refresh_epochs,
    )
    record_zarr = True
    if with_evaluation:
        n_test = _prompt_int("Evaluation test episodes", default=200, minimum=1)
        test_start_seed = _prompt_int(
            "Evaluation test-start seed", default=1000, minimum=0
        )
    else:
        n_test = 0
        test_start_seed = 0

    commands = []
    for seed, output_dir, inference_dir in runs:
        train = _training_command(
            task,
            output_dir,
            seed,
            gpu,
            epochs,
            history_decoder_samples=history_decoder_samples,
            temporal_latent_dim=temporal_latent_dim,
            temporal_encoder_hidden_dim=temporal_encoder_hidden_dim,
            temporal_encoder_hidden_layers=temporal_encoder_hidden_layers,
            history_decoder_hidden_dim=history_decoder_hidden_dim,
            history_decoder_hidden_layers=history_decoder_hidden_layers,
            temporal_embedding_cache=temporal_embedding_cache,
            cache_start_epoch=cache_start_epoch,
            cache_warmup_epochs=cache_warmup_epochs,
            cache_refresh_epochs=cache_refresh_epochs,
            temporal_multi_image_fusion=temporal_multi_image_fusion,
            temporal_rgb_keys=temporal_rgb_keys,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            lr_warmup_steps=lr_warmup_steps,
            image_augmentation=image_augmentation,
        )
        if with_evaluation:
            checkpoint = output_dir / "checkpoints" / "latest.ckpt"
            evaluate = _evaluation_command(
                checkpoint, inference_dir, test_start_seed, gpu, n_test,
                record_zarr=record_zarr,
            )
            commands.append(f"{train} && {evaluate}")
        else:
            commands.append(train)
    label = f"{'train-eval' if with_evaluation else 'train'}-{task}-x{len(runs)}"
    _start_screen_session(_make_screen_session(label), " && ".join(commands))


def _start_ptp_training(with_evaluation: bool) -> None:
    """Interactively start PTP training, optionally followed by evaluation."""
    task = _prompt_menu("Select PTP task: ", list(PTP_TASK_CONFIGS))
    config_path = PTP_TASK_CONFIGS[task]
    gpu = _prompt_gpu()
    epochs = _prompt_int(
        "Training epochs", default=_config_path_default_epochs(config_path), minimum=1
    )
    (
        batch_size,
        val_batch_size,
        learning_rate,
        lr_scheduler,
        lr_warmup_steps,
    ) = _prompt_optimization_parameters(config_path)
    image_augmentation = _prompt_bool(
        "Use ColorJitter image augmentation (can slow training)", default=False
    )
    runs = _planned_ptp_runs(task)
    if with_evaluation:
        n_test = _prompt_int("Evaluation test episodes", default=200, minimum=1)
        test_start_seed = _prompt_int(
            "Evaluation test-start seed", default=1000, minimum=0
        )
    else:
        n_test = 0
        test_start_seed = 0

    commands = []
    for seed, output_dir, inference_dir in runs:
        train = _ptp_training_command(
            config_path,
            output_dir,
            seed,
            gpu,
            epochs,
            batch_size,
            val_batch_size,
            learning_rate,
            lr_scheduler,
            lr_warmup_steps,
            image_augmentation,
        )
        if with_evaluation:
            evaluate = _evaluation_command(
                output_dir / "checkpoints" / "latest.ckpt",
                inference_dir,
                test_start_seed,
                gpu,
                n_test,
                record_zarr=True,
            )
            commands.append(f"{train} && {evaluate}")
        else:
            commands.append(train)
    label = f"ptp-{'train-eval' if with_evaluation else 'train'}-{task}-x{len(runs)}"
    _start_screen_session(_make_screen_session(label), " && ".join(commands))


def _start_training_flow(with_evaluation: bool) -> None:
    """Select between the LTE and PTP training workflows."""
    method = _prompt_menu("Select training method: ", ["LTE-IMG-NoT", "PTP"])
    if method == "PTP":
        _start_ptp_training(with_evaluation)
    else:
        _start_training(with_evaluation)


def _available_evaluation_runs() -> list[Path]:
    """Return training directories that have at least one selectable checkpoint."""
    return sorted(
        (
            run_dir
            for run_dir in _training_run_directories()
            if any((run_dir / "checkpoints").glob("*.ckpt"))
        ),
        key=lambda path: str(path),
    )


def _prompt_evaluation_run() -> Path:
    runs = _available_evaluation_runs()
    if not runs:
        raise RuntimeError("No training runs with checkpoints found in either output root.")
    labels = [str(run.relative_to(REPO_ROOT)) for run in runs]
    selected = _prompt_menu("Select training run: ", labels)
    return runs[labels.index(selected)]


def _prompt_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        (path for path in (run_dir / "checkpoints").glob("*.ckpt") if path.is_file()),
        key=lambda path: (path.name != "latest.ckpt", -path.stat().st_mtime, path.name),
    )
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir.relative_to(REPO_ROOT)}.")
    selected = _prompt_menu("Select checkpoint: ", [path.name for path in checkpoints])
    return next(path for path in checkpoints if path.name == selected)


def _evaluation_output_run_name(checkpoint: Path) -> str:
    """Keep timestamped run names, or assign one when evaluating a legacy run."""
    checkpoint_run_name = checkpoint.parents[1].name
    if re.fullmatch(
        r".+_(?:lte_(?:transformer|unet)|ptp)_\d{8}_\d{6}(?:-r\d+)?",
        checkpoint_run_name,
    ):
        return checkpoint_run_name
    legacy_match = re.fullmatch(
        r"(?P<task>.+)_lte_(?P<decoder>transformer|unet)(?:_.+)?",
        checkpoint_run_name,
    )
    if legacy_match:
        return _run_name(
            legacy_match.group("task"), legacy_match.group("decoder"), seed=0
        )
    raise ValueError(
        "Select a timestamped LTE/PTP run or a legacy LTE run "
        "(<env>_lte_<decoder>_...)."
    )


def _start_evaluation() -> None:
    run_dir = _prompt_evaluation_run()
    checkpoint = _prompt_checkpoint(run_dir)
    n_test = _prompt_int("Evaluation episodes", default=30, minimum=1)
    gpu = _prompt_gpu()
    test_start_seed = _prompt_int("Evaluation start seed", default=1000, minimum=0)
    record_zarr = _checkpoint_supports_zarr_export(checkpoint)
    if record_zarr:
        print("Image checkpoint detected: evaluation will save rollouts.zarr.")
    run_name = _evaluation_output_run_name(checkpoint)
    output_dir = _next_available_path(REPO_ROOT / "data" / "inference" / run_name)
    command = _evaluation_command(
        checkpoint, output_dir, test_start_seed, gpu, n_test,
        record_zarr=record_zarr,
    )
    _start_screen_session(
        _make_screen_session(f"eval-{run_name}-{checkpoint.stem}"), command,
    )


def _tail_lines(path: Path, count: int = 6) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-count:]


def _training_epoch_limit(run_dir: Path) -> str:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return "?"
    match = re.search(
        r"(?m)^training:\n(?:  [^\n]*\n)*?  num_epochs:\s*(\d+)\s*$",
        config_path.read_text(encoding="utf-8", errors="replace"),
    )
    return match.group(1) if match else "?"


def _latest_epoch(metrics_path: Path) -> int | None:
    for line in reversed(_tail_lines(metrics_path, count=100)):
        try:
            epoch = json.loads(line).get("epoch")
        except json.JSONDecodeError:
            continue
        if epoch is not None:
            return int(epoch) + 1  # logs use a zero-indexed epoch counter
    return None


def _inference_progress_entry(dataset_dir: Path) -> tuple[str, str, Path]:
    """Read evaluation progress from a dataset directory, never screen logs."""
    eval_log = dataset_dir / "eval_log.json"
    if eval_log.is_file():
        try:
            result = json.loads(eval_log.read_text(encoding="utf-8"))
            completed = sum(
                key.startswith(("test/sim_max_reward_", "train/sim_max_reward_"))
                for key in result
            )
        except (OSError, json.JSONDecodeError):
            try:
                completed = len(_evaluation_episode_scores(dataset_dir))
            except RuntimeError:
                return "failed", "invalid log", eval_log
            return "complete", f"{completed}/{completed}", eval_log
        if completed:
            return "complete", f"{completed}/{completed}", eval_log
        return "failed", "0/?", eval_log

    # Zarr's episode_ends array grows by one entry only after a full episode is
    # recorded, which makes its metadata a robust on-disk progress source.
    episode_ends_metadata = (
        dataset_dir / "rollouts.zarr" / "meta" / "episode_ends" / ".zarray"
    )
    if episode_ends_metadata.is_file():
        try:
            metadata = json.loads(episode_ends_metadata.read_text(encoding="utf-8"))
            completed = int(metadata["shape"][0])
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError):
            completed = 0
        state = _progress_state(completed, 0, episode_ends_metadata)
        return state, f"{completed}/?", episode_ends_metadata

    state = _progress_state(0, 0, dataset_dir)
    return state, "0/?", dataset_dir


def _progress_color(text: str, state: str) -> str:
    return f"{ANSI_COLORS[state]}{text}{ANSI_RESET}"


def _progress_state(completed: int, total: int, source: Path) -> str:
    if total > 0 and completed >= total:
        return "complete"
    try:
        age = time.time() - source.stat().st_mtime
    except OSError:
        return "stale"
    return "active" if age <= STALE_PROGRESS_SECONDS else "stale"


def _age_label(source: Path) -> str:
    try:
        seconds = max(0, int(time.time() - source.stat().st_mtime))
    except OSError:
        return "unknown"
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def _training_runtime_label(run_dir: Path, metrics_path: Path) -> str:
    """Return elapsed wall-clock time from Hydra setup to the latest log entry."""
    config_path = run_dir / ".hydra" / "config.yaml"
    try:
        seconds = max(0, int(metrics_path.stat().st_mtime - config_path.stat().st_mtime))
    except OSError:
        return "unknown"
    return f"{seconds / 3600:.1f}h"


def _evaluation_runtime_label(dataset_dir: Path, progress_source: Path) -> str:
    """Estimate evaluation wall time from its first on-disk artifact to latest progress."""
    try:
        first_artifact_time = min(
            path.stat().st_mtime
            for path in dataset_dir.rglob("*")
            if path.is_file()
        )
        latest_progress_time = progress_source.stat().st_mtime
    except (OSError, ValueError):
        return "unknown"
    return f"{max(0.0, latest_progress_time - first_artifact_time) / 3600:.1f}h"


def _disk_usage_label(path: Path) -> str:
    """Return filesystem space used by one run directory."""
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)], capture_output=True, text=True, check=False
        )
    except OSError:
        return "?"
    if result.returncode or not result.stdout.strip():
        return "?"
    return result.stdout.split(maxsplit=1)[0]


def _run_note_label(run_dir: Path) -> str:
    """Return the first line of a run's optional note."""
    try:
        note = (run_dir / RUN_NOTE_FILENAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    if not note:
        return ""
    return note[0].strip()


def _note_column_width(notes: list[str]) -> int:
    """Fit the shared note column to its content, with a sensible hard cap."""
    return min(RUN_NOTE_MAX_LENGTH, max(len("Note"), *(len(note) for note in notes)))


def _truncate_note(note: str, width: int) -> str:
    if len(note) <= width:
        return note
    return f"{note[:width - 3]}..."


def _run_sort_key(run_name: str) -> tuple[str, str, str]:
    """Sort runs by environment, then their timestamped directory suffix."""
    normalized_name = run_name.lower()
    matching_tasks = [
        task for task in TASKS
        if normalized_name.startswith(task.replace("-", "_"))
    ]
    environment = max(matching_tasks, key=len) if matching_tasks else normalized_name
    timestamp_match = re.search(r"_(\d{8}_\d{6})(?:-r\d+)?$", run_name)
    # Legacy names without a timestamp sort after timestamped runs of the same
    # environment, while retaining a stable name-based tie-breaker.
    timestamp = timestamp_match.group(1) if timestamp_match else "99999999_999999"
    return environment, timestamp, normalized_name


def _print_progress_table(
    rows: list[tuple[str, str, str, str, str, str]], progress_label: str,
) -> None:
    if not rows:
        print("  none")
        return
    run_width = max(len("Run"), *(len(run) for run, _, _, _, _, _ in rows))
    progress_width = max(
        len(progress_label),
        *(len(ANSI_ESCAPE_RE.sub("", progress)) for _, progress, _, _, _, _ in rows),
    )
    disk_width = max(len("Disk"), *(len(disk) for _, _, disk, _, _, _ in rows))
    runtime_width = max(len("Runtime"), *(len(runtime) for _, _, _, runtime, _, _ in rows))
    note_width = _note_column_width([note for _, _, _, _, note, _ in rows])
    print(
        f"  {'Run':<{run_width}} | {progress_label:<{progress_width}} | "
        f"{'Disk':<{disk_width}} | {'Runtime':<{runtime_width}} | "
        f"{'Note':<{note_width}} | Updated"
    )
    print(
        f"  {'-' * run_width}-|-{'-' * progress_width}-|-{'-' * disk_width}-|-"
        f"{'-' * runtime_width}-|-"
        f"{'-' * note_width}-|--------"
    )
    previous_environment = None
    for run, progress, disk, runtime, note, updated in rows:
        environment = _run_sort_key(run)[0]
        if previous_environment is not None and environment != previous_environment:
            print(
                f"  {'-' * run_width}-|-{'-' * progress_width}-|-{'-' * disk_width}-|-"
                f"{'-' * runtime_width}-|-"
                f"{'-' * note_width}-|--------"
            )
        visible_progress = len(ANSI_ESCAPE_RE.sub("", progress))
        print(
            f"  {run:<{run_width}} | {progress}{' ' * (progress_width - visible_progress)} | "
            f"{disk:<{disk_width}} | {runtime:<{runtime_width}} | "
            f"{_truncate_note(note, note_width):<{note_width}} | {updated}")
        previous_environment = environment


def _print_training_progress_table(rows: list[tuple[str, str, str, str, str, str]]) -> None:
    if not rows:
        print("  none")
        return
    run_width = max(len("Run"), *(len(run) for run, _, _, _, _, _ in rows))
    epoch_width = max(
        len("Epoch"),
        *(len(ANSI_ESCAPE_RE.sub("", epoch)) for _, epoch, _, _, _, _ in rows),
    )
    disk_width = max(len("Disk"), *(len(disk) for _, _, disk, _, _, _ in rows))
    runtime_width = max(len("Runtime"), *(len(runtime) for _, _, _, runtime, _, _ in rows))
    note_width = _note_column_width([note for _, _, _, _, note, _ in rows])
    print(
        f"  {'Run':<{run_width}} | {'Epoch':<{epoch_width}} | "
        f"{'Disk':<{disk_width}} | {'Runtime':<{runtime_width}} | "
        f"{'Note':<{note_width}} | Updated"
    )
    print(
        f"  {'-' * run_width}-|-{'-' * epoch_width}-|-"
        f"{'-' * disk_width}-|-{'-' * runtime_width}-|-"
        f"{'-' * note_width}-|--------"
    )
    previous_environment = None
    for run, epoch, disk, runtime, note, updated in rows:
        environment = _run_sort_key(run)[0]
        if previous_environment is not None and environment != previous_environment:
            print(
                f"  {'-' * run_width}-|-{'-' * epoch_width}-|-"
                f"{'-' * disk_width}-|-{'-' * runtime_width}-|-"
                f"{'-' * note_width}-|--------"
            )
        visible_epoch = len(ANSI_ESCAPE_RE.sub("", epoch))
        print(
            f"  {run:<{run_width}} | {epoch}{' ' * (epoch_width - visible_epoch)} | "
            f"{disk:<{disk_width}} | {runtime:<{runtime_width}} | "
            f"{_truncate_note(note, note_width):<{note_width}} | {updated}"
        )
        previous_environment = environment


def _show_progress() -> None:
    """Print compact, user-facing train and evaluation progress."""
    train_logs = sorted(
        (
            run_dir / "logs.json.txt"
            for run_dir in _training_run_directories()
            if (run_dir / "logs.json.txt").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    training_entries = []
    for path in train_logs:
        epoch = _latest_epoch(path)
        total = _training_epoch_limit(path.parent)
        current = epoch if epoch is not None else 0
        total_value = int(total) if total.isdigit() else 0
        state = _progress_state(current, total_value, path)
        progress = _progress_color(f"{current}/{total}", state)
        training_entries.append((
            state,
            path.parent.name,
            progress,
            _disk_usage_label(path.parent),
            _training_runtime_label(path.parent, path),
            _run_note_label(path.parent),
            _age_label(path),
        ))
    print("\nTraining")
    _print_training_progress_table([
        (run, progress, disk, runtime, note, updated)
        for _, run, progress, disk, runtime, note, updated in sorted(
            training_entries, key=lambda entry: _run_sort_key(entry[1]))
    ])

    inference_root = REPO_ROOT / "data" / "inference"
    evaluation_entries = []
    if inference_root.is_dir():
        for dataset_dir in inference_root.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.is_symlink():
                continue
            state, progress, source = _inference_progress_entry(dataset_dir)
            evaluation_entries.append((
                state,
                dataset_dir.name,
                _progress_color(progress, state),
                _disk_usage_label(dataset_dir),
                _evaluation_runtime_label(dataset_dir, source),
                _run_note_label(dataset_dir),
                _age_label(source),
                source,
            ))
    print("\nEvaluations")
    _print_progress_table([
        (run, progress, disk, runtime, note, updated)
        for _, run, progress, disk, runtime, note, updated, _ in sorted(
            evaluation_entries, key=lambda entry: _run_sort_key(entry[1]))
    ], "Episode")


def _add_run_note() -> None:
    """Store one short note beside a selected training or inference run."""
    run_type = _prompt_menu("Add note to: ", ["training run", "inference run", "back"])
    if run_type == "back":
        return
    root = REPO_ROOT / "data" / ("outputs" if run_type == "training run" else "inference")
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    ) if root.is_dir() else []
    if not candidates:
        print(f"No {run_type}s found.")
        return
    selected = _prompt_menu("Select run: ", [path.name for path in candidates])
    run_dir = next(path for path in candidates if path.name == selected)
    current_note = _run_note_label(run_dir)
    if current_note:
        print(f"Current note: {current_note}")
    while True:
        note = input(f"Note (max {RUN_NOTE_MAX_LENGTH} characters; blank clears): ").strip()
        if len(note) <= RUN_NOTE_MAX_LENGTH:
            break
        print(f"Enter at most {RUN_NOTE_MAX_LENGTH} characters.")
    note_path = run_dir / RUN_NOTE_FILENAME
    if note:
        note_path.write_text(f"{note}\n", encoding="utf-8")
        print(f"Saved note for {run_dir.name}.")
    elif note_path.exists():
        note_path.unlink()
        print(f"Cleared note for {run_dir.name}.")
    else:
        print("No note set.")


def _evaluation_task(eval_dir: Path) -> str | None:
    """Return the configured task name encoded in an evaluation directory."""
    name = eval_dir.name.lower()
    matches = [
        task for task in TASKS
        if name.startswith(task.replace("-", "_"))
    ]
    return max(matches, key=len) if matches else None


def _evaluation_episode_scores(eval_dir: Path) -> list[float]:
    """Read per-test-episode scores recorded by ``eval.py``."""
    eval_log_path = eval_dir / "eval_log.json"
    try:
        eval_log = json.loads(eval_log_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"Missing {eval_log_path.relative_to(REPO_ROOT)}.") from None
    except json.JSONDecodeError as exc:
        recovered_scores = _recover_scores_from_launcher_log(eval_dir)
        if recovered_scores:
            return recovered_scores
        raise RuntimeError(
            f"Invalid JSON in {eval_log_path.relative_to(REPO_ROOT)}: {exc}"
        ) from exc

    score_prefix = "test/sim_max_reward_"
    scores = [
        float(value) for key, value in eval_log.items()
        if key.startswith(score_prefix)
    ]
    if not scores:
        raise RuntimeError(
            f"{eval_log_path.relative_to(REPO_ROOT)} has no {score_prefix} entries."
        )
    return scores


def _recover_scores_from_launcher_log(eval_dir: Path) -> list[float] | None:
    """Recover a legacy truncated result file from its matching launcher log.

    Older ``eval.py`` versions could finish every rollout then fail while
    serialising NumPy scalar rewards. The complete result dictionary was
    already printed to the corresponding launcher log immediately before that
    failed write. Restrict recovery to the task and recorded mean score so an
    unrelated run cannot be mistaken for this evaluation.
    """
    task = _evaluation_task(eval_dir)
    if task is None or not CLI_LOG_DIR.is_dir():
        return None
    try:
        partial_log = (eval_dir / "eval_log.json").read_text(encoding="utf-8")
    except OSError:
        return None
    mean_match = re.search(r'"test/mean_score"\s*:\s*([-+0-9.eE]+)', partial_log)
    if mean_match is None:
        return None
    expected_mean = float(mean_match.group(1))
    task_label = task.replace("_", "-")
    score_prefix = "test/sim_max_reward_"
    candidates = sorted(
        (path for path in CLI_LOG_DIR.glob("*.log") if task_label in path.name),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if not line.startswith("{") or score_prefix not in line:
                continue
            try:
                result = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(result, dict) or "test/mean_score" not in result:
                continue
            scores = [
                float(value) for key, value in result.items()
                if key.startswith(score_prefix)
            ]
            if scores and math.isclose(
                float(result["test/mean_score"]), expected_mean,
                rel_tol=0.0, abs_tol=1e-9,
            ):
                return scores
    return None


def _check_success_rates() -> None:
    """Report saved-evaluation success rates for every available environment."""
    inference_root = REPO_ROOT / "data" / "inference"
    if not inference_root.is_dir():
        print("No inference directory exists.")
        return
    rows = []
    for task in TASKS:
        evaluation_dirs = [
            path for path in inference_root.iterdir()
            if path.is_dir() and not path.is_symlink() and _evaluation_task(path) == task
        ]
        threshold = SUCCESS_REWARD_THRESHOLDS[task]
        for evaluation_dir in evaluation_dirs:
            try:
                scores = _evaluation_episode_scores(evaluation_dir)
            except RuntimeError:
                rows.append((
                    task, evaluation_dir.name, "skipped", "", "", "",
                    _run_note_label(evaluation_dir),
                ))
                continue
            successes = sum(score >= threshold - 1e-8 for score in scores)
            rows.append((
                task,
                evaluation_dir.name,
                f"{successes}/{len(scores)}",
                f"{successes / len(scores):.1%}",
                f"{threshold:g}",
                f"{sum(scores) / len(scores):.2f}",
                _run_note_label(evaluation_dir),
            ))
    if not rows:
        print("No evaluation directories found.")
        return

    rows.sort(key=lambda row: (row[0], _run_sort_key(row[1])))
    print("\nSuccess rates")
    run_width = max(len("Run"), *(len(row[1]) for row in rows))
    result_width = max(len("Successful"), *(len(row[2]) for row in rows))
    rate_width = max(len("Rate"), *(len(row[3]) for row in rows))
    threshold_width = max(len("Thresh."), *(len(row[4]) for row in rows))
    mean_width = max(len("Mean sc."), *(len(row[5]) for row in rows))
    note_width = _note_column_width([row[6] for row in rows])
    print(
        f"  {'Run':<{run_width}} | {'Successful':>{result_width}} | "
        f"{'Rate':>{rate_width}} | {'Thresh.':>{threshold_width}} | "
        f"{'Mean sc.':>{mean_width}} | {'Note':<{note_width}}"
    )
    print(
        f"  {'-' * run_width}-|-{'-' * result_width}-|-{'-' * rate_width}-|-"
        f"{'-' * threshold_width}-|-{'-' * mean_width}-|-{'-' * note_width}"
    )
    previous_environment = None
    for environment, run, successful, rate, threshold_label, mean_score, note in rows:
        if previous_environment is not None and environment != previous_environment:
            print(
                f"  {'-' * run_width}-|-{'-' * result_width}-|-{'-' * rate_width}-|-"
                f"{'-' * threshold_width}-|-{'-' * mean_width}-|-{'-' * note_width}"
            )
        print(
            f"  {run:<{run_width}} | {successful:>{result_width}} | "
            f"{rate:>{rate_width}} | {threshold_label:>{threshold_width}} | "
            f"{mean_score:>{mean_width}} | {_truncate_note(note, note_width):<{note_width}}"
        )
        previous_environment = environment

    print("\nLegend")
    print("  No history: 2-frame context; no PTP")
    print("  Long hist. no-PTP: 16-frame context; no PTP")
    print("  Long hist. PTP: 16-frame context; PTP enabled")


def _delete_run_directory() -> None:
    """Interactively remove one complete training-run output directory."""
    display_root = REPO_ROOT / "data" / "outputs"
    output_root = display_root.resolve()
    if not output_root.is_dir():
        print("No training output directory exists.")
        return
    candidates = sorted(
        (
            path for path in output_root.iterdir()
            if path.is_dir() and not path.is_symlink() and path.resolve().parent == output_root
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        print("No training output directories found.")
        return
    entries = []
    for run_dir in candidates:
        metrics_path = run_dir / "logs.json.txt"
        epoch = _latest_epoch(metrics_path)
        total = _training_epoch_limit(run_dir)
        current = epoch if epoch is not None else 0
        total_value = int(total) if total.isdigit() else 0
        state = _progress_state(current, total_value, metrics_path)
        entries.append((
            state,
            run_dir,
            _progress_color(f"{current}/{total}", state),
        ))
    # Keep active runs at the bottom, as in the progress view.
    entries.sort(key=lambda entry: entry[0] == "active")
    print("\nTraining run directories")
    for index, (_, run_dir, epoch) in enumerate(entries, start=1):
        checkpoint_state = "checkpoints" if (run_dir / "checkpoints").is_dir() else "no checkpoints"
        print(f"  {index}. {run_dir.name} | Epoch {epoch} | {checkpoint_state}")
    while True:
        answer = input("Select training run directory to delete: ").strip()
        try:
            run_dir = entries[int(answer) - 1][1]
            break
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(entries)}.")
    target = run_dir
    selected_name = run_dir.name
    # ``data`` can be a symlink to shared storage, so the resolved deletion
    # target may be outside the checkout. Display its stable logical path
    # instead of calling ``relative_to`` on the resolved path.
    target_description = display_root / selected_name
    success_message = f"Deleted training run directory {target_description}."
    if not _prompt_bool(
        f"Permanently delete {target_description}", default=False
    ):
        print("Deletion cancelled.")
        return
    shutil.rmtree(target)
    print(success_message)


def _delete_inference_directory() -> None:
    """Interactively remove one direct, non-symlinked inference directory."""
    display_root = REPO_ROOT / "data" / "inference"
    inference_root = display_root.resolve()
    if not inference_root.is_dir():
        print("No inference directory exists.")
        return
    candidates = sorted(
        (
            path for path in inference_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.resolve().parent == inference_root
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        print("No inference directories found.")
        return

    print("\nInference directories")
    for index, inference_dir in enumerate(candidates, start=1):
        print(f"  {index}. {inference_dir.name}")
    while True:
        answer = input("Select inference directory to delete: ").strip()
        try:
            inference_dir = candidates[int(answer) - 1]
            break
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(candidates)}.")

    selected_name = inference_dir.name
    target_description = display_root / selected_name
    if not _prompt_bool(
        f"Permanently delete {target_description}", default=False
    ):
        print("Deletion cancelled.")
        return
    shutil.rmtree(inference_dir)
    print(f"Deleted inference directory {target_description}.")


def _human_size(size_bytes: int) -> str:
    """Format a byte count compactly for CLI tables."""
    size = float(size_bytes)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or suffix == "TiB":
            return f"{size:.1f} {suffix}" if suffix != "B" else f"{size:.0f} B"
        size /= 1024.0
    raise AssertionError("unreachable")


def _compress_inference_images_to_videos() -> None:
    """Export inference Zarr image arrays as H.265 videos, with optional cleanup."""
    from tools.inference_images_to_video import (
        compress_zarr_images_to_videos,
        disk_usage_bytes,
        rollout_zarr_paths,
    )

    inference_root = REPO_ROOT / "data" / "inference"
    if not inference_root.is_dir():
        print("No inference directory exists.")
        return
    entries = []
    for dataset_path in sorted(inference_root.iterdir(), key=lambda path: path.name):
        if not dataset_path.is_dir() or dataset_path.is_symlink():
            continue
        zarr_paths = rollout_zarr_paths(dataset_path)
        if zarr_paths:
            entries.append((dataset_path, zarr_paths, disk_usage_bytes(dataset_path)))
    if not entries:
        print("No inference datasets with image rollout Zarr stores were found.")
        return

    print("\nInference datasets with image rollouts")
    name_width = max(len("Dataset"), *(len(path.name) for path, _, _ in entries))
    print(f"  {'Dataset':<{name_width}} | Space used | Image Zarr")
    print(f"  {'-' * name_width}-|------------|-----------")
    for index, (dataset_path, zarr_paths, size_bytes) in enumerate(entries, start=1):
        names = ", ".join(path.name for path in zarr_paths)
        print(f"  {index}. {dataset_path.name:<{name_width}} | {_human_size(size_bytes):>10} | {names}")
    while True:
        answer = input("Select dataset to compress: ").strip()
        try:
            dataset_path, zarr_paths, before_bytes = entries[int(answer) - 1]
            break
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(entries)}.")

    zarr_path = zarr_paths[0]
    if len(zarr_paths) > 1:
        zarr_path = _prompt_menu(
            "Select image rollout Zarr: ", [str(path) for path in zarr_paths]
        )
        zarr_path = Path(zarr_path)
    fps = _prompt_int("Video FPS", default=10, minimum=1)
    delete_source_images = _prompt_bool(
        "Delete original image arrays after verified video export", default=True
    )

    summary = compress_zarr_images_to_videos(
        dataset_path,
        zarr_path,
        fps=fps,
        delete_source_images=delete_source_images,
    )
    after_bytes = disk_usage_bytes(dataset_path)
    print(f"\nVideos written to {summary.output_dir.relative_to(REPO_ROOT)}")
    print(f"Space used: {_human_size(before_bytes)} -> {_human_size(after_bytes)}")
    if summary.source_images_deleted:
        print(f"Removed source arrays: {', '.join(summary.image_keys)}")
    else:
        print("Source image arrays were kept.")


def _show_disk_usage() -> None:
    """Show capacity and free space for the project storage mounts."""
    subprocess.run(["df", "-h", "/", "/mnt/data", "/mnt/wdblack"], check=False)


def _show_gpu_status() -> None:
    """Show NVIDIA GPU utilisation in this terminal."""
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi was not found on PATH.")


def _show_gpu_memory_by_process() -> None:
    """Show GPU-memory consumers, sorted largest first, with full commands."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("nvidia-smi was not found on PATH.")
        return
    if result.returncode:
        print("nvidia-smi could not query GPU compute processes.")
        return

    processes = []
    for line in result.stdout.splitlines():
        try:
            pid_text, memory_text = (part.strip() for part in line.rsplit(",", 1))
            pid = int(pid_text)
            memory_mib = int(memory_text)
        except ValueError:
            continue
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            command_label = command.replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip() or "<command unavailable>"
        except OSError:
            command_label = "<process exited>"
        processes.append((memory_mib, pid, command_label))

    if not processes:
        print("No GPU compute processes found.")
        return
    print("\nGPU memory by process")
    for index, (memory_mib, pid, command) in enumerate(
        sorted(processes, reverse=True)
    ):
        if index:
            print()
        print(f"  {memory_mib:>6} MiB | PID {pid:<7} | {command}")


def _wandb_run_folder_artifacts() -> tuple[object, list[tuple[object, object]]]:
    """Select W&B runs that have a completed, downloadable run-folder artifact."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required; run the experiment CLI in the LDP Conda environment."
        ) from exc

    api = wandb.Api()
    entity = api.default_entity
    project = _prompt_text("W&B project", "ldp_temporal_diffusion_policy")

    def find_in_project(project_name: str) -> list[tuple[object, object]]:
        try:
            runs = api.runs(f"{entity}/{project_name}", per_page=100)
        except Exception:
            return []
        found = []
        for run in runs:
            try:
                artifacts = list(run.logged_artifacts())
            except Exception:
                continue
            for artifact in artifacts:
                if artifact.type == "run-folder":
                    found.append((run, artifact))
        return found

    candidates = find_in_project(project)
    if candidates:
        return wandb, candidates

    print(
        f"No run-folder artifacts found in {entity}/{project}; "
        "searching your other W&B projects."
    )
    try:
        project_names = sorted(
            project_item.name for project_item in api.projects(entity)
            if project_item.name != project
        )
    except Exception as exc:
        raise RuntimeError(f"Could not list W&B projects for {entity}: {exc}") from exc
    for project_name in project_names:
        candidates.extend(find_in_project(project_name))
    if not candidates:
        raise RuntimeError(
            f"No W&B run-folder artifacts found under {entity}."
        )
    return wandb, candidates


def _regular_file_inventory(root: Path) -> tuple[int, int]:
    """Return count and bytes for artifact files, excluding non-portable links."""
    files = [path for path in root.rglob("*") if path.is_file() and not path.is_symlink()]
    return len(files), sum(path.stat().st_size for path in files)


def _restore_wandb_run_folder() -> None:
    """Restore a W&B run folder externally, then remove its remote checkpoints."""
    _, candidates = _wandb_run_folder_artifacts()
    labels = []
    for run, artifact in candidates:
        output_name = artifact.metadata.get("output_dir_name", artifact.name)
        labels.append(f"{run.name} [{run.id}]  ->  {output_name}")
    selected_label = _prompt_menu("Select W&B run folder: ", labels)
    run, folder_artifact = candidates[labels.index(selected_label)]

    output_name = str(
        folder_artifact.metadata.get("output_dir_name", folder_artifact.name)
    )
    output_name = output_name.split(":", 1)[0]
    destination = _next_available_path(EXTERNAL_OUTPUT_ROOT / output_name)
    print(f"Downloading {run.name} to {destination.relative_to(REPO_ROOT)}")
    try:
        folder_artifact.download(root=str(destination))
    except Exception as exc:
        raise RuntimeError(
            "W&B folder download failed; remote artifacts were not deleted: " f"{exc}"
        ) from exc

    actual_file_count, actual_byte_count = _regular_file_inventory(destination)
    expected_file_count = folder_artifact.metadata.get("file_count")
    expected_byte_count = folder_artifact.metadata.get("byte_count")
    if expected_file_count is not None and actual_file_count != int(expected_file_count):
        raise RuntimeError(
            f"Downloaded {actual_file_count} files, expected {expected_file_count}; "
            "remote artifacts were not deleted."
        )
    if expected_byte_count is not None and actual_byte_count != int(expected_byte_count):
        raise RuntimeError(
            f"Downloaded {_human_size(actual_byte_count)}, expected "
            f"{_human_size(int(expected_byte_count))}; remote artifacts were not deleted."
        )
    checkpoints = sorted((destination / "checkpoints").glob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError(
            "Downloaded folder contains no checkpoints; remote artifacts were not deleted."
        )

    try:
        run_artifacts = list(run.logged_artifacts())
    except Exception as exc:
        raise RuntimeError(
            f"Could not enumerate remote artifacts; none were deleted: {exc}"
        ) from exc
    checkpoint_artifacts = [
        artifact for artifact in run_artifacts
        if artifact.type in {"model", "run-folder"}
    ]
    if not checkpoint_artifacts:
        raise RuntimeError("No remote checkpoint artifacts found to delete.")
    artifact_names = ", ".join(artifact.name for artifact in checkpoint_artifacts)
    if not _prompt_bool(
        f"Delete W&B checkpoint artifacts after verified download ({artifact_names})",
        default=True,
    ):
        print("Downloaded folder kept; W&B checkpoint artifacts were not deleted.")
        return

    for artifact in checkpoint_artifacts:
        artifact.delete(delete_aliases=True)
    print(
        f"Restored {destination.relative_to(REPO_ROOT)} with {len(checkpoints)} "
        f"checkpoints, then deleted {len(checkpoint_artifacts)} W&B checkpoint artifact(s)."
    )


def _saved_training_runs() -> list[Path]:
    """Find train outputs retaining either final or legacy Hydra configs."""
    runs = {
        run_dir
        for run_dir in _training_run_directories()
        if (run_dir / "final_resolved_config.yaml").is_file()
        or (run_dir / ".hydra" / "config.yaml").is_file()
    }
    return sorted(
        runs,
        key=lambda path: path.name,
    )


def _eval_run_source_name(eval_dir: Path) -> str | None:
    """Infer the train-output name used by this CLI's evaluation naming."""
    match = re.fullmatch(
        r"(.+_(?:lte_(?:transformer|unet)|ptp)_\d{8}_\d{6})(?:-r\d+)?",
        eval_dir.name,
    )
    if match:
        return match.group(1)
    match = re.fullmatch(r"(.+)_eval(?:_seed\d+)?(?:-r\d+)?", eval_dir.name)
    return match.group(1) if match else None


def _saved_evaluation_runs() -> list[tuple[Path, Path]]:
    """Find evaluation outputs whose training config can be resolved uniquely."""
    inference_root = REPO_ROOT / "data" / "inference"
    if not inference_root.is_dir():
        return []
    train_runs = _saved_training_runs()

    resolved_runs = []
    for eval_dir in sorted(inference_root.iterdir(), key=lambda path: path.name):
        if not eval_dir.is_dir() or eval_dir.is_symlink():
            continue
        eval_stem = _eval_run_source_name(eval_dir)
        if eval_stem is None:
            continue
        # Evaluating an intermediate checkpoint produces names such as
        # ``run_epoch50_eval``. Resolve that back to ``run`` by choosing the
        # longest run-name prefix, which also avoids ambiguous short prefixes.
        candidates = [
            run for run in train_runs
            if eval_stem == run.name or eval_stem.startswith(f"{run.name}_")
        ]
        if candidates:
            longest_name_length = max(len(run.name) for run in candidates)
            candidates = [
                run for run in candidates if len(run.name) == longest_name_length
            ]
        if len(candidates) == 1:
            resolved_runs.append((eval_dir, candidates[0]))
    return resolved_runs


def _select_two_run_indices(prompt: str, labels: list[str]) -> tuple[int, int]:
    """Prompt once for two distinct comma-separated run numbers."""
    print()
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {label}")
    while True:
        answer = input(f"Select two {prompt}s (for example 1,2): ").strip()
        parts = [part.strip() for part in answer.split(",")]
        try:
            if len(parts) != 2:
                raise ValueError
            first_index, second_index = (int(part) - 1 for part in parts)
            if (
                first_index == second_index
                or first_index < 0
                or second_index < 0
                or first_index >= len(labels)
                or second_index >= len(labels)
            ):
                raise ValueError
            return first_index, second_index
        except ValueError:
            print(f"Enter two different numbers from 1 to {len(labels)}, separated by a comma.")


def _resolve_hydra_config_leaves(config: object, raw_value: object, prefix: str = "") -> object:
    """Resolve each saved config value without failing on dynamic Hydra fields."""
    if isinstance(raw_value, dict):
        return {
            key: _resolve_hydra_config_leaves(
                config,
                value,
                f"{prefix}.{key}" if prefix else str(key),
            )
            for key, value in raw_value.items()
        }
    try:
        from omegaconf import OmegaConf

        return OmegaConf.select(config, prefix)
    except Exception:
        # ``${now:...}`` values in logging/Hydra metadata cannot be recreated
        # from a saved config, so retain their recorded expression.
        return raw_value


def _load_saved_run_config(run_dir: Path) -> dict:
    """Load a direct final config, or resolve the legacy Hydra snapshot."""
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError("OmegaConf is required to compare saved Hydra configs.") from exc
    final_config_path = run_dir / "final_resolved_config.yaml"
    if final_config_path.is_file():
        config = OmegaConf.load(final_config_path)
        loaded_config = OmegaConf.to_container(config, resolve=True)
        if not isinstance(loaded_config, dict):
            raise ValueError(f"Expected a mapping in {final_config_path}.")
        return loaded_config

    # The config snapshot has already had the recorded ``overrides.yaml``
    # values composed into it. Register the project resolver so expressions
    # such as ``${eval:'${n_action_steps}+${n_latency_steps}'}`` resolve too.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = run_dir / ".hydra" / "config.yaml"
    config = OmegaConf.load(config_path)
    raw_config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    resolved_config = _resolve_hydra_config_leaves(config, raw_config)
    assert isinstance(resolved_config, dict)
    return resolved_config


def _flatten_config(config: object, prefix: str = "") -> dict[str, object]:
    """Turn nested YAML mappings into dotted parameter paths."""
    if isinstance(config, dict):
        if not config:
            return {prefix: {}}
        flattened = {}
        for key in sorted(config, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(config[key], path))
        return flattened
    return {prefix: config}


_CONFIG_MISSING = object()


def _config_value_label(value: object, max_length: int = 56) -> str:
    if value is _CONFIG_MISSING:
        return "<missing>"
    rendered = json.dumps(value, sort_keys=True, default=str)
    if len(rendered) <= max_length:
        return rendered
    return f"{rendered[:max_length - 3]}..."


def _print_config_diff(first_run: Path, second_run: Path) -> None:
    first_config = _flatten_config(_load_saved_run_config(first_run))
    second_config = _flatten_config(_load_saved_run_config(second_run))
    differences = []
    for key in sorted(set(first_config) | set(second_config)):
        first_value = first_config.get(key, _CONFIG_MISSING)
        second_value = second_config.get(key, _CONFIG_MISSING)
        first_label = _config_value_label(first_value)
        second_label = _config_value_label(second_value)
        if first_label != second_label:
            differences.append((key, first_label, second_label))

    print(f"\nConfig diff: {first_run.name} vs {second_run.name}")
    if not differences:
        print("  No differences.")
        return
    parameter_width = max(len("Parameter"), *(len(key) for key, _, _ in differences))
    first_width = max(len(first_run.name), *(len(value) for _, value, _ in differences))
    second_width = max(len(second_run.name), *(len(value) for _, _, value in differences))
    print(
        f"  {'Parameter':<{parameter_width}} | {first_run.name:<{first_width}} | "
        f"{second_run.name:<{second_width}}"
    )
    print(f"  {'-' * parameter_width}-|-{'-' * first_width}-|-{'-' * second_width}")
    for parameter, first_value, second_value in differences:
        print(
            f"  {parameter:<{parameter_width}} | {first_value:<{first_width}} | "
            f"{second_value:<{second_width}}"
        )


def _compare_run_configs() -> None:
    """Interactively compare two saved training configs or their evaluations."""
    run_kind = _prompt_menu(
        "Compare configs for: ", ["training runs", "evaluation runs", "back"]
    )
    if run_kind == "back":
        return
    if run_kind == "training runs":
        runs = _saved_training_runs()
        if len(runs) < 2:
            print("Need at least two saved training runs with saved configs.")
            return
        first_index, second_index = _select_two_run_indices(
            "training run", [run.name for run in runs]
        )
        _print_config_diff(runs[first_index], runs[second_index])
        return

    evaluations = _saved_evaluation_runs()
    if len(evaluations) < 2:
        print("Need at least two evaluation runs with resolvable training configs.")
        return
    labels = [f"{eval_dir.name}  <-  {train_run.name}" for eval_dir, train_run in evaluations]
    first_index, second_index = _select_two_run_indices("evaluation run", labels)
    _print_config_diff(evaluations[first_index][1], evaluations[second_index][1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    while True:
        action = _prompt_menu(
            "Select action: ",
            [
                "train",
                "eval",
                "train+eval",
                "progress",
                "add run note",
                "check success rates",
                "delete training run directories",
                "delete inference directories",
                "compress inference images to videos",
                "restore W&B run folder to external outputs",
                "show disk usage",
                "show GPU status (nvidia-smi)",
                "show GPU memory by process",
                "config diff",
                "exit",
            ],
        )
        if action == "exit":
            return
        try:
            if action == "train":
                _start_training_flow(with_evaluation=False)
            elif action == "eval":
                _start_evaluation()
            elif action == "train+eval":
                _start_training_flow(with_evaluation=True)
            elif action == "progress":
                _show_progress()
            elif action == "add run note":
                _add_run_note()
            elif action == "check success rates":
                _check_success_rates()
            elif action == "delete training run directories":
                _delete_run_directory()
            elif action == "delete inference directories":
                _delete_inference_directory()
            elif action == "compress inference images to videos":
                _compress_inference_images_to_videos()
            elif action == "restore W&B run folder to external outputs":
                _restore_wandb_run_folder()
            elif action == "show disk usage":
                _show_disk_usage()
            elif action == "show GPU status (nvidia-smi)":
                _show_gpu_status()
            elif action == "show GPU memory by process":
                _show_gpu_memory_by_process()
            else:
                _compare_run_configs()
        except (OSError, RuntimeError, ValueError) as exc:
            # A bad output path or unavailable screen should not require
            # restarting the interactive launcher.
            print(f"\n{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
