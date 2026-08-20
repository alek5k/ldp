#!/usr/bin/env python3
"""Launch LTE-IMG-NoT training and evaluation runs in detached screen sessions."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train_lte_img_not.sh"
EVAL_SCRIPT = REPO_ROOT / "eval.py"
TASKS = (
    "square",
    "tool_hang",
    "transport",
    "lh-aloha",
    "lh-square",
)


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


def _prompt_gpu() -> str:
    while True:
        answer = input("GPU index (CUDA_VISIBLE_DEVICES) [0]: ").strip()
        if not answer:
            return "0"
        if answer.isdigit():
            return answer
        print("Enter one numeric GPU index, for example 0 or 1.")


def _make_screen_session(label: str) -> str:
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-_") or "job"
    return f"ldp-{safe_label}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def _start_screen_session(session: str, command: str) -> None:
    screen = shutil.which("screen")
    if screen is None:
        raise RuntimeError("screen was not found on PATH; cannot launch the experiment.")

    script_path = Path("/tmp") / f"{session}.sh"
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        "source ~/.bashrc 2>/dev/null || true\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
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
        f"PATH={shlex.quote(str(python_bin))}:$PATH",
    ))


def _run_name(task: str, decoder: str, seed: int) -> str:
    return f"{task.replace('-', '_')}_lte_{decoder}_seed{seed}"


def _task_name(decoder: str) -> tuple[str, str]:
    """Return the base task for naming and its launcher argument."""
    task = _prompt_menu("Select task: ", list(TASKS))
    launcher_task = task if decoder == "transformer" else f"{task}-unet"
    return task, launcher_task


def _planned_runs(task: str, decoder: str) -> list[tuple[int, Path, Path]]:
    run_count = _prompt_int("Sequential runs", default=1, minimum=1)
    first_seed = _prompt_int("First training seed", default=42, minimum=0)
    output_root = Path(input("Training output root [data/outputs]: ").strip() or "data/outputs")
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    runs = []
    for offset in range(run_count):
        seed = first_seed + offset
        name = _run_name(task, decoder, seed)
        output_dir = output_root / name
        inference_dir = REPO_ROOT / "data" / "inference" / f"{name}_eval"
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing output directory: {output_dir}. "
                "Choose another root or seed."
            )
        if inference_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing evaluation directory: {inference_dir}. "
                "Choose another seed."
            )
        runs.append((seed, output_dir, inference_dir))
    return runs


def _training_command(task: str, output_dir: Path, seed: int, gpu: str) -> str:
    command = [
        str(TRAIN_SCRIPT),
        task,
        str(output_dir),
        f"training.seed={seed}",
        "training.device=cuda:0",
    ]
    return f"{_environment_prefix(gpu)} {shlex.join(command)}"


def _evaluation_command(
    checkpoint: Path,
    output_dir: Path,
    test_start_seed: int,
    gpu: str,
    n_test: int,
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
    return f"{_environment_prefix(gpu)} {shlex.join(command)}"


def _start_training(with_evaluation: bool) -> None:
    decoder = _prompt_menu("Select decoder: ", ["transformer (default)", "unet"])
    decoder_name = "transformer" if decoder.startswith("transformer") else "unet"
    base_task, task = _task_name(decoder_name)
    gpu = _prompt_gpu()
    runs = _planned_runs(base_task, decoder_name)
    n_test = _prompt_int("Evaluation test episodes", default=100, minimum=1) if with_evaluation else 0

    commands = []
    for seed, output_dir, inference_dir in runs:
        train = _training_command(task, output_dir, seed, gpu)
        if with_evaluation:
            checkpoint = output_dir / "checkpoints" / "latest.ckpt"
            evaluate = _evaluation_command(checkpoint, inference_dir, seed, gpu, n_test)
            commands.append(f"{train} && {evaluate}")
        else:
            commands.append(train)
    label = f"{'train-eval' if with_evaluation else 'train'}-{task}-x{len(runs)}"
    _start_screen_session(_make_screen_session(label), " && ".join(commands))


def _available_checkpoints() -> list[Path]:
    root = REPO_ROOT / "data" / "outputs"
    if not root.is_dir():
        return []
    return sorted(root.glob("**/checkpoints/latest.ckpt"), key=lambda path: str(path))


def _prompt_checkpoint() -> Path:
    checkpoints = _available_checkpoints()
    if not checkpoints:
        raise RuntimeError("No latest checkpoints found under data/outputs.")
    choice_labels = [str(path.relative_to(REPO_ROOT)) for path in checkpoints]
    selected = _prompt_menu("Select checkpoint: ", choice_labels)
    return checkpoints[choice_labels.index(selected)]


def _start_evaluation() -> None:
    checkpoint = _prompt_checkpoint()
    gpu = _prompt_gpu()
    run_count = _prompt_int("Sequential evaluation runs", default=1, minimum=1)
    first_seed = _prompt_int("First evaluation seed", default=42, minimum=0)
    n_test = _prompt_int("Test episodes per run", default=100, minimum=1)

    run_name = checkpoint.parents[1].name
    commands = []
    for offset in range(run_count):
        seed = first_seed + offset
        output_dir = REPO_ROOT / "data" / "inference" / f"{run_name}_eval_seed{seed}"
        if output_dir.exists():
            raise FileExistsError(
                f"Refusing to reuse existing evaluation directory: {output_dir}."
            )
        commands.append(_evaluation_command(checkpoint, output_dir, seed, gpu, n_test))
    _start_screen_session(
        _make_screen_session(f"eval-{run_name}-x{run_count}"),
        " && ".join(commands),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    action = _prompt_menu("Select action: ", ["train", "eval", "train+eval"])
    if action == "train":
        _start_training(with_evaluation=False)
    elif action == "eval":
        _start_evaluation()
    else:
        _start_training(with_evaluation=True)


if __name__ == "__main__":
    main()
