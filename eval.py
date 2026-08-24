"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output -p perturbation
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
from omegaconf import OmegaConf, open_dict
import wandb
import json
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-p', '--force_perturbs', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('-n', '--num_samples', default=1)
@click.option('--zarr_path', default=None,
    help='Optional Zarr store for raw image-policy evaluation episodes.')
@click.option('--n_test', type=int, default=None,
    help='Override the number of fixed-seed test episodes.')
@click.option('--n_train', type=int, default=None,
    help='Optional number of training-seed episodes (defaults to 0 for Zarr evaluation).')
@click.option('--n_test_vis', type=int, default=None,
    help='Override how many test episodes also receive MP4 recordings.')
@click.option('--test_start_seed', type=int, default=None,
    help='Override the first deterministic test seed.')
@click.option('--max_steps', type=int, default=None,
    help='Override the maximum number of environment steps per episode.')
@click.option('--n_action_steps', type=int, default=None,
    help='Override the number of predicted actions executed before replanning.')
@click.option('--num_inference_steps', type=int, default=None,
    help='Override the diffusion denoising steps used by the loaded policy.')
def main(checkpoint, output_dir, force_perturbs, device, num_samples,
         zarr_path, n_test, n_train, n_test_vis, test_start_seed, max_steps,
         n_action_steps, num_inference_steps):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if force_perturbs:
        perturb_cfg = OmegaConf.load(force_perturbs)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if force_perturbs and "chunk" in force_perturbs:
        policy.n_action_steps = perturb_cfg["chunk"]
        with open_dict(cfg):
            cfg.task.env_runner.n_action_steps = policy.n_action_steps
        print(policy.n_action_steps, "cfg: ", cfg.task.env_runner.n_action_steps)

    # rewrite config for env_runner
    if force_perturbs:
        with open_dict(cfg):
            print(perturb_cfg)
            cfg.task.env_runner.n_samples = num_samples
            cfg.task.env_runner.perturbations = perturb_cfg
            cfg.task.env_runner.n_test = 150 # many evals for lower variance

    if num_inference_steps is not None:
        policy.num_inference_steps = num_inference_steps

    if zarr_path is not None:
        supported_zarr_runners = (
            "temporal_image_runner",
            "robomimic_image_runner",
            "robomimic_longhist_image_runner",
            "aloha_image_runner",
        )
        if not any(name in cfg.task.env_runner._target_
                   for name in supported_zarr_runners):
            raise click.UsageError(
                "--zarr_path is supported by the image environment runners only")

    if any(value is not None for value in
           (zarr_path, n_test, n_train, n_test_vis, test_start_seed, max_steps,
            n_action_steps)):
        with open_dict(cfg):
            # Image runners accept these rollout controls. Their optional
            # Zarr export records raw environment transitions, including every
            # action inside a multi-step action chunk.
            if zarr_path is not None:
                cfg.task.env_runner.zarr_path = zarr_path
                cfg.task.env_runner.zarr_mode = 'w'
            if n_test is not None:
                cfg.task.env_runner.n_test = n_test
            # Inference is normally test-only. The training runner's default
            # visual rollouts are useful during fitting but should not be
            # mixed into a checkpoint evaluation Zarr unless requested.
            cfg.task.env_runner.n_train = 0 if n_train is None else n_train
            cfg.task.env_runner.n_train_vis = 0
            if n_test_vis is not None:
                cfg.task.env_runner.n_test_vis = n_test_vis
            if test_start_seed is not None:
                cfg.task.env_runner.test_start_seed = test_start_seed
            if max_steps is not None:
                cfg.task.env_runner.max_steps = max_steps
            if n_action_steps is not None:
                if n_action_steps < 1:
                    raise click.UsageError("--n_action_steps must be positive")
                # The transformer always predicts the full horizon here; this
                # setting chooses how many of those predictions to execute.
                policy.n_action_steps = n_action_steps
                cfg.task.env_runner.n_action_steps = n_action_steps

    # run eval
    total_eval_episodes = (
        int(cfg.task.env_runner.n_train) + int(cfg.task.env_runner.n_test)
    )
    print(f"EVAL_PROGRESS total={total_eval_episodes} completed=0")
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    print(runner_log)
    # Dump a complete, portable result file only after every value has been
    # converted. Image runners commonly return NumPy scalar rewards, which
    # the standard JSON encoder does not handle directly.
    def json_value(value):
        if isinstance(value, wandb.sdk.data_types.video.Video):
            return value._path
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value

    json_log = {key: json_value(value) for key, value in runner_log.items()}
    out_path = pathlib.Path(output_dir) / 'eval_log.json'
    temporary_out_path = out_path.with_suffix('.json.tmp')
    print(json_log)
    with temporary_out_path.open('w', encoding='utf-8') as file:
        json.dump(json_log, file, indent=2, sort_keys=True, allow_nan=False)
    temporary_out_path.replace(out_path)

if __name__ == '__main__':
    main()
