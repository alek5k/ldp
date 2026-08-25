if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.wandb_checkpoint import (
    sync_checkpoints_to_wandb,
    sync_run_folder_to_wandb,
)
from hsic import batch_hsic
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        dataset.__getitem__(0)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        if cfg.training.debug:
            cfg.task.env_runner.n_envs = 10
            cfg.task.env_runner.n_test = 5
            cfg.task.env_runner.n_train = 5
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # configure env
        env_runner: BaseImageRunner = None
        if "env_runner" in cfg.task:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_kwargs = OmegaConf.to_container(cfg.logging, resolve=True)
        # This is a trainer control, not a W&B init option.
        wandb_kwargs.pop("log_every_n_steps", None)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **wandb_kwargs
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        
        # save batch for sampling
        train_sampling_batch = None

        # LTE history frame indices select from a CPU-resident NumPy/Zarr
        # cache. Keeping this metadata on CPU avoids synchronizing back from
        # CUDA once per temporal image chunk.
        temporal_history_cpu_keys = {
            "temporal_history_image_indices",
            "temporal_history_mask",
            "temporal_obs_history_indices",
        }

        def transfer_batch_to_device(batch):
            return {
                key: value if key in temporal_history_cpu_keys
                else (
                    dict_apply(value, lambda x: x.to(device, non_blocking=True))
                    if isinstance(value, dict)
                    else value.to(device, non_blocking=True)
                )
                for key, value in batch.items()
            }

        def lte_loss_components() -> dict[str, float]:
            getter = getattr(self.model, "get_last_loss_components", None)
            if getter is None:
                return {}
            return {
                name: float(value.item())
                for name, value in getter().items()
            }

        # training loop
        debug = True
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                refresh_temporal_cache = getattr(
                    self.model, "refresh_temporal_embedding_cache", None
                )
                if refresh_temporal_cache is not None and refresh_temporal_cache(self.epoch):
                    print(
                        "Refreshed detached LTE image-embedding cache "
                        f"at epoch {self.epoch}."
                    )

                train_losses = list()
                train_component_losses: dict[str, list[float]] = {}
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = transfer_batch_to_device(batch)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        
                        # compute loss
                        raw_loss = self.model.compute_loss(batch, debug)
                        debug = False
                        component_values = lte_loss_components()
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        for name, value in component_values.items():
                            train_component_losses.setdefault(name, []).append(value)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update({
                            f'train_loss_{name}': value
                            for name, value in component_values.items()
                        })

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                for name, values in train_component_losses.items():
                    step_log[f'train_loss_{name}'] = np.mean(values)

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if ((self.epoch + 1) % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # Match the checkpoint schedule below: epoch numbers are
                # one-based externally, so validation must run at epochs
                # 10, 20, ... rather than 1, 11, ....  This also guarantees
                # that val_loss exists when a 50-epoch checkpoint is ranked.
                if ((self.epoch + 1) % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_component_losses: dict[str, list[float]] = {}
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = transfer_batch_to_device(batch)
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                for name, value in lte_loss_components().items():
                                    val_component_losses.setdefault(name, []).append(value)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            for name, values in val_component_losses.items():
                                step_log[f'val_loss_{name}'] = np.mean(values)

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = transfer_batch_to_device(train_sampling_batch)
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        
                        if not policy.past_action_pred:
                            pred_action = pred_action[:, policy.n_obs_steps - 1:]
                            gt_action = gt_action[:, policy.n_obs_steps - 1:]

                        step_log["hsic_action_pred_offline"] = batch_hsic(pred_action).mean()
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if ((self.epoch + 1) % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    if topk_manager.monitor_key not in metric_dict:
                        # Some configs save ``latest.ckpt`` more often than
                        # they run rollouts.  Their ranked checkpoint metric
                        # (for example ``test_mean_score``) is therefore not
                        # available on every checkpoint epoch.  Preserve the
                        # latest checkpoint and wait for the next epoch that
                        # provides the ranking metric instead of aborting.
                        print(
                            "Skipping ranked checkpoint: missing monitor "
                            f"{topk_manager.monitor_key!r}."
                        )
                        topk_ckpt_path = None
                    else:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        # Save the actual completed model rather than only the most recent
        # scheduled checkpoint, then upload every retained checkpoint after
        # pending background writes have completed.
        if self._saving_thread is not None:
            self._saving_thread.join()
        final_checkpoint = self.save_checkpoint(use_thread=False)
        try:
            checkpoint_count = sync_checkpoints_to_wandb(
                wandb,
                wandb_run,
                output_dir=self.output_dir,
                checkpoint_dir=pathlib.Path(final_checkpoint).parent,
                epoch=self.epoch,
            )
            print(f"Synced {checkpoint_count} checkpoints to Weights & Biases.")
            file_count = sync_run_folder_to_wandb(
                wandb,
                wandb_run,
                output_dir=self.output_dir,
                epoch=self.epoch,
            )
            print(f"Synced {file_count} run-folder files to Weights & Biases.")
        except Exception as error:
            # A W&B/network failure should not make an otherwise successful
            # train (or train+eval) screen command fail.
            print(f"Could not sync checkpoints to Weights & Biases: {error}")
        finally:
            try:
                wandb_run.finish()
            except Exception as error:
                print(f"Could not finish Weights & Biases run: {error}")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
