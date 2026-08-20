"""Evaluation and analysis rollout runner for WaitAtGoal and LiftQA."""
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.real_world.video_recorder import VideoRecorder


class TemporalImageRunner(BaseImageRunner):
    """Run native temporal environments and persist policy trajectories.

    Unlike the Robomimic runners, this is deliberately single-process: it
    keeps Pygame and MuJoCo rendering predictable and writes an ``.npz`` plus
    an MP4 for each requested visual rollout.  The NPZ keeps timing values used
    by the TemporalDiffusionPolicy analysis scripts.
    """

    def __init__(
        self,
        output_dir,
        env_name: str,
        n_train: int = 0,
        n_train_vis: int = 0,
        train_start_seed: int = 0,
        n_test: int = 10,
        n_test_vis: int = 2,
        test_start_seed: int = 10000,
        max_steps: int = 1000,
        n_obs_steps: int = 16,
        n_action_steps: int = 1,
        subsample_frames: int = 1,
        fps: int = 10,
        render_size: int = 144,
        constrain_motion: bool = True,
        zarr_path: str = None,
        zarr_mode: str = "w",
        tqdm_interval_sec: float = 1.0,
        **kwargs,
    ):
        super().__init__(output_dir)
        if env_name not in {"waitatgoal", "liftqa"}:
            raise ValueError(f"Unsupported temporal environment: {env_name}")
        self.env_name = env_name
        self.n_train = n_train
        self.n_train_vis = n_train_vis
        self.train_start_seed = train_start_seed
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.subsample_frames = subsample_frames
        self.fps = fps
        self.render_size = render_size
        self.constrain_motion = constrain_motion
        if zarr_mode not in {"w", "a"}:
            raise ValueError("zarr_mode must be 'w' (new store) or 'a' (append)")
        self.zarr_path = zarr_path
        self.zarr_mode = zarr_mode
        self.replay_buffer = None

    def _make_env(self, seed: int):
        if self.env_name == "waitatgoal":
            from diffusion_policy.env.waitatgoal.waitatgoal_env import WaitAtGoal
            env = WaitAtGoal(render_size=self.render_size)
            env.seed(seed)
            return env
        from diffusion_policy.env.liftqa.lift_qa import create_env
        return create_env(
            render=False,
            constrain_motion=self.constrain_motion,
            control_hz=self.fps,
            # LiftQA's reset includes seven settling steps. Keep its native
            # horizon so a deliberately short evaluation cannot terminate
            # while reset is still initializing; this runner limits rollouts
            # to self.max_steps below.
            max_episode_length=max(self.max_steps, 1000),
            seed=seed,
            camera_height_width=(self.render_size, self.render_size),
        )

    def _policy_obs(self, history):
        history = list(history)[self.subsample_frames - 1::self.subsample_frames]
        return {
            "image": np.stack([x["full_image"] for x in history]),
            "agent_pose": np.stack([x["agent_pose"] for x in history]),
        }

    def _env_action(self, action):
        if self.env_name == "waitatgoal":
            return action
        from diffusion_policy.env.liftqa.lift_qa import translate_3_dim_action_to_7d
        return translate_3_dim_action_to_7d(action)

    @staticmethod
    def _video_frame(obs):
        image = np.asarray(obs["full_image"])
        if image.ndim != 3:
            raise ValueError(f"Expected CHW image, received {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return np.moveaxis(image, 0, -1)

    def _run_episode(self, policy, prefix: str, seed: int, save_video: bool):
        env = self._make_env(seed)
        media_dir = Path(self.output_dir) / "media"
        rollout_dir = Path(self.output_dir) / "rollouts"
        media_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix.rstrip('/')}_seed{seed}"
        video_path = media_dir / f"{stem}.mp4"
        recorder = None
        if save_video:
            recorder = VideoRecorder.create_h264(
                fps=self.fps, codec="h264", input_pix_fmt="rgb24", crf=22,
                thread_type="FRAME", thread_count=1,
            )

        try:
            obs = env.reset()
            raw_history_length = self.n_obs_steps * self.subsample_frames
            history = deque([obs] * raw_history_length, maxlen=raw_history_length)
            trajectory: Dict[str, list] = {
                "image": [], "agent_pose": [], "action": [], "reward": [],
                "done": [], "wait_time": [], "step_count": [],
            }
            zarr_episode = []
            total_reward = 0.0
            done = False
            steps = 0
            policy.reset()
            # LTE-IMG-NoT is stateful between replans.  Feature extraction is
            # deliberately done once for every real observation, not only once
            # for each predicted action chunk.
            if hasattr(policy, "advance_temporal_state"):
                policy.advance_temporal_state(
                    torch.from_numpy(np.asarray(obs["full_image"])).to(policy.device)
                )
            with tqdm(total=self.max_steps, desc=f"{prefix}seed {seed}", unit="step") as pbar:
                while not done and steps < self.max_steps:
                    np_obs = self._policy_obs(history)
                    torch_obs = dict_apply(
                        {key: value[None] for key, value in np_obs.items()},
                        lambda value: torch.from_numpy(value).to(policy.device),
                    )
                    with torch.no_grad():
                        action_batch = policy.predict_action(torch_obs)["action"]
                    actions = action_batch[0].detach().cpu().numpy()
                    for action in actions:
                        trajectory["image"].append(obs["full_image"])
                        trajectory["agent_pose"].append(obs["agent_pose"])
                        trajectory["action"].append(action.astype(np.float32))
                        trajectory["wait_time"].append(np.float32(obs.get("wait_time", 0.0)))
                        trajectory["step_count"].append(np.int64(obs.get("step_count", steps)))
                        if recorder is not None:
                            if not recorder.is_ready():
                                recorder.start(str(video_path))
                            recorder.write_frame(self._video_frame(obs))
                        obs, reward, done, _ = env.step(self._env_action(action.copy()))
                        if hasattr(policy, "advance_temporal_state"):
                            policy.advance_temporal_state(
                                torch.from_numpy(np.asarray(obs["full_image"])).to(policy.device)
                            )
                        if self.replay_buffer is not None:
                            # Match TemporalDiffusionPolicy inference: save the
                            # pre-action observation and the policy action as one
                            # replay-buffer step, with the resulting transition
                            # reward and terminal flag alongside it.
                            step_data = {
                                key: np.asarray(value)
                                for key, value in history[-1].items()
                                if isinstance(value, (np.ndarray, np.number, float, int, bool))
                            }
                            step_data["action"] = action.astype(np.float32)
                            step_data["reward"] = np.asarray(reward, dtype=np.float32)
                            step_data["done"] = np.asarray(done, dtype=np.bool_)
                            zarr_episode.append(step_data)
                        trajectory["reward"].append(np.float32(reward))
                        trajectory["done"].append(bool(done))
                        total_reward += float(reward)
                        history.append(obs)
                        steps += 1
                        pbar.update(1)
                        if done or steps >= self.max_steps:
                            break
            if recorder is not None and recorder.is_ready():
                recorder.stop()
            np.savez_compressed(
                rollout_dir / f"{stem}.npz",
                **{key: np.asarray(value) for key, value in trajectory.items()},
            )
            if self.replay_buffer is not None and zarr_episode:
                self.replay_buffer.add_episode(
                    {key: np.stack([step[key] for step in zarr_episode])
                     for key in zarr_episode[0]},
                    compressors="disk",
                )
            return total_reward, str(video_path) if save_video and video_path.exists() else None
        finally:
            if recorder is not None and recorder.is_ready():
                recorder.stop()
            env.close()

    def run(self, policy) -> Dict:
        was_training = policy.training
        policy.eval()
        scores = {"train/": [], "test/": []}
        log_data = {}
        if self.zarr_path is not None:
            zarr_path = Path(self.zarr_path)
            zarr_path.parent.mkdir(parents=True, exist_ok=True)
            self.replay_buffer = ReplayBuffer.create_from_path(
                str(zarr_path), mode=self.zarr_mode)
        specs = [
            ("train/", self.n_train, self.n_train_vis, self.train_start_seed),
            ("test/", self.n_test, self.n_test_vis, self.test_start_seed),
        ]
        for prefix, count, visual_count, start_seed in specs:
            for index in range(count):
                seed = start_seed + index
                score, video_path = self._run_episode(
                    policy, prefix, seed, save_video=index < visual_count)
                scores[prefix].append(score)
                log_data[f"{prefix}sim_max_reward_{seed}"] = score
                if video_path is not None:
                    log_data[f"{prefix}sim_video_{seed}"] = wandb.Video(video_path)
        for prefix, values in scores.items():
            if values:
                log_data[f"{prefix}mean_score"] = float(np.mean(values))
        if was_training:
            policy.train()
        self.replay_buffer = None
        return log_data
