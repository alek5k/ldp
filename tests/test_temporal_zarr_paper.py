"""Exercise native task data and PTP configuration without external datasets."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.temporal_zarr_image_dataset import TemporalZarrImageDataset


class TestPaperTemporalData(unittest.TestCase):
    def test_native_image_windows_without_step_count(self):
        with tempfile.TemporaryDirectory() as directory:
            store = zarr.open(directory, mode='w')
            data, meta = store.create_group('data'), store.create_group('meta')
            data.array('action', np.arange(16, dtype=np.float32).reshape(8, 2))
            data.array('agent_pose', np.zeros((8, 2), dtype=np.float32))
            data.array('full_image', np.zeros((8, 3, 16, 16), dtype=np.uint8))
            meta.array('episode_ends', np.array([4, 8]))
            for cached in [False, True]:
                dataset = TemporalZarrImageDataset(
                    directory, horizon=4, n_obs_steps=2, pad_before=1,
                    pad_after=1, val_ratio=0., cache_images_in_memory=cached,
                )
                item = dataset[0]
                self.assertEqual(item['obs']['image'].shape, (2, 3, 16, 16))
                self.assertEqual(item['action'].shape, (4, 2))
                self.assertEqual(set(item), {'obs', 'action'})
                dataset.get_normalizer()

    def test_native_ptp_task_configs_compose(self):
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        for task in ['waitatgoal_image', 'liftqa_image']:
            with initialize_config_dir(config_dir=str(root / 'diffusion_policy/config'), version_base=None):
                cfg = compose(config_name='train_diffusion_transformer_hybrid_workspace', overrides=[
                    'task=' + task, 'horizon=32', 'n_obs_steps=16',
                    'n_action_steps=8', '+policy.past_action_pred=true',
                    '+policy.past_steps_reg=-1',
                ])
                OmegaConf.resolve(cfg)
                self.assertTrue(cfg.policy.past_action_pred)
                self.assertEqual(cfg.policy.horizon, 32)
                self.assertEqual(cfg.task.dataset.n_obs_steps, 16)


if __name__ == '__main__':
    unittest.main()
