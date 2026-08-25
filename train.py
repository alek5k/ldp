"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pathlib

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _write_final_resolved_config(cfg: OmegaConf) -> None:
    """Persist the complete, interpolation-free Hydra configuration."""
    output_path = pathlib.Path(HydraConfig.get().runtime.output_dir) / "final_resolved_config.yaml"
    OmegaConf.save(config=cfg, f=output_path, resolve=True)
    print(f"Wrote final resolved config: {output_path}")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    _write_final_resolved_config(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
