---
language:
- en
tags:
- robotics
- imitation-learning
- diffusion-policy
---
# TC-DP paper reproduction data

Demonstrations, selected checkpoints and evaluation rollouts for the main
three-training-seed comparison in *Mitigating Temporal Failures in Diffusion-Based
Robotic Imitation Learning*. Rollout images are retained as uint8 pixels.

## Current paper selection

- `demonstration/`: WaitAtGoal, LiftQA and LiftQA-Real demonstrations.
- `checkpoints/`: 21 selected DP/TC-DP checkpoints and their configurations.
- `inference/`: 28 evaluation datasets, including the separate PIP evaluations.
- `ldp/outputs/`: six selected PTP checkpoints and resolved configurations.
- `ldp/inference/`: six PTP evaluations (three seeds for each simulated task).

Zarr datasets are distributed as `.zarr.zip` archives. Each archive contains the
original `.zarr` directory, all its arrays (including images), and metadata.
Extract beside the archive to restore the dataset. PTP uses the same WaitAtGoal
and LiftQA demonstrations; these are not duplicated.

## Download and reproduce

In the matching TemporalDiffusionPolicy code release, run:

```bash
python src/temporal_diffusion_policy/tools/dataset_upload.py --download
python -m temporal_diffusion_policy.analysis.report
```

In the matching ldp release, download the PTP files into `data/` with:

```bash
python dataset_upload.py --download
```

Download the shared demonstrations with TemporalDiffusionPolicy. Set paths in
analysis study lists and training configurations to your local repository paths;
PTP studies should point at the downloaded ldp `data/inference/` directories.
The default `paper_main` report uses three training seeds for WaitAtGoal and
LiftQA and the single-policy LiftQA-Real evaluations. It reports means and
sample standard deviations across training seeds for the simulated tasks.

The matching TemporalDiffusionPolicy code's `tcdp_paper/paper_upload_files.json`
and ldp's `paper_upload_files.json` list the exact current-paper selection. The upload adds these files without deleting older hosted experiments.
The selection excludes ablations and other exploratory runs.

Robomimic checkpoints and rollouts are hosted separately at
https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy_robomimic.
Can, Lift and Square demonstrations should be obtained from robomimic upstream.

## Citation

Temporary citation for this repository during anonymous review. Replace this
entry with the final publication details when available.

```bibtex
@misc{tcdp2026,
  title={Mitigating Temporal Failures in Diffusion-Based Robotic Imitation Learning},
  author={{Anonymous Authors}},
  year={2026},
  howpublished={Hugging Face dataset repository},
  url={https://huggingface.co/datasets/erdrsvhr9/temporally_conditioned_diffusion_policy},
  note={Anonymous review version; temporary citation}
}
```
