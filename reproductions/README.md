# Scene Graph Generation Reproduction Notes

This folder tracks reproducible CVPR 2025-2026 scene graph generation work.

## Current Target

The first runnable target is:

- CVPR 2025: Universal Scene Graph Generation
- Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Universal_Scene_Graph_Generation_CVPR_2025_paper.html
- Code: https://github.com/ChocoWu/USG
- Local checkout: `reproductions/USG`
- Pinned commit: `39e6b77 update configs`

## Why This One First

Among the 2025-2026 CVPR SGG papers found, USG is the clearest first reproduction target because it has public code, config files, training scripts, evaluation scripts, and dataset preparation notes in the repository.

Other directly related papers to track:

- CVPR 2025: Hybrid Reciprocal Transformer with Triplet Feature Alignment for Scene Graph Generation. Project page says code is still being cleaned up.
- CVPR 2025: Navigating the Unseen: Zero-shot Scene Graph Generation via Capsule-Based Equivariant Features. The paper says code will be available, but no official code was found in the first pass.
- CVPR 2025: Conformal Prediction and MLLM aided Uncertainty Quantification in Scene Graph Generation. No official code was found in the first pass.
- CVPR 2025: Towards Unbiased and Robust Spatio-Temporal Scene Graph Generation and Anticipation. Needs a separate Action Genome setup.
- CVPR 2026: Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation. The CVF page has the paper, but no code link.
- CVPR 2026 Findings: GEODE: Geometry-Guided Discrete Diffusion for Open-Vocabulary 3D Scene Graph Generation. Needs a separate 3D setup and code availability check.

## Local Reproduction Flow

Use the helper script:

```powershell
.\reproductions\scripts\usg.ps1 -Action check
.\reproductions\scripts\usg.ps1 -Action install
.\reproductions\scripts\usg.ps1 -Action smoke
.\reproductions\scripts\usg.ps1 -Action train-psg
.\reproductions\scripts\usg.ps1 -Action eval-psg -Checkpoint checkpoints\psg\best.pt
```

Dataset files are not included. For the first image-only reproduction, prepare PSG under:

```text
reproductions/USG/data/PSG/
```

The official repo currently uses lower-case config paths such as `data/psg/psg.json`, while the included dataset notes live under `data/PSG/`. If the dataset is placed in `data/PSG/`, either create a matching lower-case directory/link or edit `configs/psg.yaml` before running.

## Status

- Source checkout: done.
- Environment script: done.
- Local core-model smoke test: done.
- Dataset download: pending user/local dataset availability.
- Training run: pending dataset and GPU environment.
- Evaluation run: pending checkpoint.
