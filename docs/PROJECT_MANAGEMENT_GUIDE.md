# Jira and Confluence Setup Guide

This guide provides a simple way to track and document the **Self-Repairing Neural Cellular Automata for 2D Medical Image Segmentation** project.

## 1. Jira setup

### Create the project

1. Create a Jira project named **Self-Repairing Segmentation**.
2. Use a **Kanban** or **Scrum** template.
3. Use a short project key such as `SRS`.
4. Add these workflow columns:
   - To Do
   - In Progress
   - Review
   - Done
5. Create one epic:
   - **Epic:** Self-Repairing NCA Segmentation
   - **Goal:** Train, evaluate, and document a 2D NCA that can recover segmentation quality after hidden-state damage.

### Create the initial issues

Create the following stories or tasks under the epic. Replace `SRS` with the project key selected in Jira.

| Issue | Summary | Main acceptance criteria |
|---|---|---|
| `SRS-1` | Prepare hippocampus dataset | Task04 data can be downloaded, preprocessed, and split at patient level into train, validation, and test sets. |
| `SRS-2` | Train baseline NCA segmentation model | The training script runs and saves a model checkpoint plus training history. |
| `SRS-3` | Add augmentation and final model training | Albumentations and TorchIO augmentations run successfully; the final 24-channel/96-hidden-unit model is trained. |
| `SRS-4` | Run self-repair experiments | Damage at 10%, 30%, and 50% severity is tested at multiple inference steps and metrics are saved. |
| `SRS-5` | Generate result visualizations | Training curves, repair plots, healing grids/GIFs, and severity comparisons are generated in `results/`. |
| `SRS-6` | Publish technical documentation | README, benchmark results, limitations, and reproducible commands are documented. |

### Use this issue format

```text
Description
----------
What needs to be done and why.

Scope
-----
- Files or scripts involved
- Dataset, model, or experiment affected

Acceptance criteria
-------------------
- [ ] The required command or workflow runs successfully.
- [ ] Expected output is saved in the documented location.
- [ ] Results or limitations are recorded.

Evidence
--------
Link to a commit, checkpoint, result file, plot, or Confluence page.
```

### Recommended Jira routine

1. Move an issue to **In Progress** before starting work.
2. Add the command, configuration, and expected output to the issue.
3. Attach or link generated results such as checkpoints, JSON metrics, plots, or GIFs.
4. Move the issue to **Review** when the acceptance criteria are complete.
5. Move it to **Done** after the result is reproducible and linked from Confluence.

## 2. Confluence setup

### Create the project space and home page

1. Create a Confluence space named **Self-Repairing Segmentation**.
2. Create a home page named **Self-Repairing NCA Segmentation**.
3. Add this summary near the top:

> This project evaluates whether a Neural Cellular Automata segmentation model can recover its prediction after part of its hidden cell-state grid is corrupted during inference. The study uses 2D hippocampus MRI slices from Medical Segmentation Decathlon Task04.

4. Add links to the Jira project, repository, latest checkpoint, and latest experiment results.

### Create these child pages

#### 1. Project overview

Record:

- Objective and self-repair hypothesis
- Dataset and hardware target
- Current best results: validation Dice `0.7475` and test Dice `0.7173`
- Project limitations: 2D slices, 64×64 resolution, 40-volume subset, and rectangular damage

#### 2. Setup and runbook

Copy the reproducible workflow from the repository README:

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/preprocess_data.py
python scripts/train.py --epochs 180 --batch_size 16 --n_channels 24 --hidden_dim 96 --nca_steps_min 20 --nca_steps_max 36 --eval_steps 32 --augment
python scripts/repair_experiment.py
python scripts/visualize.py
```

Also document the expected output locations:

- `checkpoints/` — trained model weights and training history
- `results/` — metrics, plots, and healing visualizations
- `docs/REPORT.md` — detailed technical report

#### 3. Model and data pipeline

Document the 64×64 input size, 24-channel hidden state, local 3×3 perception, 96-unit update MLP, firing mask, loss function, augmentation pipeline, and patient-level data split.

#### 4. Experiments and results

Add the benchmark table from `docs/REPORT.md`, link the Jira experiment issue, and embed the plots from `results/`.

#### 5. Limitations and next steps

Track follow-up work such as full 3D segmentation, larger datasets, higher resolution, non-rectangular damage, and comparison with a U-Net baseline.

### Link Jira and Confluence

1. Add the Confluence home page link to the Jira epic.
2. Add the relevant Jira issue link to each experiment or documentation page.
3. Keep one page section called **Open work** with links to all unfinished Jira issues.
4. When an issue is completed, add its result link to the matching Confluence page.

## 3. Definition of done

An item is complete when:

- The relevant script or documentation change is finished.
- The workflow has been run, or the reason it could not be run is recorded.
- Outputs are stored in the expected project folder.
- Jira acceptance criteria are checked.
- The result is linked from Confluence.
