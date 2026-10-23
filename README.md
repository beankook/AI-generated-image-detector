# Closing the Gap Among AI Image Detectors Through Adversarial Hardening and CNN-Based Generalisation

An AI-generated-image detector built from an ImageNet-pretrained ResNet-18 with **growable specialist branches**, hardened against a **PGD adversary**, and updatable after deployment to catch a new generator family by training under 1% of its parameters.

**Team 33** · Srinidhi Sudarsan (2023103001) · CS23E02 Artificial Intelligence · Department of CSE, CEG, Anna University, Chennai

**Base paper:** D. Park, H. Na, D. Choi, *Performance Comparison and Visualization of AI-Generated-Image Detection Methods*, IEEE Access 12 (2024), pp. 62609–62627. DOI: 10.1109/ACCESS.2024.3394250

---

## What this project does

Detectors trained on one generator (for example ProGAN) usually fail on images from another family (for example Stable Diffusion), and they are easily fooled by invisible pixel changes. This project tests two remedies and reports honestly where each one works and where it doesn't:

1. **Adversarial hardening.** The detector is trained in rounds against a projected-gradient-descent (PGD) adversary on a difficulty curriculum calibrated from the baseline's own robustness.
2. **A growable branch architecture.** Small specialist branches share a frozen backbone and are combined by a fusion head. After deployment, a new generator family is learnt by adding one branch and training only that branch and the fusion head.

## Key results

| Claim tested | Evidence | Verdict |
| --- | --- | --- |
| Hardening improves robustness | Accuracy at ε = 1/255: 0.003 → 0.242 | ✅ Supported |
| Hardening generalises to unseen generators | StyleGAN2 AUROC 0.594 → 0.668 (sealed test); diffusion 0.461 → 0.407; BigGAN at chance | ⚠️ Same family only (GAN → GAN) |
| Growing branches during training helps | Ties with a one-branch model trained identically | ❌ Not supported |
| A post-deployment update learns a new family cheaply | Stable Diffusion AUROC 0.405 → 0.793, Midjourney 0.461 → 0.645, training 0.85% of parameters | ✅ Supported |
| Frozen parts stay untouched | Backbone and old branches bit-identical after every update | ✅ Supported |
| Forgetting can be controlled | ProGAN accuracy loss 8.6 → 3.0 points with Learning without Forgetting (λ = 1) | ✅ Supported, with a trade-off |

The notebook and report also document **five failed experiments** (augmentation leak, a label bug, DCGAN mode collapse, growth on an untrained model, and an uncalibrated attack budget), each with its diagnosis and fix.

## Repository contents

| File | What it is |
| --- | --- |
| `kaggle_data_sampling.ipynb` | **Kaggle notebook.** Samples the source datasets, resizes to 128×128, builds the seven splits and writes `aigi_data.zip`. |
| `AI_project_final.ipynb` | **Colab notebook.** The full project, organised into Modules 1–7, with recorded outputs, failed runs marked ❌, and evaluation. |
| `AI_Project_Report_Final.docx` | The project report. |
| `README.md` | This file. |

> Rename the two notebook rows above to match your actual file names.

## Project structure (Colab notebook)

| Module | Purpose | File it writes to Drive |
| --- | --- | --- |
| 1 Data pipeline | Seven splits, augmentation, loaders | `data_module.py` |
| 2 Backbone | ResNet-18 that can be locked (weights *and* BatchNorm) | `model_module.py` |
| 3 Branch bank + fusion head | Growable classifier; growth leaves outputs unchanged | `model_module.py` |
| 4 Adversaries | Staged degradation ❌, live DCGAN ❌, PGD ✅ | `generator_module.py` |
| 5 Training | Baseline, adversarial loop with gated growth | `train_utils.py`, `train_loop.py` |
| 6 Deployment + update | Freeze, add one branch, Learning without Forgetting | `deploy_module.py` |
| 7 Evaluation | Hardened vs baseline, growth ablation, sealed Pool C test | — |

## Data

All data comes from public Kaggle datasets and is **not stored in this repository**.

| Source | Kaggle dataset | Used for |
| --- | --- | --- |
| ForenSynths | `chandlertimm/forensynths` | LSUN cat + horse reals, ProGAN fakes (training, validation), StyleGAN2 and BigGAN (sealed test) |
| GenImage subset | `renhuang8/genimage-subset-detection` | Stable Diffusion and Midjourney fakes with ImageNet reals |

Seven disjoint, class-balanced splits, all 128×128 PNG:

| Split | Real / Fake | Role |
| --- | --- | --- |
| `train` | 5,000 / 5,000 | Only training data (ProGAN) |
| `val` | 400 / 400 | Early stopping |
| `val2` | 1,500 / 1,500 | Evaluation gate (held out) |
| `poolB_sd` | 1,500 / 1,500 | Unseen Stable Diffusion; split 1,000 / 500 per class for the update |
| `poolB_mj` | 500 / 500 | Unseen Midjourney, never trained on |
| `poolC_stylegan2` | 1,000 / 1,000 | Sealed final test, run once |
| `poolC_biggan` | 1,000 / 1,000 | Sealed final test, run once |

## How to run

**Requirements:** a Kaggle account (for step 1) and Google Colab with a GPU runtime (a T4 is enough) plus Google Drive.

1. **Build the data.** Open `kaggle_data_sampling.ipynb` on Kaggle, attach the two datasets above, and run it. Download the resulting `aigi_data.zip`.
2. **Set up Drive.** Create a folder named `AI project` in your Google Drive (`/content/drive/MyDrive/AI project`) and upload `aigi_data.zip` into it.
3. **Run the project.** Open `AI_project_final.ipynb` in Colab, set the runtime to GPU, and run **Setup** and then **Quick start**. Run the module sections in order the first time: each one writes its `.py` file to Drive, and later sections import from them.
4. **Training switches** (set in Setup):
   - `RETRAIN = False` (default) loads saved checkpoints instead of retraining. Set it to `True` to retrain and overwrite them.
   - `RERUN_FAILED_EXPERIMENTS = False` (default) skips the failed runs and shows their recorded logs. Set it to `True` to re-run the ones that still work with the current code.

Training from scratch takes a few hours on a T4. Running the notebook with the default switches and existing checkpoints takes about 10–15 minutes.

### Note on checkpoints

The trained models (`baseline.pt` is about 45 MB, and the others are similar) are not included here, because GitHub warns on files over 50 MB and rejects files over 100 MB. Either regenerate them by setting `RETRAIN = True`, or host them with Git LFS or a GitHub Release and link them here.

## Notes on reproducibility

- Each configuration was run **once** (twice for the plain update). Robust-accuracy readings vary by about ±0.03, so small differences are not significant.
- Scale is small: 10,000 training images from two LSUN categories at 128×128.
- Hardened and no-growth models are compared at matched gradient steps, but the baseline did not get a matched-steps control. Part of the hardened model's StyleGAN2 gain may come from longer training.
- Pool B and BigGAN use ImageNet reals while training used LSUN, so content shift and generator shift are mixed on those pools.

## Future work

Repeat over several seeds; add a matched-steps baseline; train on several generator families at higher resolution; combine artifact features with frozen encoder features; calibrate the decision threshold after updates; and add **explainability**, such as per-branch Grad-CAM heatmaps and LIME, to show which regions and branches drive each decision.

## References

1. D. Park, H. Na, D. Choi, "Performance comparison and visualization of AI-generated-image detection methods", *IEEE Access*, 2024.
2. S.-Y. Wang et al., "CNN-generated images are surprisingly easy to spot... for now", *CVPR*, 2020 (ForenSynths).
3. M. Zhu et al., "GenImage: A million-scale benchmark for detecting AI-generated image", *NeurIPS*, 2023.
4. A. Madry et al., "Towards deep learning models resistant to adversarial attacks", *ICLR*, 2018 (PGD).
5. Z. Li and D. Hoiem, "Learning without forgetting", *IEEE TPAMI*, 2018.
6. A. A. Rusu et al., "Progressive neural networks", arXiv:1606.04671, 2016.

The full bibliography is in the report.

## Licence

Add a licence of your choice here (for example MIT for the code). Note that the datasets are covered by their own licences on Kaggle.
