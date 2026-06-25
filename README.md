# TPCA: Text-Prompt Guided Cross-View Multimodal Adaptation for Mammogram Lesion Diagnosis

Official implementation of **TPCA**, a multimodal framework for benign-malignant classification of breast lesions in dual-view (CC/MLO) mammography.

## Overview

TPCA builds upon the [BiomedCLIP](https://arxiv.org/abs/2303.00915) vision-language backbone and introduces a **pathology text-guided visual modulation mechanism** that uses BiomedCLIP's PubMedBERT text encoder to generate rich semantic prompt vectors. These vectors modulate visual features through **Feature-wise Linear Modulation (FiLM)**, enabling fine-grained attention to lesion-relevant image regions.

## Key Contributions

1. **Text-Guided Visual Modulation**: Full diagnostic descriptions are encoded via PubMedBERT into 512-dim semantic prompt vectors, which condition visual features channel-wise through a FiLM-based Prompt Attention Gate. This improves AUROC by **+2.46 pp** over keyword-based multi-hot encoding (0.7960 vs. 0.7714).

2. **Cross-View Consistency Loss**: A symmetric MSE loss enforces prediction agreement between CC and MLO single-view branches, emulating the radiologist's dual-view cross-validation practice.

3. **Systematic Ablation Study**: 7 ablation conditions on CBIS-DDSM quantify each component's marginal contribution, revealing that text semantic quality dominates performance while additional architectural complexity provides negligible or negative returns.

## Installation

```bash
# Clone the repository
git clone https://github.com/rl-team2026/TCPA-Mammogram-Lesion-Diagnosis.git
cd TCPA-Mammogram-Lesion-Diagnosis

# Install dependencies
pip install torch torchvision open_clip_torch tqdm pandas numpy psutil

# Download BiomedCLIP weights
mkdir -p external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
cd external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# Download from https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT-256-vit_base_patch16_224
# Required files: open_clip_config.json, open_clip_pytorch_model.bin, pubmedbert_config/
cd ../..
```

## Dataset

We use the **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) dataset. After preprocessing:

| Split   | Single-View Samples | CC/MLO Pairs |
|---------|---------------------|--------------|
| Train   | 2,771               | 1,018        |
| Test    | 698                 | 266          |

Preprocessing pipeline:
1. Resize images to 224×224 and convert to RGB format
2. Enhance lesion ROIs using annotation masks
3. Generate structured diagnostic text from annotations
4. Pair CC and MLO views by patient ID and breast laterality

## Model Architecture

```
BiomedCLIP (Visual) ──► Global + Local Features ──► GlobalLocalGate ──► FiLM Gate ──► Classifier
                                                                          ▲
BiomedCLIP (Text)  ──► Diagnostic Text ──► PubMedBERT ──► Prompt Vector ──┘

CC View ──┐
          ├──► 0.5·(f_cc + f_mlo) ──► Classifier ──► ŷ_pair
MLO View ─┘

CC-only ──► Classifier ──► ŷ_cc ──┐
                                  ├──► MSE(ŷ_cc, ŷ_mlo)  (Consistency)
MLO-only ─► Classifier ──► ŷ_mlo ──┘
```

**Core Components:**
- `BiomedCLIPTextPromptEncoder` — Encodes full diagnostic descriptions via PubMedBERT
- `PromptAttentionGate` — FiLM-based channel-wise modulation of visual features
- `GlobalLocalGate` — Adaptive fusion of global anatomical and local lesion features
- Cross-View Consistency Loss — Symmetric MSE between CC-only and MLO-only predictions

## Training

```bash
# Run V3.1 ablation experiments (6 conditions)
bash scripts/31_run_v31_ablation.sh

# Run attention regularization ablation (5 conditions)
bash scripts/32_run_attention_ablation.sh
```

Key hyperparameters:
- Batch size: 16
- Epochs: 20 (early stopping with patience=10)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- MixUp: α=0.2
- Label Smoothing: 0.1
- λ_single = λ_pair = 1.0, λ_consistency = 0.05

## Results

| Variant                      | Test AUROC | AUPRC | F1    | ECE   |
|------------------------------|-----------|-------|-------|-------|
| **TPCA (BiomedCLIP text)**   | **0.7960**| 0.7063| 0.6751| 0.0959|
| TPCA + consistency           | 0.7907    | 0.7096| 0.6840| 0.1604|
| w/o Text Encoder (keywords)  | 0.7714    | 0.7043| 0.6294| 0.1201|
| w/ Aligner                   | 0.7750    | 0.6938| 0.6291| 0.0729|
| w/ LoRA                      | 0.7211    | 0.6365| 0.5701| 0.1896|
| w/ Complex Fusion            | 0.7744    | 0.6929| 0.6291| 0.0680|
| w/ All Components            | 0.7696    | 0.6857| 0.6323| 0.1605|

## File Structure

```
TCPA-Mammogram-Lesion-Diagnosis/
├── cdf_vlm/
│   ├── multiview_v31.py         # Core TPCA model (V3.1)
│   ├── multiview_v2.py          # V2 modules (PromptAttentionGate)
│   ├── biomedclip.py            # BiomedCLIP model loader
│   ├── ddsm.py                  # DDSM dataset loader
│   ├── text_prompts.py          # Pathology keyword extraction
│   ├── metrics.py               # Evaluation metrics
│   ├── lora.py                  # Decoupled LoRA module
│   ├── training.py              # Training utilities
│   ├── config.py / io.py        # Configuration and I/O
│   └── cli/
│       └── train_ddsm_joint_v31.py  # V3.1 training entry point
├── scripts/
│   ├── 31_run_v31_ablation.sh   # Main ablation experiment script
│   └── 32_run_attention_ablation.sh  # Attention regularization ablation
├── paper/
│   └── figures/
│       └── architecture.png     # Model architecture diagram
├── push_to_github.sh            # Script to push to GitHub
└── README.md
```

## Citation

```bibtex
@article{tpca2025,
  title={Text-Prompt Guided Cross-View Multimodal Adaptation for Mammogram Lesion Diagnosis},
  author={},
  journal={},
  year={2025}
}
```

## License

MIT License.

## Acknowledgments

This work builds upon [BiomedCLIP](https://github.com/microsoft/BiomedCLIP) (Microsoft Research) and uses the [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) dataset.
