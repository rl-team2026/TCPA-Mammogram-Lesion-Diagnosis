#!/bin/bash
# Push TPCA code to GitHub
# Usage: bash push_to_github.sh

REPO="https://github.com/rl-team2026/TCPA-Mammogram-Lesion-Diagnosis.git"
USERNAME="rl-team2026"
TOKEN="Allinrl_2026"

echo "Initializing git repo..."
git init
git add -A
git commit -m "Initial release: TPCA code for mammogram lesion diagnosis

- BiomedCLIP text encoder with FiLM-based visual modulation
- Cross-view consistency loss for CC/MLO dual-view mammograms
- Systematic ablation experiments on CBIS-DDSM
- Best test AUROC: 0.7960"

git remote add origin "$REPO"
git push -u "https://${USERNAME}:${TOKEN}@github.com/rl-team2026/TCPA-Mammogram-Lesion-Diagnosis.git" main --force
echo "Done!"
