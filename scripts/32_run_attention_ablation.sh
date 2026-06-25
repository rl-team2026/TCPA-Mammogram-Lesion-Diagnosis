#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_ROOT="data/processed/ddsm"
RUN_ROOT="outputs/ddsm_ablation_attn"
BIOMEDCLIP_DIR="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

ENV=(
  "PYTHONUNBUFFERED=1" "DATA_ROOT=$DATA_ROOT"
  "SINGLE_CSV=$DATA_ROOT/breast_cancer_train.csv"
  "PAIR_CSV=$DATA_ROOT/breast_cancer_train_cc_mlo_pair_train.csv"
  "VAL_PAIR_CSV=$DATA_ROOT/breast_cancer_train_cc_mlo_pair_val.csv"
  "TEST_PAIR_CSV=$DATA_ROOT/breast_cancer_test_cc_mlo_pair.csv"
  "EPOCHS=20" "BATCH_SIZE=16"
)

run() {
  local name="$1"; shift
  echo "== $(date) $name =="
  env "${ENV[@]}" "OUT_DIR=$RUN_ROOT/$name" \
    python -m cdf_vlm.cli.train_ddsm_joint_v31 \
    --single-csv "$DATA_ROOT/breast_cancer_train.csv" \
    --pair-csv "$DATA_ROOT/breast_cancer_train_cc_mlo_pair_train.csv" \
    --val-pair-csv "$DATA_ROOT/breast_cancer_train_cc_mlo_pair_val.csv" \
    --test-pair-csv "$DATA_ROOT/breast_cancer_test_cc_mlo_pair.csv" \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/$name" \
    --biomedclip-dir "$BIOMEDCLIP_DIR" \
    --epochs 20 --batch-size 16 --lambda-attn-reg 0.1 \
    --disable-consistency --disable-contrastive --disable-channel-fusion \
    "$@"
}

run a0_baseline
run a1_binary_mask   --use-attn-reg --attn-reg-mode binary
run a2_gaussian_s3   --use-attn-reg --attn-reg-mode gaussian --attn-reg-sigma 3
run a3_gaussian_s5   --use-attn-reg --attn-reg-mode gaussian --attn-reg-sigma 5
run a4_dilated       --use-attn-reg --attn-reg-mode dilated  --attn-reg-radius 3

echo "== $(date) Done =="
