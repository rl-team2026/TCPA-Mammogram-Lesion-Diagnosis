#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-data/processed/ddsm}"
RUN_ROOT="${RUN_ROOT:-outputs/ddsm_ablation_v31}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BIOMEDCLIP_DIR="${BIOMEDCLIP_DIR:-external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"

ENV=(
  "PYTHONUNBUFFERED=1" "DATA_ROOT=$DATA_ROOT"
  "SINGLE_CSV=${SINGLE_CSV:-$DATA_ROOT/breast_cancer_train.csv}"
  "PAIR_CSV=${PAIR_CSV:-$DATA_ROOT/breast_cancer_train_cc_mlo_pair_train.csv}"
  "VAL_PAIR_CSV=${VAL_PAIR_CSV:-$DATA_ROOT/breast_cancer_train_cc_mlo_pair_val.csv}"
  "TEST_PAIR_CSV=${TEST_PAIR_CSV:-$DATA_ROOT/breast_cancer_test_cc_mlo_pair.csv}"
  "EPOCHS=$EPOCHS" "BATCH_SIZE=$BATCH_SIZE"
)

run() {
  local name="$1"; shift
  echo "=============================================="
  echo "== V3.1: $name  $(date) =="
  echo "=============================================="
  env "${ENV[@]}" "OUT_DIR=$RUN_ROOT/v31_$name" \
    python -m cdf_vlm.cli.train_ddsm_joint_v31 \
    --single-csv "${SINGLE_CSV:-$DATA_ROOT/breast_cancer_train.csv}" \
    --pair-csv "${PAIR_CSV:-$DATA_ROOT/breast_cancer_train_cc_mlo_pair_train.csv}" \
    --val-pair-csv "${VAL_PAIR_CSV:-$DATA_ROOT/breast_cancer_train_cc_mlo_pair_val.csv}" \
    --test-pair-csv "${TEST_PAIR_CSV:-$DATA_ROOT/breast_cancer_test_cc_mlo_pair.csv}" \
    --data-root "$DATA_ROOT" --output-dir "$RUN_ROOT/v31_$name" \
    --biomedclip-dir "$BIOMEDCLIP_DIR" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --lambda-contrastive 0.1 --lambda-consistency 0.05 \
    "$@"
}

# ══════════════════════════════════════════════════════════
# V3.1 Ablation: 6 conditions
# ══════════════════════════════════════════════════════════

# 1. Baseline: BiomedCLIP text encoder only (same as V3 best)
run baseline \
  --disable-consistency --disable-contrastive --disable-channel-fusion

# 2. + Channel-wise fusion (Improvement 2)
run channel_fusion \
  --disable-consistency --disable-contrastive

# 3. + Contrastive loss (Improvement 1)
run contrastive \
  --disable-consistency --disable-channel-fusion

# 4. + Both channel fusion + contrastive
run channel_and_contrastive \
  --disable-consistency

# 5. Full V3.1: all three + consistency
run full_v31

# 6. No text prompts (quantify text encoder contribution)
run no_text \
  --disable-text-prompts --disable-consistency \
  --disable-contrastive --disable-channel-fusion

echo ""
echo "== V3.1 done at $(date) =="
