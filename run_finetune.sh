#!/bin/bash

BASE_MODELS_DIR=/expanse/lustre/projects/csd697/nmallina/bootstrap/models
RUN_ID_TAG="n=5000, aug=2, iid=False"

for d in ${BASE_MODELS_DIR}/*/ ; do
  base=$(basename -- "$d")
  # dir=$(dirname -- "$d")
  # echo "$d"
  # echo "$base $dir"
  echo "$base, ${RUN_ID_TAG}"
  echo "$d/model.pt"
  python -m inftrain.finetune \
         --proj cf100-finetune-nov28 \
         --wandb_mode offline \
         --run_id_tag "$base, ${RUN_ID_TAG}" \
         --k 64 \
         --pretrained $d/model.pt \
         --aug 4 \
         --lr 1e-3 \
         --epochs 30 \
         --opt adamw
done
