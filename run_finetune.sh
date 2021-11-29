#!/bin/bash

BASE_MODELS_DIR=/expanse/lustre/projects/csd697/nmallina/bootstrap/models/d9f09x6c
mkdir -p local_logs
RUN_ID_TAG="n=5000, aug=2, iid=False"
NUM_CONCCURENT=2

idx=0
for d in ${BASE_MODELS_DIR}/*/ ; do
  base=$(basename -- "$d")
  # dir=$(dirname -- "$d")
  # echo "$d"
  # echo "$base $dir"
  echo "$base, ${RUN_ID_TAG}"
  echo "$d/model.pt"
  python -m inftrain.finetune \
         --proj cf100-finetune-nov28 \
         --wandb_mode online \
         --dataset cifar100 \
         --run_id_tag "$base, ${RUN_ID_TAG}" \
         --k 64 \
         --pretrained $d/model.pt \
         --aug 4 \
         --lr 1e-3 \
         --epochs 30 \
         --opt adamw >local_logs/$base.log 2>&1 &
  idx=$((idx+1))
  if $(($idx%$NUM_CONCCURENT)) ; do
    wait
  done
done
