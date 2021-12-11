#!/bin/bash

BASE_PROJ=aug2-cifar5m-base-models
BASE_RUN_ID=d9f09x6c
BASE_MODELS_DIR=/expanse/lustre/projects/csd697/nmallina/bootstrap/models/${BASE_PROJ}/${BASE_RUN_ID}
mkdir -p local_logs
RUN_ID_TAG="n=5000, aug=2, iid=False"
NUM_CONCURRENT=4

TMP_COMPLETED_DIR=/expanse/lustre/projects/csd697/nmallina/bootstrap/tmp_completed/${BASE_PROJ}/${BASE_RUN_ID}
mkdir -p $TMP_COMPLETED_DIR

idx=0
for d in ${BASE_MODELS_DIR}/*/ ; do
  base=$(basename -- "$d")
  # dir=$(dirname -- "$d")
  # echo "$d"
  # echo "$base $dir"
  echo "Launching $base, ${RUN_ID_TAG}"
  echo "$d/model.pt"
  python -m inftrain.finetune \
         --proj cf100-finetune-nov28 \
         --wandb_mode online \
         --dataset cifar100 \
         --run_id_tag "$base, ${RUN_ID_TAG}" \
         --k 64 \
         --pretrained $d/model.pt \
         --aug 4 \
         --lr 2e-5 \
         --epochs 10 \
         --fast \
         --opt adamw >local_logs/$base.log 2>&1 &
  idx=$((idx+1))
  if ((  $idx % $NUM_CONCURRENT == 0  )); then
    echo "waiting to complete.."
    wait
  fi
  mv $d $TMP_COMPLETED_DIR/$base
done
# mv $TMP_COMPLETED_DIR/* $BASE_MODELS_DIR
