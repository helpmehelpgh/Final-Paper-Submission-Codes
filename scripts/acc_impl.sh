#!/bin/bash

mkdir -p results/acc_final

nohup python scripts/acc_impl.py \
  --data_dir /data/CPE_487-587/ACCDataset \
  --output_dir results/acc_final \
  --k 10 \
  --sample_size 300000 \
  --test_size 0.2 \
  --epochs 20 \
  --batch_size 256 \
  --lr 0.001 \
  --num_workers 2 \
  > results/acc_final/train.log 2>&1 &