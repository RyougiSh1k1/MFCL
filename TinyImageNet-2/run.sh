#!/bin/bash
# Run with 6 tasks, ~30 classes per task, and 1000 samples per task
python main.py \
    --dataset=tiny_imagenet \
    --method=MFCL \
    --num_clients=100 \
    --n_tasks=6 \
    --samples_per_task=1000 \
    --global_round=20 \
    --frac=0.1 \
    --lr=0.1 \
    --batch_size=32 \
    --seed=1 \
    --path=data \
    --epochs=50