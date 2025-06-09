#!/bin/bash
python main.py \
    --dataset=tiny_imagenet \
    --method=MFCL \
    --num_clients=100 \
    --n_tasks=10 \
    --global_round=20 \
    --frac=0.1 \
    --lr=0.1 \
    --batch_size=32 \
    --seed=1 \
    --path=data \
    --epochs=50