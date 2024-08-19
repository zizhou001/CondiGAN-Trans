#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate base

# Run the Python script in the background
python main.py \
    --batch-size=64 \
    --hidden-size=64 \
    --num-layers=6 \
    --seq-length=256 \
    --patience=5 \
    --missing-rate=0.9 \
    --max-missing-rate=0.7 \

