#!/bin/bash

# 定义常量
BATCH_SIZE=64
HIDDEN_SIZE=128
SEQ_LENGTH=360
NUM_LAYERS=6
PATIENCE=5

cd /root/autodl-tmp/base/

for MISSING_RATE in 0.80; do
    for MAX_MISSING_RATE in 0.10 0.15 0.20 0.25; do
        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

        # 执行 Python 程序
        python /root/autodl-tmp/base/main.py \
            --batch-size $BATCH_SIZE \
            --hidden-size $HIDDEN_SIZE \
            --num-layers $NUM_LAYERS \
            --seq-length $SEQ_LENGTH \
            --patience $PATIENCE \
            --missing-rate $MISSING_RATE \
            --max-missing-rate $MAX_MISSING_RATE

        # 检查 Python 脚本的退出状态
        if [ $? -ne 0 ]; then
            echo "Error occurred while running the script with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"
            exit 1
        fi
    done
done

# 关机
#shutdown -h now
