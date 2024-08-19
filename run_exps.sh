#!/bin/bash

# 定义常量
BATCH_SIZE=64
HIDDEN_SIZE=64
NUM_LAYERS=4
PATIENCE=5

# 遍历 missing-rate 和 max-missing-rate 的所有组合
for MISSING_RATE in 0.3 0.5 0.7 0.9; do
    for MAX_MISSING_RATE in 0.2 0.4 0.6 0.8; do
        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

        # 计算 SEQ_LENGTH
        PRODUCT=$(echo "$MISSING_RATE * $MAX_MISSING_RATE * 600" | bc)

        if (( $(echo "$PRODUCT > 200" | bc -l) )); then
            SEQ_LENGTH=256
        elif (( $(echo "$PRODUCT > 100" | bc -l) )); then
            SEQ_LENGTH=128
        else
            SEQ_LENGTH=64
        fi

        echo "Using SEQ_LENGTH=$SEQ_LENGTH"


        # 执行 Python 程序
        python /root/autodl-tmp/project/main.py \
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
