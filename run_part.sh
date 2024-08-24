#!/bin/bash

# 定义常量
BATCH_SIZE=32
HIDDEN_SIZE=64
SEQ_LENGTH=256
NUM_LAYERS=4
PATIENCE=10

cd /root/autodl-tmp/project/

for MISSING_RATE in 0.6 0.8; do
    for MAX_MISSING_RATE in 0.5 0.6; do
        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

        # 计算 SEQ_LENGTH
#        PRODUCT=$(echo "scale=2; $MISSING_RATE * $MAX_MISSING_RATE * 1160" | bc)

        # # 输出计算的 PRODUCT 值
        # echo "Calculated PRODUCT=$PRODUCT"

        # if (( $(echo "$PRODUCT <= 50" | awk '{print ($1 <= 50)}') )); then
        #     SEQ_LENGTH=64
        #     HIDDEN_SIZE=64
        # elif (( $(echo "$PRODUCT <= 85" | awk '{print ($1 <= 85)}') )); then
        #     SEQ_LENGTH=128
        #     HIDDEN_SIZE=128
        # elif (( $(echo "$PRODUCT <= 180" | awk '{print ($1 <= 180)}') )); then
        #     SEQ_LENGTH=256
        #     HIDDEN_SIZE=256
        # else
        #     SEQ_LENGTH=360
        #     HIDDEN_SIZE=360
        # fi

#        echo "Using SEQ_LENGTH=$SEQ_LENGTH"

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

# 关机
shutdown -h now
