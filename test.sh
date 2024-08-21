#!/bin/bash
echo "Testing"

for MISSING_RATE in 0.2 0.4 0.6 0.8; do
    for MAX_MISSING_RATE in 0.10 0.15 0.20 0.25; do
        echo "Running with missing-rate=$MISSING_RATE and max-missing-rate=$MAX_MISSING_RATE"

        # 计算 SEQ_LENGTH
        PRODUCT=$(echo "scale=2; $MISSING_RATE * $MAX_MISSING_RATE * 1200" | bc)

        # 输出计算的 PRODUCT 值
        echo "Calculated PRODUCT=$PRODUCT"

        if (( $(echo "$PRODUCT <= 50" | awk '{print ($1 <= 50)}') )); then
            SEQ_LENGTH=64
            HIDDEN_SIZE=64
        elif (( $(echo "$PRODUCT <= 85" | awk '{print ($1 <= 85)}') )); then
            SEQ_LENGTH=128
            HIDDEN_SIZE=128
        elif (( $(echo "$PRODUCT <= 180" | awk '{print ($1 <= 180)}') )); then
            SEQ_LENGTH=256
            HIDDEN_SIZE=256
        else
            SEQ_LENGTH=360
            HIDDEN_SIZE=360
        fi

        # 输出 SEQ_LENGTH
        echo "Using SEQ_LENGTH=$SEQ_LENGTH"
        echo "Using HIDDEN_SIZE=$HIDDEN_SIZE"

        echo "--------------------"
    done
done