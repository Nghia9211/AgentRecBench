#!/bin/bash

# --- CẤU HÌNH ---
BUILD_SCRIPT="build_graph_lgcn.py"
TRAIN_SCRIPT="train.py"

DATASET="amazon"

BASE_PATH="graph_data"
USER_FILE="${BASE_PATH}/user_${DATASET}.json"
ITEM_FILE="${BASE_PATH}/item_${DATASET}.json"
REVIEW_FILE="${BASE_PATH}/review_${DATASET}.json"
GT_FILE="${BASE_PATH}/ground_truth_gcn_${DATASET}.json"

OUTPUT_DATA="gcn_graph/processed_graph_data_${DATASET}.pt"
OUTPUT_EMB="gcn_embedding/gcn_embeddings_3hop_${DATASET}.pt"

echo ""
echo "========================================================"
echo "       STEP 1: STARTING GRAPH BUILD PROCESS"
echo "========================================================"
echo "Input User:   $USER_FILE"
echo "Output Data:  $OUTPUT_DATA"
echo ""

# Chạy script build graph (Dùng python3 cho Linux)
python3 "$BUILD_SCRIPT" \
    --user_file "$USER_FILE" \
    --item_file "$ITEM_FILE" \
    --review_file "$REVIEW_FILE" \
    --gt_folder "$GT_FILE" \
    --output_file "$OUTPUT_DATA"

# Kiểm tra lỗi (Trong Linux dùng $? thay cho ERRORLEVEL)
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Build Graph Failed! Check if files exist."
    exit 1
fi

echo "[SUCCESS] Graph built successfully!"

# --- 2. PHASE TRAINING ---

EPOCHS=3000
BATCH_SIZE=1024
LR=0.001
REG=1e-3
DEVICE="cuda"

echo ""
echo "========================================================"
echo "       STEP 2: STARTING TRAINING MODEL"
echo "========================================================"
echo "Settings: Epochs=$EPOCHS - Device=$DEVICE"
echo ""

python3 "$TRAIN_SCRIPT" \
    --data_file "$OUTPUT_DATA" \
    --export_file "$OUTPUT_EMB" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --reg $REG \
    --device "$DEVICE" \
    --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Training failed or interrupted!"
    exit 1
fi

echo "========================================================"
echo "       ALL PROCESS COMPLETED SUCCESSFULLY"
echo "========================================================"