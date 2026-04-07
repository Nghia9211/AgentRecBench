#!/bin/bash

work_dir="."
cd "$work_dir"

# --- Dataset ---
DATASET="goodreads"
SCENARIO="user_cold_start"

DATA_DIR="./data/${DATASET}/"
STAGE="test"
CANS_NUM=20
MAX_SAMPLES=100
MAX_EPOCH=5
MODEL_PATH="./saved_models/${DATASET}_best_model.pt"
CANDIDATE_DIR="../../dataset/tasks5/${SCENARIO}/${DATASET}/tasks"
ITEMFILE="../../dataset/output_data_all/item.json"
MAX_RETRY_NUM=5
SEED=303
MP=5
TEMPERATURE="0.0"
INPUT_JSON_FILE="./ground_truth.json"
HIDDEN_SIZE=64

USE_ARAG="--use_arag"
USE_HYBRID="--use_hybrid"
FAISS_DB_PATH="../storage/item_storage_${DATASET}"
GCN_PATH="../gcn/gcn_embedding/gcn_embeddings_3hop_${DATASET}.pt"
NLI_THRESHOLD="4.5"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

P_MODEL="SASRec"

MODEL="${MODEL:-qwen-small}"
API_KEY="${API_KEY:-EMPTY}"
BASE_URL="http://localhost:8036/v1"

OUTPUT_FILE="./output/${DATASET}_${SCENARIO}_${USE_ARAG}_${USE_HYBRID}_rec_eval/${P_MODEL}_${USE_ARAG}_${USE_HYBRID}_${MODEL}_${SEED}_${TEMPERATURE}_${MP}_${MAX_EPOCH}.jsonl"
SAVE_REC_DIR="./output/${DATASET}_${SCENARIO}_${USE_ARAG}_${USE_HYBRID}_rec_save/rec_${P_MODEL}_${USE_ARAG}_${USE_HYBRID}_${MODEL}_${MAX_EPOCH}"
SAVE_USER_DIR="./output/${DATASET}_${SCENARIO}_${USE_ARAG}_${USE_HYBRID}_rec_save/user_${P_MODEL}_${USE_ARAG}_${USE_HYBRID}_${MODEL}_${MAX_EPOCH}"
RESULT_FILE="../../baseline/results/${SCENARIO}/evaluation_results_AFL_${USE_ARAG}_${USE_HYBRID}_${DATASET}.json"

# --- Print config ---
echo "============================================================"
echo "  AFL + ARAG Integration"
echo "  Dataset:       $DATASET"
echo "  FAISS DB:      $FAISS_DB_PATH"
echo "  GCN Path:      $GCN_PATH"
echo "  NLI Threshold: $NLI_THRESHOLD"
echo "  Max Epoch:     $MAX_EPOCH"
echo "  Max Samples:   $MAX_SAMPLES"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=0

python3 ./main.py \
    --data_dir="$DATA_DIR" \
    --model_path="$MODEL_PATH" \
    --input_json_file="$INPUT_JSON_FILE" \
    --stage="$STAGE" \
    --cans_num=$CANS_NUM \
    --max_epoch=$MAX_EPOCH \
    --max_samples=$MAX_SAMPLES \
    --output_file="$OUTPUT_FILE" \
    --model="$MODEL" \
    --api_key="$API_KEY" \
    --base_url="$BASE_URL" \
    --candidate_dir="$CANDIDATE_DIR" \
    --max_retry_num=$MAX_RETRY_NUM \
    --seed=$SEED \
    --item_mapping_file="$ITEMFILE" \
    --mp=$MP \
    --temperature=$TEMPERATURE \
    --hidden_size=$HIDDEN_SIZE \
    --save_info \
    --save_rec_dir="$SAVE_REC_DIR" \
    --save_user_dir="$SAVE_USER_DIR" \
    --result_file="$RESULT_FILE" \
    $USE_ARAG \
    $USE_HYBRID \
    --faiss_db_path="$FAISS_DB_PATH" \
    --gcn_path="$GCN_PATH" \
    --nli_threshold="$NLI_THRESHOLD" \
    --embed_model_name="$EMBED_MODEL"

read -p "Press Enter to continue..."