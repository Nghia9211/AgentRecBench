#!/bin/bash


set -e 

work_dir="."
cd "$work_dir"

DATASET="goodreads"                   # amazon | yelp | goodreads
SCENARIO="user_cold_start"

DATA_DIR="./data/${DATASET}/"
MODEL_PATH="./saved_models/${DATASET}_best_model.pt"
CANDIDATE_DIR="../../dataset/tasks5/${SCENARIO}/${DATASET}/tasks"
ITEMFILE="../../dataset/output_data_all/item.json"
INPUT_JSON_FILE="./data/ground_truth.json"

FAISS_DB_PATH="./faiss_dbs/${DATASET}"
GCN_PATH="./saved_models/${DATASET}_gcn_emb_remapped.pt"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"

GATING_MODEL_PATH="./saved_models/moe/${DATASET}_gating_model.pt"

RERANKER_MODE="llm"

RERANKER_TOP_LLM=15

# ── LLM (chỉ cần khi RERANKER_MODE = llm hoặc hybrid) ────────────────────
MODEL="${MODEL:-qwen-research}"
API_KEY="${API_KEY:-EMPTY}"
BASE_URL="YOUR_URL"

# ── Hyperparameters ───────────────────────────────────────────────────────
STAGE="test"
CANS_NUM=20
MAX_EPOCH=5
MAX_SAMPLES=100         # -1 = toàn bộ dataset
MAX_RETRY_NUM=5
SEED=303
MP=10
TEMPERATURE="0.0"
HIDDEN_SIZE=64

# ── Output paths ──────────────────────────────────────────────────────────
P_MODEL="SASRec_MoE"

OUTPUT_FILE="./output/${DATASET}_${SCENARIO}_moe/${P_MODEL}_${MODEL}_${SEED}_${TEMPERATURE}_mp${MP}_ep${MAX_EPOCH}.jsonl"
SAVE_REC_DIR="./output/${DATASET}_${SCENARIO}_moe_save/rec_${P_MODEL}_${MODEL}_ep${MAX_EPOCH}"
SAVE_USER_DIR="./output/${DATASET}_${SCENARIO}_moe_save/user_${P_MODEL}_${MODEL}_ep${MAX_EPOCH}"
RESULT_FILE="../../baseline/results/${SCENARIO}/evaluation_results_MoE_${DATASET}.json"

# ── Print config ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  MoE Early Fusion Pipeline"
echo "  Dataset:        $DATASET / $SCENARIO"
echo "  GCN:            $GCN_PATH"
echo "  FAISS:          $FAISS_DB_PATH"
echo "  Gating model:   $GATING_MODEL_PATH"
echo "  Reranker mode:  $RERANKER_MODE"
echo "  Max epoch:      $MAX_EPOCH"
echo "  Max samples:    $MAX_SAMPLES"
echo "  Workers (mp):   $MP"
echo "  Output:         $OUTPUT_FILE"
echo "============================================================"

export CUDA_VISIBLE_DEVICES=0

# ── Tạo output dirs nếu chưa có ──────────────────────────────────────────
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$SAVE_REC_DIR"
mkdir -p "$SAVE_USER_DIR"
mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run ───────────────────────────────────────────────────────────────────
python3 ./main_moe.py \
    --data_dir="$DATA_DIR" \
    --model_path="$MODEL_PATH" \
    --input_json_file="$INPUT_JSON_FILE" \
    --dataset="$DATASET" \
    --stage="$STAGE" \
    --cans_num=$CANS_NUM \
    --max_epoch=$MAX_EPOCH \
    --max_samples=$MAX_SAMPLES \
    --candidate_dir="$CANDIDATE_DIR" \
    --item_mapping_file="$ITEMFILE" \
    --faiss_db_path="$FAISS_DB_PATH" \
    --gcn_path="$GCN_PATH" \
    --embed_model_name="$EMBED_MODEL" \
    --gating_model_path="$GATING_MODEL_PATH" \
    --reranker_mode="$RERANKER_MODE" \
    --reranker_top_llm=$RERANKER_TOP_LLM \
    --model="$MODEL" \
    --api_key="$API_KEY" \
    --base_url="$BASE_URL" \
    --max_retry_num=$MAX_RETRY_NUM \
    --seed=$SEED \
    --mp=$MP \
    --temperature=$TEMPERATURE \
    --hidden_size=$HIDDEN_SIZE \
    --output_file="$OUTPUT_FILE" \
    --result_file="$RESULT_FILE" \
    --save_info \
    --save_rec_dir="$SAVE_REC_DIR" \
    --save_user_dir="$SAVE_USER_DIR"

echo ""
echo "============================================================"
echo "  Done. Results: $OUTPUT_FILE"
echo "  Summary:       $RESULT_FILE"
echo "============================================================"

read -p "Press Enter to continue..."