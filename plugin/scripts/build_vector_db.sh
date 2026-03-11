#!/bin/bash

set -e

# Change Mode here
MODE="review"

PYTHON_SCRIPT="create_vector_db.py"
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=256
DATA_DIR="../data"
STORAGE_DIR="../storage"

if [ "$MODE" == "review" ]; then
    echo ">>> Selected Mode : Review <<<"
    echo "======================================================"
    echo "Building vector store for user reviews..."
    echo "======================================================"

    python $PYTHON_SCRIPT \
        --data_path "$DATA_DIR/review.json" \
        --save_path "$STORAGE_DIR/user_storage" \
        --embed_model "$EMBED_MODEL" \
        --batch_size $BATCH_SIZE

    echo ""
    echo "User review database created successfully in '$STORAGE_DIR/user_storage'."

elif [ "$MODE" == "item" ]; then
    echo ">>> Selected Mode : item <<<"
    echo "======================================================"
    echo "Building vector store for items..."
    echo "======================================================"

    python $PYTHON_SCRIPT \
        --data_path "$DATA_DIR/item.json" \
        --save_path "$STORAGE_DIR/item_storage" \
        --embed_model "$EMBED_MODEL" \
        --batch_size $BATCH_SIZE

    echo ""
    echo "Item database created successfully in '$STORAGE_DIR/item_storage'."



