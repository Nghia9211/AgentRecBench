
#!/bin/bash

# 1. Khai báo mảng chứa tên các dataset bạn muốn chạy
# DATASETS=("amazon" "yelp" "goodreads")
DATASETS=("amazon" "yelp")

# 2. Khai báo các tham số dùng chung để dễ chỉnh sửa sau này
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE=256

echo "Bắt đầu tiến trình build FAISS cho ${#DATASETS[@]} datasets..."

# 3. Vòng lặp duyệt qua từng dataset
for ds in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "🚀 Đang khởi tạo FAISS Builder cho dataset: ${ds^^}"
    echo "=========================================================="

    # Gọi script python tương ứng với tên dataset
    python3 "build_faiss_${ds}.py" \
        --data_path "../gcn/graph_data/item_${ds}.json" \
        --save_path "../MoE/faiss_dbs/${ds}_rich" \
        --embed_model "$EMBED_MODEL" \
        --batch_size $BATCH_SIZE

    # Kiểm tra mã lỗi (exit code) của lệnh python vừa chạy
    if [ $? -eq 0 ]; then
        echo "✅ Đã build xong FAISS index cho: $ds"
    else
        echo "❌ Có lỗi xảy ra khi build dataset: $ds"
        # Bỏ dấu # ở dòng dưới nếu bạn muốn toàn bộ script dừng lại ngay lập tức khi 1 cái bị lỗi
        # exit 1 
    fi
    echo ""
done

echo "🎉 Hoàn tất toàn bộ tiến trình Build FAISS!"