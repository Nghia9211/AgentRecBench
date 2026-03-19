import os
import json

def save_final_metrics(args, total_samples, h1_count, h3_count, h5_count):
    """Lưu kết quả đánh giá cuối cùng vào file JSON theo định dạng yêu cầu"""
    
    # Tính toán các tỷ lệ
    h1_rate = h1_count / total_samples if total_samples > 0 else 0
    h3_rate = h3_count / total_samples if total_samples > 0 else 0
    h5_rate = h5_count / total_samples if total_samples > 0 else 0
    
    # Average Hit Rate = trung bình cộng của 3 loại hit rate (theo ví dụ của bạn)
    avg_hit_rate = (h1_rate + h3_rate + h5_rate) / 3
    
    # Tạo cấu trúc JSON
    result_data = {
        "type": "recommendation",
        "metrics": {
            "top_1_hit_rate": h1_rate,
            "top_3_hit_rate": h3_rate,
            "top_5_hit_rate": h5_rate,
            "average_hit_rate": avg_hit_rate,
            "total_scenarios": total_samples,
            "top_1_hits": h1_count,
            "top_3_hits": h3_count,
            "top_5_hits": h5_count
        },
        "data_info": {
            "evaluated_count": total_samples,
            "original_simulation_count": total_samples,
            "original_ground_truth_count": total_samples
        }
    }


    try:
        # --- BỔ SUNG: Tự động tạo thư mục nếu chưa có ---
        output_dir = os.path.dirname(args.result_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # -----------------------------------------------

        with open(args.result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"\n📊 Đã lưu báo cáo chi tiết tại: {args.result_file}")
    except Exception as e:
        print(f"\n❌ Lỗi khi lưu file summary: {e}")