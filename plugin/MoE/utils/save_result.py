import os
import json

def save_final_metrics(args, total_samples, h1_count, h3_count, h5_count, ndcg5_score):
    
    h1_rate = h1_count / total_samples if total_samples > 0 else 0
    h3_rate = h3_count / total_samples if total_samples > 0 else 0
    h5_rate = h5_count / total_samples if total_samples > 0 else 0
    
    avg_hit_rate = (h1_rate + h3_rate + h5_rate) / 3
    
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
            "top_5_hits": h5_count,
            "ndcg@5": ndcg5_score,
        },
        "data_info": {
            "evaluated_count": total_samples,
            "original_simulation_count": total_samples,
            "original_ground_truth_count": total_samples
        }
    }


    try:
        output_dir = os.path.dirname(args.result_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"\n📊 Save Detail Results in File : {args.result_file}")
    except Exception as e:
        print(f"\n❌ Error while saving file, summary: {e}")