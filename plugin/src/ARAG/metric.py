import os
import json
import csv
import threading
from datetime import datetime

class ARAGMetrics:
    def __init__(self, experiment_name="ARAG_Retrie"):
        self.experiment_name = experiment_name
        self.filename = None
        self.lock = threading.Lock()
        self.initialized = False  

    def _initialize_file(self):
        if not self.initialized:
            log_dir = "experiment_logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.filename = os.path.join(log_dir, f"stats_{self.experiment_name}_{timestamp_str}.csv")

            
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Task_Set", "Index", "Stage", "GT_ID", "Hit", "Total_Items", "Position"])
            self.initialized = True

    def log_hit(self, task_set, index, stage, gt_id, is_hit, total_items, position):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            self._initialize_file()
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, task_set, index, stage, gt_id, is_hit, total_items, position])


metrics_logger = ARAGMetrics()

def evaluate_hit_rate(index: int, stage: str, items: list, gt_folder: str, task_set: str = "unknown"):
    """Hàm tách biệt để kiểm tra ground truth"""
    file_path = os.path.join(gt_folder, f"groundtruth_{index}.json")
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            gt_id = str(gt_data.get("ground truth", ""))
    except Exception:
        return False

    current_ids = [
        str(item.get('item_id', '')) if isinstance(item, dict) else str(getattr(item, 'item_id', ''))
        for item in items
    ]

    is_hit = gt_id in current_ids
    position = current_ids.index(gt_id) + 1 if is_hit else -1

    metrics_logger.log_hit(task_set, index, stage, gt_id, is_hit, len(current_ids), position)
    
    status_icon = "✅" if is_hit else "❌"
    print(f"{status_icon} [{stage}] Index {index}: GT {gt_id} {'FOUND at pos ' + str(position) if is_hit else 'NOT FOUND'}")
    return is_hit