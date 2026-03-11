import csv
import os
import json

def get_task_mapping_file(scenario, task_set):
    mapping_dir = "./debug/mappings"
    if not os.path.exists(mapping_dir):
        os.makedirs(mapping_dir)
    return os.path.join(mapping_dir, f"mapping_{scenario}_{task_set}.csv")

def build_and_save_mapping(tasks_dir, mapping_file):
    """Quét tasks folder, tạo map và lưu ra CSV"""
    mapping = {}
    print(f"📄 Creating new mapping file: {mapping_file}")
    files = [f for f in os.listdir(tasks_dir) if f.startswith("task_") and f.endswith(".json")]
    
    with open(mapping_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'user_id']) # Header
        
        for filename in files:
            try:
                idx = int(filename.split('_')[1].split('.')[0])
                with open(os.path.join(tasks_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    u_id = data.get('user_id')
                    if u_id:
                        mapping[u_id] = idx
                        writer.writerow([idx, u_id])
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return mapping

def load_mapping(mapping_file):
    """Load map từ file CSV có sẵn"""
    mapping = {}
    print(f"📖 Loading mapping from: {mapping_file}")
    with open(mapping_file, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mapping[row['user_id']] = int(row['index'])
    return mapping

if __name__ == "__main__":
    scenario = "classic"
    task_set = "yelp"
    tasks_directory = f"../dataset/task/{scenario}/{task_set}/tasks"
    mapping_file = get_task_mapping_file(scenario, task_set)

    # Kiểm tra: Nếu có file rồi thì load, chưa có thì tạo
    if os.path.exists(mapping_file):
        USER_TO_ID_MAP = load_mapping(mapping_file)
    else:
        USER_TO_ID_MAP = build_and_save_mapping(tasks_directory, mapping_file)