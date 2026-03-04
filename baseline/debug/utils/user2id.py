import csv

def load_user_to_idx_map(csv_path):
    mapping = {}
    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f) 
            for row in reader:
                u_id = row['user_id']
                idx = int(row['index'])
                mapping[u_id] = idx
        print(f"✅ Successfully loaded mapping from {csv_path}")
    except Exception as e:
        print(f"❌ Error loading CSV mapping: {e}")
    return mapping
