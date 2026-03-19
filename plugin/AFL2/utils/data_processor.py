import os
import json
from tqdm import tqdm

def load_candidate_map(candidate_dir):
    candidate_map = {}
    if not candidate_dir or not os.path.exists(candidate_dir):
        return candidate_map
    
    candidate_files = [f for f in os.listdir(candidate_dir) if f.startswith('task_') and f.endswith('.json')]
    for file_name in tqdm(candidate_files, desc="Loading Candidates"):
        try:
            with open(os.path.join(candidate_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                candidate_map[str(entry.get('user_id'))] = entry.get('candidate_list', [])
        except: continue
    return candidate_map

def load_item_name_map(mapping_file):
    name_map = {}
    if not mapping_file or not os.path.exists(mapping_file):
        return name_map
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                name = item.get('title') or item.get('name') or "Unknown"
                name_map[str(item.get('item_id')).strip()] = name.strip()
            except: continue
    return name_map
def prepare_merge_data(new_input_list, data_map, candidate_map, item_id_to_name_map, user_agent_sasrec, args):
    merge_data_list = []
    skipped = 0
    padding_id = user_agent_sasrec.item_num 

    for entry in tqdm(new_input_list, desc="Merging Data"):
        user_id = str(entry.get('user_id'))
        gt_item_id = str(entry.get('item_id', '')).strip()
        
        # --- DEBUG GROUND TRUTH ---
        name_from_map = item_id_to_name_map.get(gt_item_id)
        name_from_model = user_agent_sasrec.id2name.get(int(gt_item_id) if gt_item_id.isdigit() else -1)
        
        if not name_from_map and not name_from_model:
            print(f"\n[CRITICAL] Ground Truth Item {gt_item_id} (User: {user_id}) KHÔNG TỒN TẠI trong item.json lẫn id2name.txt!")
        
        gt_item_name = name_from_map or name_from_model or gt_item_id

        if user_id not in data_map:
            data = {'id': user_id, 'uid': user_id, 'seq': [padding_id] * user_agent_sasrec.seq_size, 'seq_str': "Empty History", 'len_seq': 0, 'seq_unpad': []}
        else:
            data = data_map[user_id].copy()

        data['correct_answer'] = gt_item_name
        
        if user_id in candidate_map:
            new_ids, new_names = [], []
            for rid in candidate_map[user_id]:
                rid_str = str(rid).strip()
                
                # Tìm tên
                name = item_id_to_name_map.get(rid_str) or user_agent_sasrec.id2name.get(int(rid_str) if rid_str.isdigit() else -1) or rid_str
                
                # Tìm ID trong mô hình SASRec
                iid = user_agent_sasrec.name2id.get(name)
                
                # --- DEBUG CANDIDATES ---
                if iid is None:
                    # ĐÂY LÀ NGUYÊN NHÂN GÂY LỖI: Item có trong danh sách ứng viên nhưng mô hình không biết nó là gì
                    print(f"[MISSING ITEM] Item '{name}' (ID: {rid_str}) không có trong từ điển Model. Sẽ bị loại bỏ khỏi cans.")
                    continue # Bỏ qua item này, không thêm vào new_ids
                
                if iid >= user_agent_sasrec.item_num:
                    print(f"[INVALID ID] Item '{name}' có ID {iid} >= item_num {user_agent_sasrec.item_num}. Sẽ bị loại bỏ.")
                    continue

                new_names.append(name)
                new_ids.append(iid)
            
            # Kiểm tra nếu sau khi lọc không còn ứng viên nào
            if len(new_ids) == 0:
                print(f"[WARNING] User {user_id} không có ứng viên nào hợp lệ sau khi lọc!")
                skipped += 1
                continue

            data['cans'] = new_ids
            data['cans_str'] = args.sep.join(new_names)
            data['len_cans'] = len(new_ids)
            data['cans_name'] = new_names
        else:
            skipped += 1
            continue

        try:
            data['prior_answer'] = user_agent_sasrec.model_generate(data['seq'], data['len_seq'], data['cans'])
            merge_data_list.append(data)
        except Exception as e:
            # In chi tiết lỗi tại đây
            print(f"\n[ERROR] Lỗi tại model_generate cho User {user_id}: {e}")
            print(f"Dữ liệu gây lỗi - Cans: {data.get('cans')}, Item_num: {user_agent_sasrec.item_num}")
            skipped += 1
            
    return merge_data_list, skipped