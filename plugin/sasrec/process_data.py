import json
import os
import pandas as pd
import collections
from tqdm import tqdm
from datetime import datetime

RAW_DIR          = '../../dataset/output_data_all'
OUTPUT_DIR       = './data'
GROUND_TRUTH_FILE = './ground_truth.json'
MAX_SEQ_LEN      = 50
MIN_INTERACTION  = 5


def get_normalized_timestamp(data, source):
    try:
        if source == 'amazon':
            return int(data.get('timestamp', 0))
        elif source == 'yelp':
            date_str = data.get('date')
            if date_str:
                return int(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").timestamp())
        elif source == 'goodreads':
            date_str = data.get('date_added') or data.get('date_updated')
            if date_str:
                return int(datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y").timestamp())
        return 0
    except Exception:
        return 0


def pad_seq(s, max_len, pad_val):
    """Keep most recent max_len items, left-pad with pad_val."""
    s = s[-max_len:]
    return [pad_val] * (max_len - len(s)) + s


def process_source(target_source):
    print(f"\n{'='*60}")
    print(f"  PROCESSING: {target_source.upper()}")
    print(f"{'='*60}")

    save_dir = os.path.join(OUTPUT_DIR, target_source)
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  STEP 1: Load items (cần trước để build raw_id_to_inner_id)
    # ------------------------------------------------------------------ #
    item_file = os.path.join(RAW_DIR, 'item.json')
    raw_id_to_inner_id = {}
    inner_id_to_raw_id = {}
    id2name    = {}
    item_count = 1
    title_key  = 'name' if target_source == 'yelp' else 'title'

    with open(item_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('source') == target_source:
                    raw_id = data.get('item_id')
                    if raw_id and raw_id not in raw_id_to_inner_id:
                        raw_id_to_inner_id[raw_id] = item_count
                        inner_id_to_raw_id[item_count] = raw_id
                        id2name[item_count] = data.get(title_key, 'Unknown').strip()
                        item_count += 1
            except Exception:
                continue

    print(f"[Items] Vocabulary size : {item_count - 1}")

    # ------------------------------------------------------------------ #
    #  STEP 2: Load reviews — build user_interactions & valid_users
    # ------------------------------------------------------------------ #
    review_file       = os.path.join(RAW_DIR, 'review.json')
    user_interactions = collections.defaultdict(list)
    valid_users       = set()   # users thực sự thuộc target_source
    review_count      = 0

    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('source') != target_source:
                    continue
                uid     = data.get('user_id')
                iid_raw = data.get('item_id')
                ts      = get_normalized_timestamp(data, target_source)
                if uid and iid_raw in raw_id_to_inner_id:
                    user_interactions[uid].append((ts, raw_id_to_inner_id[iid_raw], iid_raw))
                    valid_users.add(uid)
                    review_count += 1
            except Exception:
                continue

    print(f"[Reviews] Total interactions loaded : {review_count}")
    print(f"[Reviews] Unique valid users        : {len(valid_users)}")

    # ------------------------------------------------------------------ #
    #  STEP 3: Load groundtruth — CHỈ lấy users thuộc target_source
    # ------------------------------------------------------------------ #
    gt_map = {}
    if os.path.exists(GROUND_TRUTH_FILE):
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)

        gt_all   = {item["user_id"]: item["item_id"] for item in gt_list}
        gt_map   = {uid: iid for uid, iid in gt_all.items() if uid in valid_users}

        leaked   = set(gt_all.keys()) - valid_users
        print(f"\n[GT] Total entries in file          : {len(gt_all)}")
        print(f"[GT] Valid for {target_source:<12}      : {len(gt_map)}")
        if leaked:
            print(f"[GT] ⚠️  Filtered out (other sources) : {len(leaked)} users")
        else:
            print(f"[GT] ✅ No cross-source leakage detected")
    else:
        print(f"[GT] ⚠️  Ground truth file not found: {GROUND_TRUTH_FILE}")

    # ------------------------------------------------------------------ #
    #  STEP 4: Build train / val / test splits
    # ------------------------------------------------------------------ #
    train_data, val_data, test_data = [], [], []

    gt_found        = 0
    gt_not_in_items = 0

    all_test_user_ids = set(gt_map.keys())

    # --- Test users (từ groundtruth đã được filter) ---
    for uid in tqdm(all_test_user_ids, desc=f"[{target_source}] Processing test users"):
        target_raw_iid = gt_map[uid]
        interactions   = user_interactions.get(uid, [])
        interactions.sort(key=lambda x: x[0])

        raw_ids   = [x[2] for x in interactions]
        inner_ids = [x[1] for x in interactions]

        if target_raw_iid in raw_ids:
            idx        = raw_ids.index(target_raw_iid)
            seq_before = inner_ids[:idx]

            test_data.append({
                'uid':     uid,
                'seq':     pad_seq(seq_before, MAX_SEQ_LEN, 0),
                'len_seq': min(len(seq_before), MAX_SEQ_LEN),
                'next':    inner_ids[idx],
                'cold':    len(seq_before) == 0,
            })
            if len(seq_before) > 0:
                gt_found += 1
        else:
            # GT item không có trong review history của user này
            next_inner = raw_id_to_inner_id.get(target_raw_iid, -1)
            seq_before = inner_ids

            test_data.append({
                'uid':     uid,
                'seq':     pad_seq(seq_before, MAX_SEQ_LEN, 0),
                'len_seq': min(len(seq_before), MAX_SEQ_LEN),
                'next':    next_inner,
                'cold':    True,
            })
            gt_not_in_items += 1

    # --- Train / Val users (không nằm trong test) ---
    non_gt_users = [u for u in user_interactions if u not in all_test_user_ids]

    for uid in tqdm(non_gt_users, desc=f"[{target_source}] Processing train/val users"):
        interactions = user_interactions[uid]
        interactions.sort(key=lambda x: x[0])
        inner_ids = [x[1] for x in interactions]

        if len(inner_ids) < MIN_INTERACTION:
            continue

        val_data.append({
            'uid':     uid,
            'seq':     pad_seq(inner_ids[:-1], MAX_SEQ_LEN, 0),
            'len_seq': min(len(inner_ids) - 1, MAX_SEQ_LEN),
            'next':    inner_ids[-1],
        })
        if len(inner_ids) >= 3:
            train_data.append({
                'uid':     uid,
                'seq':     pad_seq(inner_ids[:-2], MAX_SEQ_LEN, 0),
                'len_seq': min(len(inner_ids) - 2, MAX_SEQ_LEN),
                'next':    inner_ids[-2],
            })

    # ------------------------------------------------------------------ #
    #  STEP 5: Save artifacts
    # ------------------------------------------------------------------ #
    with open(os.path.join(save_dir, 'id2name.txt'), 'w', encoding='utf-8') as f:
        for iid, name in id2name.items():
            f.write(f"{iid}::{name.replace(chr(10), ' ').replace(chr(13), ' ')}\n")

    with open(os.path.join(save_dir, 'id2rawid.txt'), 'w', encoding='utf-8') as f:
        for inner_id, raw_id in inner_id_to_raw_id.items():
            f.write(f"{inner_id}::{raw_id}\n")

    statis = pd.DataFrame({'seq_size': [MAX_SEQ_LEN], 'item_num': [item_count]})
    statis.to_pickle(os.path.join(save_dir, 'data_statis.df'))

    pd.DataFrame(train_data).to_pickle(os.path.join(save_dir, 'train_data.df'))
    pd.DataFrame(val_data).to_pickle(os.path.join(save_dir, 'Val_data.df'))
    pd.DataFrame(test_data).to_pickle(os.path.join(save_dir, 'Test_data.df'))

    # ------------------------------------------------------------------ #
    #  STEP 6: Summary
    # ------------------------------------------------------------------ #
    cold_count = sum(1 for d in test_data if d.get('cold'))
    print(f"\n✅ {target_source.upper()} complete:")
    print(f"   Train samples               : {len(train_data)}")
    print(f"   Val samples                 : {len(val_data)}")
    print(f"   Test samples total          : {len(test_data)}")
    print(f"     ├─ Classic (non-cold)     : {gt_found}")
    print(f"     ├─ Cold start             : {cold_count}")
    print(f"     └─ GT item not in history : {gt_not_in_items}")
    print(f"   Item vocab size             : {item_count - 1}")


if __name__ == "__main__":
    # for source in ['yelp', 'amazon', 'goodreads']:
    #     process_source(source)
    for source in ['amazon']:
        process_source(source)