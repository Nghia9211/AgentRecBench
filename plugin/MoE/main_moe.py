"""
main_moe.py
────────────
Mode: MULTI-ROUND MoE (Feedback Loop 4.0 - Soft Penalty)
"""

import argparse
import os
import sys
import json
import math
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
root_dir    = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(root_dir)

baseline_dir = os.path.join(root_dir, 'baseline')
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from utils.data_processor import (
    load_candidate_map, load_item_name_map,
    prepare_merge_data, build_candidate_order,
)
from AFL2.utils.save_result import save_final_metrics
from utils.rw_process import append_jsonl
from dataset.general_dataset import GeneralDataset
from utils.agent import UserModelAgent
from moe_rec_agent import MoERecAgent
from websocietysimulator.tools import CacheInteractionTool


def get_args():
    parser = argparse.ArgumentParser(description="MoE Early Fusion — standalone runner")
    parser.add_argument('--data_dir',          type=str, required=True)
    parser.add_argument('--model_path',        type=str, required=True)
    parser.add_argument('--input_json_file',   type=str, required=True)
    parser.add_argument('--candidate_dir',     type=str, default=None)
    parser.add_argument('--item_mapping_file', type=str, default=None)
    parser.add_argument('--raw_data_dir',      type=str, default=None)
    parser.add_argument('--stage',             type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--dataset',           type=str, default='amazon', choices=['amazon', 'yelp', 'goodreads'])
    parser.add_argument('--cans_num',  type=int, default=20)
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--faiss_db_path', type=str, default=None)
    parser.add_argument('--gcn_path',      type=str, default=None)
    parser.add_argument('--embed_model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--gating_model_path', type=str, default=None)
    parser.add_argument('--reranker_mode', type=str, default='embed_only', choices=['embed_only', 'llm', 'hybrid'])
    parser.add_argument('--reranker_top_llm', type=int, default=15)
    parser.add_argument('--rerank_only', action='store_true')
    parser.add_argument('--model',       type=str, default='qwen-small')
    parser.add_argument('--api_key',     type=str, default=None)
    parser.add_argument('--base_url',    type=str, default='http://localhost:8036/v1')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--mp',             type=int, default=4)
    parser.add_argument('--seed',           type=int, default=333)
    parser.add_argument('--max_samples',    type=int, default=-1)
    parser.add_argument('--max_retry_num',  type=int, default=5)
    parser.add_argument('--hidden_size',    type=int, default=64)
    parser.add_argument('--dropout',        type=float, default=0.1)
    parser.add_argument('--sep',            type=str, default=', ')
    parser.add_argument('--output_file',   type=str, default='./output/moe_results.jsonl')
    parser.add_argument('--result_file',   type=str, default='./output/moe_evaluation_summary.json')
    parser.add_argument('--save_info',     action='store_true')
    parser.add_argument('--save_rec_dir',  type=str, default='./output/moe_rec_logs')
    parser.add_argument('--save_user_dir', type=str, default='./output/moe_user_logs')
    return parser.parse_args()

def recommend_moe(data: dict, args) -> tuple:
    import time
    from utils.regular_function import split_user_response
    
    user_id    = data.get('id')
    llm_client = None
    
    if getattr(args, 'reranker_mode', 'embed_only') in ['llm', 'hybrid']:
        try:
            from langchain_openai import ChatOpenAI
            llm_client = ChatOpenAI(model=args.model, openai_api_key=args.api_key or "EMPTY", openai_api_base=args.base_url, temperature=args.temperature, max_retries=3)
        except ImportError:
            pass

    rec_agent  = MoERecAgent(args, llm=llm_client)
    shared     = rec_agent.get_shared_sasrec()
    user_agent = UserModelAgent(args, shared_sasrec=shared)

    flag          = False
    epoch         = 1
    rec_item_list = []
    new_data_list = []
    hit_at_n      = {1: False, 3: False, 5: False, 'rank': None}

    # Chứa danh sách bị trừ điểm
    rejected_items: list = []
    _invalid_reasons = {'', 'Could not parse user response.', 'Fallback: MoE pipeline unavailable.', 'Fallback: recommendation agent unavailable.'}

    while not flag and epoch <= args.max_epoch:
        prefix = f"[MoE][User {user_id}][Round {epoch}]"

        max_retries      = 3
        rec_reason       = None
        current_rec_list = []
        debug_info       = {}

        for attempt in range(max_retries):
            try:
                rec_reason, current_rec_list, debug_info = rec_agent.act(
                    data           = data,
                    epoch          = epoch,
                    rejected_items = rejected_items, # Truyền xuống để phạt mềm
                )
            except Exception as e:
                print(f"{prefix} act() error: {e}, retry {attempt+1}/{max_retries}")
                time.sleep(1)
                continue

            if rec_reason and current_rec_list:
                rec_item_list = current_rec_list
                break
        else:
            rec_item_list = data.get('cans_name', [])[:5]
            rec_reason    = "Fallback: MoE pipeline unavailable."
            debug_info    = {'error': 'Fallback trigger - All retries failed'}

        max_user_retries    = 3
        user_agent_response = None
        user_reason         = None

        for attempt in range(max_user_retries):
            user_agent_response = user_agent.act(data, rec_reason, rec_item_list)
            user_reason, flag   = split_user_response(user_agent_response)
            if user_reason is not None and flag is not None:
                break
            time.sleep(1)
        else:
            user_reason = "Could not parse user response."
            flag        = False

        user_reason_clean = (user_reason or '').strip()
        
        # ── Cập nhật danh sách phạt mềm ───────────────────────────────────────
        if not flag and user_reason_clean and user_reason_clean not in _invalid_reasons:
            for item in rec_item_list: 
                if item not in rejected_items:
                    rejected_items.append(item)
            print(f"{prefix} [FL 4.0] Items penalized for next round: {len(rejected_items)}")
        
        rec_res = (f"Reason: {rec_reason}\nItems: {', '.join(current_rec_list[:5])}")
        gt_name = data.get('correct_answer', '').strip()

        new_data_list.append({
            'id':              str(user_id),
            'epoch':           epoch,
            'gt_item':         gt_name,
            'rec_res':         rec_res,
            'user_res':        user_agent_response,
            'rec_items':       rec_item_list,
            'penalized_items': list(rejected_items),
            'flag':            flag,
            'pipeline':        'moe',
            'debug_rerank':    debug_info,
        })

        gt_name_lower       = gt_name.lower()
        current_top_n_lower = [item.lower().strip() for item in rec_item_list]
        print(f"{prefix} Rank list : {rec_item_list}")
        print(f"{prefix} GT item   : {gt_name}")

        if flag or epoch == args.max_epoch:
            if gt_name_lower in current_top_n_lower:
                rank             = current_top_n_lower.index(gt_name_lower) + 1
                hit_at_n['rank'] = rank
                if rank <= 1: hit_at_n[1] = True
                if rank <= 3: hit_at_n[3] = True
                if rank <= 5: hit_at_n[5] = True
                print(f"{prefix} Hit! GT rank={rank}")
            else:
                print(f"{prefix} Miss — GT not in final rec list.")
            if flag: break

        if user_reason_clean and user_reason_clean not in _invalid_reasons:
            memory_info = {
                'epoch':         epoch,
                'rec_reason':    rec_reason,
                'rec_item_list': rec_item_list,
                'user_reason':   user_reason,
            }
            rec_agent.update_memory(memory_info)
            user_agent.update_memory(memory_info)

        epoch += 1

    return new_data_list, hit_at_n, args, []


def error_handler(e):
    import traceback
    traceback.print_exc()

def make_counters(manager):
    return {
        'finish_num': manager.Value('i', 0), 'correct_hit1': manager.Value('i', 0),
        'correct_hit3': manager.Value('i', 0), 'correct_hit5': manager.Value('i', 0),
        'total_ndcg5': manager.Value('d', 0.0), 'total': manager.Value('i', 0),
        'lock': manager.Lock(),
    }

def setcallback_safe(result, counters, args):
    data_list, hit_at_n, _args, _ = result
    for step in data_list: append_jsonl(args.output_file, step)

    with counters['lock']:
        counters['finish_num'].value += 1
        if hit_at_n.get(1): counters['correct_hit1'].value += 1
        if hit_at_n.get(3): counters['correct_hit3'].value += 1
        if hit_at_n.get(5): counters['correct_hit5'].value += 1
        rank = hit_at_n.get('rank')
        if rank is not None and rank <= 5:
            counters['total_ndcg5'].value += 1.0 / math.log2(rank + 1)
        fn, tot = counters['finish_num'].value, counters['total'].value
        h1, h3, h5 = counters['correct_hit1'].value, counters['correct_hit3'].value, counters['correct_hit5'].value
        ndcg = counters['total_ndcg5'].value

    print(f"[MoE][{fn}/{tot}] Hit@1: {h1/fn*100:.2f}% | Hit@3: {h3/fn*100:.2f}% | Hit@5: {h5/fn*100:.2f}% | NDCG@5: {ndcg/fn:.4f}", flush=True)


def main(args):
    args.use_moe  = True
    args.use_arag = False
    
    dataset  = GeneralDataset(args, stage=args.stage)
    data_map = {str(d['id']): d for d in dataset}
    with open(args.input_json_file, 'r', encoding='utf-8') as f: new_input_list = json.load(f)
    candidate_map = load_candidate_map(args.candidate_dir)
    item_name_map = load_item_name_map(args.item_mapping_file)

    import argparse as _ap
    temp_args       = _ap.Namespace(**vars(args))
    temp_args.model = 'sasrec_inference'
    sasrec_tool     = UserModelAgent(temp_args, mode='prior_rec')

    merge_data_list, skipped = prepare_merge_data(new_input_list, data_map, candidate_map, item_name_map, sasrec_tool, args)

    if not args.raw_data_dir: args.raw_data_dir = os.path.join(root_dir, 'dataset', 'output_data_all')
    args.raw_data_dir = os.path.abspath(args.raw_data_dir)
    interaction_tool = CacheInteractionTool(data_dir=args.raw_data_dir)

    id2rawid = {}
    rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
    if os.path.exists(rawid_path):
        with open(rawid_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2: id2rawid[int(parts[0])] = parts[1].strip()

    for d in merge_data_list:
        d['interaction_tool'] = interaction_tool
        d['id2rawid']         = id2rawid

    candidate_order = build_candidate_order(args.candidate_dir)
    if candidate_order: merge_data_list.sort(key=lambda d: candidate_order.get(str(d['id']), float('inf')))
    if args.max_samples > 0: merge_data_list = merge_data_list[:args.max_samples]

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    total = len(merge_data_list)
    effective_workers = max(1, args.mp)

    with multiprocessing.Manager() as manager:
        counters = make_counters(manager)
        counters['total'].value = total
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(recommend_moe, data, args): data for data in merge_data_list}
            for future in tqdm(as_completed(futures), total=total, desc="Processing (MoE)"):
                try: setcallback_safe(future.result(), counters, args)
                except Exception as e: error_handler(e)

        final_hit1, final_hit3, final_hit5, final_ndcg = counters['correct_hit1'].value, counters['correct_hit3'].value, counters['correct_hit5'].value, counters['total_ndcg5'].value

    save_final_metrics(args, total, final_hit1, final_hit3, final_hit5, final_ndcg)
    print(f"\n[Main] Results saved to {args.output_file}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = get_args()
    random.seed(args.seed)
    main(args)