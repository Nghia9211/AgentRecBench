import argparse
import os
import time
import json
import pandas as pd
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys

current_dir  = os.path.dirname(os.path.abspath(__file__))  # plugin/AFL2
parent_dir   = os.path.dirname(current_dir)                 # plugin
root_dir     = os.path.dirname(parent_dir)                  # AgentRecBench

sys.path.append(parent_dir)
sys.path.append(root_dir)

# --- Thêm baseline vào sys.path để tìm thấy websocietysimulator ---
baseline_dir = os.path.join(root_dir, 'baseline')
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)

from utils.dialogue_manager import recommend, error_handler
from utils.data_processor import load_candidate_map, load_item_name_map, prepare_merge_data
from AFL2.utils.save_result import save_final_metrics
from utils.rw_process import append_jsonl
from dataset.general_dataset import GeneralDataset
from utils.agent import UserModelAgent, RecAgent

# --- Import InteractionTool ---
from websocietysimulator.tools import CacheInteractionTool

finish_num = 0
total = 0
correct_hit1 = 0
correct_hit3 = 0
correct_hit5 = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_json_file', type=str, default='')
    parser.add_argument('--candidate_dir', type=str, default=None)
    parser.add_argument('--item_mapping_file', type=str, default=None)
    parser.add_argument('--stage', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--output_file', type=str, default='./output/dialogue_results.jsonl')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=4)
    parser.add_argument("--save_info", action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default='./output/rec_logs')
    parser.add_argument("--save_user_dir", type=str, default='./output/user_logs')
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--result_file', type=str, default='evaluation_summary.json')

    # ARAG args
    parser.add_argument('--use_arag', action='store_true')
    parser.add_argument('--faiss_db_path', type=str, default=None)
    parser.add_argument('--gcn_path', type=str, default=None)
    parser.add_argument('--nli_threshold', type=float, default=5.5)
    parser.add_argument('--embed_model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2')

    # --- Thêm arg cho raw data dir ---
    parser.add_argument('--raw_data_dir', type=str,
                        default=None,
                        help='Path to raw dataset dir chứa review.json, item.json, user.json')

    return parser.parse_args()


def setcallback(x):
    global finish_num, total, correct_hit1, correct_hit3, correct_hit5
    data_list, hit_at_n, args = x
    for step in data_list:
        append_jsonl(args.output_file, step)
    finish_num += 1
    if hit_at_n[1]: correct_hit1 += 1
    if hit_at_n[3]: correct_hit3 += 1
    if hit_at_n[5]: correct_hit5 += 1
    agent_tag = "ARAG" if getattr(args, 'use_arag', False) else "AFL"
    print(f"[{agent_tag}][{finish_num}/{total}] "
          f"Hit@1: {correct_hit1/finish_num*100:.2f}% | "
          f"Hit@3: {correct_hit3/finish_num*100:.2f}% | "
          f"Hit@5: {correct_hit5/finish_num*100:.2f}%")


def main(args):
    if args.use_arag:
        print("=" * 60)
        print("  MODE: AFL + ARAG (Agentic RAG) Integration")
        print(f"  FAISS DB:      {args.faiss_db_path}")
        print(f"  GCN Path:      {args.gcn_path}")
        print(f"  NLI Threshold: {args.nli_threshold}")
        print(f"  Raw Data Dir:  {args.raw_data_dir}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  MODE: Vanilla AFL (no ARAG)")
        print("=" * 60)

    dataset  = GeneralDataset(args, stage=args.stage)
    data_map = {str(d['id']): d for d in dataset}

    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        new_input_list = json.load(f)

    print(args.candidate_dir)
    candidate_map = load_candidate_map(args.candidate_dir)
    item_name_map = load_item_name_map(args.item_mapping_file)

    temp_args = argparse.Namespace(**vars(args))
    temp_args.model = 'sasrec_inference'
    sasrec_tool = UserModelAgent(temp_args, mode='prior_rec')

    merge_data_list, skipped = prepare_merge_data(
        new_input_list, data_map, candidate_map, item_name_map, sasrec_tool, args
    )

    if args.max_samples > 0:
        merge_data_list = merge_data_list[:args.max_samples]

    if args.use_arag:
    
        if not args.raw_data_dir:
            args.raw_data_dir = os.path.join(root_dir, 'dataset', 'output_data_all')
        args.raw_data_dir = os.path.abspath(args.raw_data_dir)

        if not os.path.exists(args.raw_data_dir):
            raise FileNotFoundError(
                f"[Main] raw_data_dir không tồn tại: {args.raw_data_dir}\n"
                f"Hãy truyền --raw_data_dir=<đường dẫn tuyệt đối đến thư mục chứa item.json>"
            )

        print(f"[Main] Initializing InteractionTool from {args.raw_data_dir} ...")
        interaction_tool = CacheInteractionTool(data_dir=args.raw_data_dir)
        print("[Main] InteractionTool ready.")

        # Load id2rawid: inner_id (SASRec int) → raw_id gốc (string)
        id2rawid = {}
        rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
        if os.path.exists(rawid_path):
            with open(rawid_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        id2rawid[int(parts[0])] = parts[1].strip()
            print(f"[Main] Loaded id2rawid.txt ({len(id2rawid)} entries).")
        else:
            print(f"[Main] WARNING: id2rawid.txt not found at {rawid_path}. "
                  f"Please run process_data.py first to generate it.")

        # Gắn vào từng data dict
        for d in merge_data_list:
            d['interaction_tool'] = interaction_tool
            d['id2rawid']         = id2rawid
            d['task_set']         = args.data_dir

    global total
    total = len(merge_data_list)
    print(f"Ready: {total} samples. Skipped: {skipped}")

    # 6. Run
    effective_mp = 1 if args.use_arag else args.mp
    if args.use_arag and args.mp > 1:
        print(f"[WARNING] ARAG mode: forcing mp=1 (was {args.mp}) for LangGraph compatibility.")

    if effective_mp <= 1:
        for data in tqdm(merge_data_list, desc="Processing"):
            try:
                result = recommend(data, args)
                setcallback(result)
            except Exception as e:
                error_handler(e)
    else:
        pool = multiprocessing.Pool(effective_mp)
        for data in merge_data_list:
            pool.apply_async(recommend, args=(data, args),
                             callback=setcallback, error_callback=error_handler)
        pool.close()
        pool.join()

    save_final_metrics(args, total, correct_hit1, correct_hit3, correct_hit5)


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    main(args)