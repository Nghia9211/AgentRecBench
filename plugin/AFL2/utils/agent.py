import os
import time
import argparse
import json
import torch
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model

sys.modules['SASRecModules_ori'] = model

from utils.regular_function import split_user_response, split_rec_reponse
from utils.rw_process import append_jsonl, write_jsonl, read_jsonl
from utils.api_request import api_request
from utils.model import SASRec

class RecAgent:
    def __init__(self, args, mode='prior_rec'):
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

    def load_prompt(self):
        if self.mode =='prior_rec':
            if 'amazon' in self.args.data_dir:
                from constant.amazon_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            elif 'goodreads' in self.args.data_dir:
                from constant.goodreads_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            elif 'yelp' in self.args.data_dir:
                from constant.yelp_prior_model_prompt import rec_system_prompt, rec_user_prompt, rec_memory_system_prompt, rec_memory_user_prompt, rec_build_memory
            else:
                raise ValueError("Invalid mode: {}".format(self.args.data_dir))
            self.rec_system_prompt = rec_system_prompt
            self.rec_user_prompt = rec_user_prompt
            self.rec_memory_system_prompt = rec_memory_system_prompt
            self.rec_memory_user_prompt = rec_memory_user_prompt
            self.rec_build_memory = rec_build_memory
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def act(self, data, reason=None, item=None):
        if self.mode =='prior_rec':
            try:
                if len(self.memory) == 0:
                    system_prompt = self.rec_system_prompt
                    user_prompt = self.rec_user_prompt.format(data['seq_str'], data['len_cans'], data['cans_str'], data['prior_answer'])
                else:
                    system_prompt = self.rec_memory_system_prompt
                    user_prompt = self.rec_memory_user_prompt.format(data['seq_str'], data['len_cans'], data['cans_str'], '\n'.join(self.memory))
                response = api_request(system_prompt, user_prompt, self.args)
                print(f"Response : {response} ")
                return response
            except Exception as e:
                print(f"LỖI KHI GỌI MODEL TRONG REC_AGENT (User ID: {data.get('id', 'N/A')}): {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def build_memory(self, info):
        rec_item_str = ', '.join(info['rec_item_list']) if isinstance(info.get('rec_item_list'), list) else info.get('rec_item')
        return self.rec_build_memory.format(info['epoch'], rec_item_str, info['rec_reason'], info['user_reason'])

    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]


class UserModelAgent:
    def __init__(self, args, mode='prior_rec', shared_sasrec=None):
        """
        Args:
            shared_sasrec: dict với các key 'model', 'id2name', 'name2id',
                           'id2rawid', 'seq_size', 'item_num', 'device'.
                           Nếu được truyền vào, bỏ qua load_model() và
                           build_id2name() để tránh load SASRec lần 2.
        """
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

        if shared_sasrec is not None:
            # Tái sử dụng SASRec đã load từ ARAGRecAgent
            self.model    = shared_sasrec['model']
            self.id2name  = shared_sasrec['id2name']
            self.name2id  = shared_sasrec['name2id']
            self.id2rawid = shared_sasrec['id2rawid']
            self.seq_size = shared_sasrec['seq_size']
            self.item_num = shared_sasrec['item_num']
            self.device   = shared_sasrec['device']
            print("[UserModelAgent] Reusing shared SASRec — skipping reload.")
        else:
            # Load bình thường khi dùng độc lập (không có ARAG)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.id2name  = dict()
            self.name2id  = dict()
            self.id2rawid = dict()
            self.build_id2name()
            self.load_model()

    def build_id2name(self):
        if any(x in self.args.data_dir for x in ['yelp', 'amazon', 'goodreads']):
            item_path = os.path.join(self.args.data_dir, 'id2name.txt')
            with open(item_path, 'r', encoding='utf-8') as f:
                for l in f.readlines():
                    ll = l.strip('\n').split('::', 1)
                    self.id2name[int(ll[0])] = ll[1].strip()
                    self.name2id[ll[1].strip()] = int(ll[0])

            rawid_path = os.path.join(self.args.data_dir, 'id2rawid.txt')
            if os.path.exists(rawid_path):
                with open(rawid_path, 'r', encoding='utf-8') as f:
                    for l in f.readlines():
                        ll = l.strip('\n').split('::')
                        if len(ll) >= 2:
                            self.id2rawid[int(ll[0])] = ll[1].strip()
            else:
                print(f"[UserModelAgent] WARNING: id2rawid.txt not found at {rawid_path}. "
                      f"Run process_data.py to generate it.")
        else:
            raise ValueError("Invalid data dir: {}".format(self.args.data_dir))

    def load_model(self):
        print(f"Loading model from {self.args.model_path}")
        data_directory = self.args.data_dir
        data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
        self.seq_size = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]

        self.model = SASRec(64, self.item_num, self.seq_size, 0.1, self.device)
        self.model.to(self.device)

        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model.eval()
        print("Load model weights success!")

    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'amazon' in self.args.data_dir:
                from constant.amazon_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'goodreads' in self.args.data_dir:
                from constant.goodreads_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'yelp' in self.args.data_dir:
                from constant.yelp_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2

    def act(self, data, reason=None, item_list=None):
        if self.mode == 'prior_rec':
            # SASRec output — dùng làm collaborative hint trong user_prompt,
            # KHÔNG dùng làm target trong system_prompt
            model_output = self.model_generate(data['seq'], data['len_seq'], data['cans'])
            rec_list_str = ', '.join(item_list) if item_list else "None"

            if len(self.memory) == 0:
                # FIX: chỉ truyền seq_str, bỏ prior_answer khỏi system_prompt
                system_prompt = self.user_system_prompt.format(data['seq_str'])
                user_prompt = self.user_user_prompt.format(
                    data['cans_str'],
                    model_output,   # collaborative hint — "your history suggests you enjoy"
                    rec_list_str,
                    reason,
                )
            else:
                # FIX: chỉ truyền seq_str, bỏ prior_answer khỏi system_prompt
                system_prompt = self.user_memory_system_prompt.format(data['seq_str'])
                user_prompt = self.user_memory_user_prompt.format(
                    data['cans_str'],
                    model_output,   # collaborative hint
                    '\n'.join(self.memory),
                    rec_list_str,
                    reason,
                )

            response = api_request(system_prompt, user_prompt, self.args)
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'], '\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'], score)

        response = api_request(system_prompt, user_prompt, self.args)
        return response

    def build_memory(self, info):
        if info['user_reason'] is not None:
            return self.user_build_memory.format(info['epoch'], info['rec_item_list'], info['rec_reason'], info['user_reason'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['rec_item_list'], info['rec_reason'])

    def update_memory(self, info):
        self.info_list.append(info)
        self.memory.append(self.build_memory(info))

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]

    # ------------------------------------------------------------------
    # Strategy 3: Dynamic Sequence Augmentation helpers
    # ------------------------------------------------------------------

    def update_dynamic_sequence(self, data, positive_item_names):
        if not positive_item_names:
            return 0

        seq_unpad = list(data.get('seq_unpad', []))
        padding_id = self.item_num
        added = 0

        for name in positive_item_names:
            item_id = self.name2id.get(name.strip())
            if item_id is None or item_id >= self.item_num:
                continue
            if seq_unpad and seq_unpad[-1] == item_id:
                continue
            seq_unpad.append(item_id)
            added += 1

        if added == 0:
            return 0

        if len(seq_unpad) > self.seq_size:
            seq_unpad = seq_unpad[-self.seq_size:]

        new_len = len(seq_unpad)
        padded = [padding_id] * (self.seq_size - new_len) + seq_unpad

        data['seq'] = padded
        data['len_seq'] = new_len
        data['seq_unpad'] = seq_unpad

        if '_original_seq_str' not in data:
            data['_original_seq_str'] = data.get('seq_str', 'Empty History')
            data['_original_seq_len'] = data.get('len_seq', 0) - added

        original_str = data['_original_seq_str']
        original_len = data['_original_seq_len']

        all_pseudo_ids = seq_unpad[original_len:] if original_len >= 0 else seq_unpad
        all_pseudo_names = [self.id2name.get(iid, f'Item_{iid}') for iid in all_pseudo_ids]

        if original_str == 'Empty History':
            data['seq_str'] = 'Inferred interests: ' + ', '.join(all_pseudo_names)
        else:
            data['seq_str'] = original_str + ' | Inferred interests: ' + ', '.join(all_pseudo_names)

        return added

    def regenerate_prior(self, data):
        data['prior_answer'] = self.model_generate(
            data['seq'], data['len_seq'], data['cans']
        )

    def model_generate(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[id] for id in topK]
        len_ret = int(len(name_list) / 4)
        return ', '.join(name_list[:len_ret])

    def score(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        values = values.numpy()[0]
        topK = topK.numpy()[0]
        score_dict = {}
        for i in range(len(topK)):
            id = topK[i]
            score = values[i]
            name = self.id2name[id]
            score_dict[name] = score
        return score_dict