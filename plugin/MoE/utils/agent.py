import os
import torch
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model

sys.modules['SASRecModules_ori'] = model

from utils.helper_function import write_jsonl, read_jsonl, api_request
from utils.model import SASRec

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
            self.model    = shared_sasrec['model']
            self.id2name  = shared_sasrec['id2name']
            self.name2id  = shared_sasrec['name2id']
            self.id2rawid = shared_sasrec['id2rawid']
            self.seq_size = shared_sasrec['seq_size']
            self.item_num = shared_sasrec['item_num']
            self.device   = shared_sasrec['device']
            print("[UserModelAgent] Reusing shared SASRec — skipping reload.")
        else:
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

    def act(self, data, reason=None, item_list=None, docstore_cache=None):
        if self.mode == 'prior_rec':
            
            def enrich(items_input):
                if not items_input:
                    return "None"
                
                items = [i.strip() for i in items_input.split(',')] if isinstance(items_input, str) else items_input
                
                if not docstore_cache:
                    return ", ".join(items)
                
                enriched = []
                for item in items:
                    key = item.lower()
                    if key in docstore_cache:
                        rich_text = docstore_cache[key]
                        parts = rich_text.split(' | ')
                        
                        meta_parts = []
                        for p in parts:
                            p_clean = p.strip()
                            
                            if p_clean.lower() == key:
                                continue

                            if ':' not in p_clean:
                                continue
                                
                            meta_parts.append(p_clean)
                        
                        if meta_parts:
                            meta = " | ".join(meta_parts[:2])
                            enriched.append(f"{item} [{meta}]")
                        else:
                            enriched.append(item)
                    else:
                        enriched.append(item)
                        
                return ", ".join(enriched)

            enriched_seq_str = enrich(data['seq_str'])
            print(f"Enriched Seq_str : {enriched_seq_str} \n")
            enriched_rec_list_str = enrich(item_list)
            print(f"Enriched rec list : {enriched_rec_list_str} \n")
            enriched_cans_str = enrich(data.get('cans_str', ''))
            print(f"Enriched cans_str : {enriched_cans_str} \n")

            if len(self.memory) == 0:
                system_prompt = self.user_system_prompt.format(enriched_seq_str)
                user_prompt = self.user_user_prompt.format(
                    enriched_cans_str,
                    enriched_seq_str,   
                    enriched_rec_list_str,
                    reason,
                )
            else:
                system_prompt = self.user_memory_system_prompt.format(enriched_seq_str)
                user_prompt = self.user_memory_user_prompt.format(
                    enriched_cans_str,
                    enriched_seq_str,   
                    '\n'.join(self.memory),
                    enriched_rec_list_str,
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