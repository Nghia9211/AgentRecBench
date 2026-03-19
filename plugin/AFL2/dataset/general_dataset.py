import torch.utils.data as data
import torch # <-- Thêm import torch
import os
import pandas as pd
import random
import numpy as np

class GeneralDataset(data.Dataset):
    def __init__(self, args, stage=None):
        self.args = args
        self.data_dir = args.data_dir
        self.stage = stage
        self.cans_num = 20 
        self.sep = ", "
        
        # Đọc thông tin thống kê
        statis_path = os.path.join(self.data_dir, 'data_statis.df')
        statis = pd.read_pickle(statis_path)
        self.item_num = int(statis['item_num'][0])
        self.padding_item_id = 0 # Thường quy ước padding là 0
        
        # Load dữ liệu
        self.check_files()
    
    def __len__(self):
        return len(self.session_data)

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        
        # Tạo danh sách ứng viên (Negative Sampling)
        candidates = self.negative_sampling(temp['seq_unpad'], temp['next'])
        cans_name = [self.item_id2name.get(can, "Unknown") for can in candidates]
        
        # CHUYỂN ĐỔI SANG TENSOR TẠI ĐÂY
        sample = {
            'id': temp['uid'],
            # Chuyển list ID thành LongTensor [Seq_Len]
            'seq': torch.tensor(temp['seq'], dtype=torch.long), 
            
            'seq_str': self.sep.join(temp['seq_title']),
            
            # Chuyển len thành tensor đơn lẻ
            'len_seq': torch.tensor(temp['len_seq'], dtype=torch.long),
            
            'cans': torch.tensor(candidates, dtype=torch.long),
            'cans_name': cans_name,
            'cans_str': self.sep.join(cans_name),
            'len_cans': torch.tensor(self.cans_num, dtype=torch.long),
            
            # Đổi 'item_id' thành tensor để dùng trong Loss Function
            'next': torch.tensor(temp['next'], dtype=torch.long), 
            
            'item_name': temp['next_item_name'],
            'correct_answer': temp['next_item_name']
        }
        return sample
    
    def negative_sampling(self, seq_unpad, next_item):
        canset = []
        while len(canset) < self.cans_num - 1:
            rand_id = random.randint(1, self.item_num - 1) # Tránh ID 0 nếu đó là padding
            if rand_id not in seq_unpad and rand_id != next_item:
                canset.append(rand_id)
        
        candidates = canset + [next_item]
        random.shuffle(candidates)
        return candidates  

    def check_files(self):
        self.item_id2name = self.get_id2name()
        if self.stage == 'train':
            filename = "train_data.df"
        elif self.stage == 'val':
            filename = "Val_data.df"
        elif self.stage == 'test':
            filename = "Test_data.df"
        else:
            filename = "train_data.df"
        
        data_path = os.path.join(self.data_dir, filename)
        # Truyền self.padding_item_id vào để xử lý đồng bộ
        self.session_data = self.session_data4frame(data_path, self.item_id2name)  

    def get_id2name(self):
        id2name = dict()
        item_path = os.path.join(self.data_dir, 'id2name.txt')
        if not os.path.exists(item_path):
            return {i: f"Item_{i}" for i in range(self.item_num + 1)}
            
        with open(item_path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                ll = l.strip('\n').split('::')
                if len(ll) >= 2:
                    id2name[int(ll[0])] = ll[1].strip()
        return id2name
    
    def session_data4frame(self, datapath, id2name):
        df = pd.read_pickle(datapath)
        
        # Loại bỏ padding để lấy tên sản phẩm thật
        def remove_padding(xx):
            return [x for x in xx if x != self.padding_item_id]
            
        df['seq_unpad'] = df['seq'].apply(remove_padding)
        
        def seq_to_title(x): 
            return [id2name.get(x_i, "Unknown") for x_i in x]
        df['seq_title'] = df['seq_unpad'].apply(seq_to_title)
        
        def next_item_title(x): 
            return id2name.get(x, "Unknown")
        df['next_item_name'] = df['next'].apply(next_item_title)
        
        return df