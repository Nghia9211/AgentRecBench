import numpy as np
import torch
import matplotlib.pyplot as plt 
import os
import json

def load_groundtruth_pairs(gt_file_path):
    masked_pairs = set()
    print(f"--- Loading Groundtruth from file: {gt_file_path} ---")

    if not os.path.exists(gt_file_path):
        print(f"Error: File path does not exist: {gt_file_path}")
        return masked_pairs

    try:
        with open(gt_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            count = 0
            for entry in data:
                u_id = entry.get('user_id')
                i_id = entry.get('item_id')

                if u_id and i_id:

                    masked_pairs.add((str(u_id), str(i_id)))
                    count += 1
            
            print(f"Processed {count} entries.")

    except json.JSONDecodeError:
        print(f"Error: File content is not valid JSON or empty.")
    except Exception as e:
        print(f"Error reading groundtruth file: {e}")

    print(f"Total unique pairs found in Groundtruth: {len(masked_pairs)}")
    return masked_pairs

def sample_bpr_batch(train_dict, num_users, num_nodes, batch_size = 1024):
    """ Retrun triplet : (user, Pos_item, Neg_item) """
    existing_users = list(train_dict.keys())

    users = []
    pos_items = []
    neg_items = []

    sampled_users = np.random.choice(existing_users, batch_size)

    for u in sampled_users:
        pos_list = train_dict[u]

        if len(pos_list) == 0 : continue
        pos_i = np.random.choice(pos_list)

        while True:
            neg_i = np.random.randint(num_users, num_nodes)

            if neg_i not in pos_list:
                break

        users.append(u)
        pos_items.append(pos_i)
        neg_items.append(neg_i)

    return (torch.tensor(users),
            torch.tensor(pos_items),
            torch.tensor(neg_items))

def plot_loss(loss_history,reg_history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Total Loss', color='blue')
    plt.title('Training Loss (BPR + Reg)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(reg_history, label='L2 Regularization', color='orange')
    plt.title('Embedding L2 Norm (Overfitting Check)')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Value')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_loss.png') 
    plt.show() 