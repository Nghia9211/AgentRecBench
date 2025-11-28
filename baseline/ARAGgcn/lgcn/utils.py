import numpy as np
import torch
import matplotlib.pyplot as plt 


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