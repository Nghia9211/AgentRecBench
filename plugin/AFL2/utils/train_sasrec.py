import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import sys

# Thiết lập đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset.general_dataset import GeneralDataset
from utils.model import SASRec
from utils.trainingfigure import plot_training_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/amazon', type=str)
    parser.add_argument('--output_dir', default='./saved_models', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50) 
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=2)    
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset (Train & Valid)
    print("Loading data...")
    try:
        train_dataset = GeneralDataset(args, stage='train')
        valid_dataset = GeneralDataset(args, stage='valid') # Load tập valid
    except Exception as e:
        print(f"Lỗi load dataset: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    item_num = train_dataset.item_num 
    seq_size = getattr(train_dataset, 'max_seq_len', 50) 
    
    print(f"Dataset info: Item_num={item_num}, Seq_size={seq_size}")
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    # 2. Khởi tạo Model
    model = SASRec(args.hidden_size, item_num, seq_size, args.dropout, device, args.num_heads)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Khởi tạo các list lưu lịch sử
    train_loss_history, val_loss_history, l2_history = [], [], []
    best_val_loss = float('inf') # Để theo dõi valid loss thấp nhất

    # 3. Training Loop
    print(f"🚀 Bắt đầu train {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # --- PHASE: TRAINING ---
        model.train()
        total_train_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            seq = batch['seq'].to(device) 
            pos = batch['next'].to(device)
            len_seq = batch['len_seq'].to(device)

            optimizer.zero_grad()
            logits = model(seq, len_seq) 
            loss = criterion(logits, pos)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- PHASE: VALIDATION ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                seq = batch['seq'].to(device)
                pos = batch['next'].to(device)
                len_seq = batch['len_seq'].to(device)
                
                logits = model(seq, len_seq)
                loss = criterion(logits, pos)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(valid_loader)
        val_loss_history.append(avg_val_loss)

        # Lấy L2 norm để theo dõi
        with torch.no_grad():
            l2_norm = torch.norm(model.item_embeddings.weight, p=2).item()
            l2_history.append(l2_norm)
            
        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"L2: {l2_norm:.2f} | "
              f"Time: {time.time()-start_time:.1f}s")

        # --- LƯU MODEL TỐT NHẤT (BEST MODEL) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = os.path.basename(args.data_dir) + "_best_model.pt"
            save_path = os.path.join(args.output_dir, model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'item_num': item_num
            }, save_path)
            # print(f" --> Saved Best Model at Epoch {epoch+1}")

    # 4. Lưu Model cuối cùng (Last Epoch)
    final_model_name = os.path.basename(args.data_dir) + "_last_model.pt"
    torch.save(model.state_dict(), os.path.join(args.output_dir, final_model_name))
    print(f"✅ Training hoàn tất. Best Val Loss: {best_val_loss:.4f}")

    # 5. Vẽ biểu đồ (Cần cập nhật hàm plot_training_results để nhận thêm val_loss)
    plot_name = os.path.basename(args.data_dir) + "_training_plot.png"
    plot_path = os.path.join(args.output_dir, plot_name)
    
    # Giả sử hàm plot_training_results của bạn có thể nhận thêm tập valid_loss
    # Nếu không, bạn hãy sửa hàm đó để vẽ 2 đường Train/Valid Loss
    plot_training_results(train_loss_history, l2_history, plot_path, val_loss_history=val_loss_history)

if __name__ == '__main__':
    main()