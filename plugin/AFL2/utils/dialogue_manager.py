import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.regular_function import split_user_response, split_rec_reponse_top_n, extract_positive_mentions
from utils.agent import RecAgent, UserModelAgent

try:
    from langsmith import traceable
except ImportError:
    def traceable(*a, **kw):
        def dec(fn): return fn
        if len(a) == 1 and callable(a[0]): return a[0]
        return dec

def error_handler(e):
    print(f"!!! ERROR IN SUBPROCESS: {e}")
    import traceback
    traceback.print_exc()


@traceable(run_type="chain")
def recommend(data, args):
    user_id = data.get('id')

    if getattr(args, 'use_arag', False):
        from utils.arag_rec_agent import ARAGRecAgent

        rec_agent = ARAGRecAgent(args)
        print(f"[User {user_id}] Using ARAG RecAgent")

        # FIX: đóng gói shared SASRec resources thành dict
        # ARAGRecAgent đã load SASRec vào class variables → tái sử dụng
        shared_sasrec = {
            'model':    ARAGRecAgent._shared_sasrec_model,
            'id2name':  ARAGRecAgent._shared_id2name,
            'name2id':  ARAGRecAgent._shared_name2id,
            'id2rawid': ARAGRecAgent._shared_id2rawid,
            'seq_size': ARAGRecAgent._shared_seq_size,
            'item_num': ARAGRecAgent._shared_item_num,
            'device':   ARAGRecAgent._shared_device,
        }
        # UserModelAgent nhận shared_sasrec → bỏ qua load_model()
        user_agent = UserModelAgent(args, shared_sasrec=shared_sasrec)

    else:
        rec_agent  = RecAgent(args, 'prior_rec')
        user_agent = UserModelAgent(args)   # load bình thường
        print(f"[User {user_id}] Using vanilla RecAgent")

    flag = False
    epoch = 1
    rec_item_list = []
    new_data_list = []
    hit_at_n = {1: False, 3: False, 5: False}

    while flag is False and epoch <= args.max_epoch:
        prefix = f"[User {user_id}][Round {epoch}]"

        # --- Rec Agent turn ---
        max_rec_retries = 3
        rec_reason = None
        current_rec_list = []

        for attempt in range(max_rec_retries):
            if getattr(args, 'use_arag', False):
                rec_reason, current_rec_list = rec_agent.act(data)
            else:
                rec_agent_response = rec_agent.act(data)

                if rec_agent_response is None:
                    print(f"{prefix} RecAgent returned None, retry {attempt + 1}/{max_rec_retries}")
                    time.sleep(5)
                    continue

                print(f"{prefix} Rec Agent Response : {rec_agent_response}\n")
                rec_reason, current_rec_list = split_rec_reponse_top_n(rec_agent_response)

            if rec_reason and current_rec_list:
                rec_item_list = current_rec_list
                break

            print(f"{prefix} Split failed, retry {attempt + 1}/{max_rec_retries}")
            time.sleep(1)
        else:
            print(f"{prefix} RecAgent failed all retries. Using fallback.")
            rec_item_list = data.get('cans_name', [])[:5]
            rec_reason = "Fallback: recommendation agent unavailable."

        # --- User Agent turn ---
        max_user_retries = 3
        for attempt in range(max_user_retries):
            user_agent_response = user_agent.act(data, rec_reason, rec_item_list)
            user_reason, flag = split_user_response(user_agent_response)
            if user_reason is not None and flag is not None:
                break
            print(f"{prefix} UserAgent parse failed, retry {attempt + 1}/{max_user_retries}")
            time.sleep(1)
        else:
            print(f"{prefix} UserAgent failed all retries. Forcing continue.")
            user_reason = "Could not parse user response."
            flag = False

        # --- Log this round ---
        if getattr(args, 'use_arag', False):
            rec_res = f"Reason: {rec_reason}\nItems: {', '.join(current_rec_list[:5])}"
        else:
            rec_res = rec_agent_response

        current_step_data = {
            'id':       str(user_id),
            'epoch':    epoch,
            'rec_res':  rec_res,
            'user_res': user_agent_response,
            'rec_items': rec_item_list,
            'flag':     flag,
        }
        new_data_list.append(current_step_data)

        # --- Check hit nếu user accepted ---
        if flag:
            gt_name = data.get('correct_answer', '').lower().strip()
            current_top_n_lower = [item.lower().strip() for item in rec_item_list]
            print(f"{prefix} Rank list : {rec_item_list}\n"
                  f"{prefix} GT item   : {gt_name}\n")

            if gt_name in current_top_n_lower:
                rank = current_top_n_lower.index(gt_name) + 1
                if rank <= 1: hit_at_n[1] = True
                if rank <= 3: hit_at_n[3] = True
                if rank <= 5: hit_at_n[5] = True
            break

        # --- Update memory ---
        memory_info = {
            "epoch":         epoch,
            "rec_reason":    rec_reason,
            "rec_item_list": rec_item_list,
            "user_reason":   user_reason,
        }
        rec_agent.update_memory(memory_info)
        user_agent.update_memory(memory_info)

        epoch += 1

    return new_data_list, hit_at_n, args