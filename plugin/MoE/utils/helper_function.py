import os
import json
import time
import requests
import re

"""
Save Final Metrics
"""

def save_final_metrics(args, total_samples, h1_count, h3_count, h5_count, ndcg5_score):
    
    h1_rate = h1_count / total_samples if total_samples > 0 else 0
    h3_rate = h3_count / total_samples if total_samples > 0 else 0
    h5_rate = h5_count / total_samples if total_samples > 0 else 0
    
    avg_hit_rate = (h1_rate + h3_rate + h5_rate) / 3
    
    result_data = {
        "type": "recommendation",
        "metrics": {
            "top_1_hit_rate": h1_rate,
            "top_3_hit_rate": h3_rate,
            "top_5_hit_rate": h5_rate,
            "average_hit_rate": avg_hit_rate,
            "total_scenarios": total_samples,
            "top_1_hits": h1_count,
            "top_3_hits": h3_count,
            "top_5_hits": h5_count,
            "ndcg@5": ndcg5_score,
        },
        "data_info": {
            "evaluated_count": total_samples,
            "original_simulation_count": total_samples,
            "original_ground_truth_count": total_samples
        }
    }


    try:
        output_dir = os.path.dirname(args.result_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(args.result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"\n📊 Save Detail Results in File : {args.result_file}")
    except Exception as e:
        print(f"\n❌ Error while saving file, summary: {e}")

"""
API Request
"""

def api_request(system_prompt, user_prompt, args, few_shot=None):
    return gpt_api(system_prompt, user_prompt, args, few_shot)

def gpt_api(system_prompt, user_prompt, args, few_shot=None):
    retry_count = 0
    max_retry_num = args.max_retry_num

    # url = "https://api.openai.com/v1/chat/completions"
    if hasattr(args, 'base_url') and args.base_url:
        # Đảm bảo url kết thúc bằng /chat/completions
        url = f"{args.base_url.rstrip('/')}/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"

    api_key = args.api_key.strip('"') if args.api_key else "EMPTY"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    if few_shot is not None:
        if isinstance(few_shot, list):
            messages.extend(few_shot)
        elif isinstance(few_shot, str):
            messages.append({"role": "user", "content": few_shot})
        else:
            messages.append({"role": "user", "content": str(few_shot)})

    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
    }

    while retry_count < max_retry_num:
        request_result = None
        try:
            request_result = requests.post(url, headers=headers, json=payload, timeout=120)

            if request_result.status_code != 200:
                result_json = request_result.json()
                error_message = result_json.get('error', {}).get('message', f"Unknown HTTP error {request_result.status_code}")
                print(f"[ERROR] API Call Failed (Status: {request_result.status_code}, Retry: {retry_count+1}/{max_retry_num}): {error_message}")

                if request_result.status_code == 401:
                    print("[FATAL] API Key Unauthorized (401). Exiting retries.")
                    return None

                raise Exception(error_message)

            result_json = request_result.json()
            if 'error' not in result_json:
                model_output = result_json['choices'][0]['message']['content']


                return model_output.strip()
            else:
                error_message = result_json.get('error', {}).get('message', "Internal API error.")
                print(f"[ERROR] API Response Error (Retry: {retry_count+1}/{max_retry_num}): {error_message}")
                raise Exception(error_message)

        except requests.exceptions.Timeout:
            print(f"[WARNING] Request Timeout (Retry: {retry_count+1}/{max_retry_num}). Retrying...")

        except requests.exceptions.RequestException as req_e:
            print(f"[WARNING] Network/Connection Error (Retry: {retry_count+1}/{max_retry_num}): {req_e}")

        except Exception as e:
            print(f"[WARNING] General Error (Retry: {retry_count+1}/{max_retry_num}): {e}")

        retry_count += 1
        if retry_count < max_retry_num:
            time.sleep(min(2 ** retry_count, 10))

    return None

"""
Main split functions
"""

def split_rec_reponse(response):
    """Parse rec-agent response dạng Item: <single item>."""
    if response is None:
        print("[split_rec_reponse] response is None")
        return None, None
    response = str(response) + '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_rec_reponse] cannot split, response =", response)
        return None, None
    return matches[0][0].strip(), matches[0][1].strip()


def split_user_response(response):
    """Parse user-agent response dạng Decision: yes/no."""
    if response is None:
        print("[split_user_response] response is None")
        return None, None
    response = str(response) + '\n'
    pattern = r'Reason: (.*?)\nDecision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_response] cannot split, response =", response)
        return None, None
    reason, decision = matches[0][0].strip(), matches[0][1].strip().lower()
    if decision.startswith('yes'):
        return reason, True
    elif decision.startswith('no'):
        return reason, False
    print("[split_user_response] cannot find flag, response =", response)
    return None, None

"""
Read And Write Process
"""

def read_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip() 
            if line: 
                json_data = json.loads(line)
                data_list.append(json_data)
    return data_list

def write_jsonl(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
    
            line = json.dumps(data, ensure_ascii=False)
            f.write(line + '\n')
            
def append_jsonl(file_path, data):
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    with open(file_path, 'a', encoding='utf-8') as f:
        line = json.dumps(data, ensure_ascii=False)
        f.write(line + '\n')

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file: 
        data_list = json.load(json_file)
    return data_list
            
def write_json(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=4)  