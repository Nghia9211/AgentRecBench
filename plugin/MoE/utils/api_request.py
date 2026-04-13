import requests
import time
import json

def api_request(system_prompt, user_prompt, args, few_shot=None):
    return gpt_api(system_prompt, user_prompt, args, few_shot)

def gpt_api(system_prompt, user_prompt, args, few_shot=None):
    retry_count = 0
    max_retry_num = args.max_retry_num

    # url = "https://api.openai.com/v1/chat/completions"
    if hasattr(args, 'base_url') and args.base_url:
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
            request_result = requests.post(url, headers=headers, json=payload, timeout=30)

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