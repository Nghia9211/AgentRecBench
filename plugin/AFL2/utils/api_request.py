import requests
import time
import json

# LangSmith tracing — optional, degrades gracefully if not installed
try:
    from langsmith import traceable
    HAS_LANGSMITH = True
except ImportError:
    HAS_LANGSMITH = False
    def traceable(*args, **kwargs):
        """No-op decorator fallback."""
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


def api_request(system_prompt, user_prompt, args, few_shot=None):
    if "gpt" in args.model:
        return gpt_api(system_prompt, user_prompt, args, few_shot)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


@traceable(
    run_type="llm",
    name="AFL-GPT-Call",
    metadata={"source": "afl_vanilla"}
)
def gpt_api(system_prompt, user_prompt, args, few_shot=None):
    retry_count = 0
    max_retry_num = args.max_retry_num

    url = "https://api.openai.com/v1/chat/completions"

    api_key = args.api_key.strip('"') if args.api_key else ""

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

                # --- Log token usage for LangSmith metadata ---
                usage = result_json.get('usage', {})
                if usage and HAS_LANGSMITH:
                    try:
                        from langsmith import get_current_run_tree
                        rt = get_current_run_tree()
                        if rt:
                            rt.extra = rt.extra or {}
                            rt.extra["token_usage"] = {
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                            }
                    except Exception:
                        pass  # Tracing metadata is best-effort

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