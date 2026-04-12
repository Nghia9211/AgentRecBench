import re

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_item_name(name: str) -> str:
    """Xóa phần description thừa sau tên item.
    
    Một số format model hay trả về:
      "Grandma Stinks! - Since this book was..."   → "Grandma Stinks!"
      "Peter Nimble (#1) - This book falls into..."→ "Peter Nimble (#1)"
      "Some Book: subtitle - description here"     → "Some Book: subtitle"
    
    Quy tắc: cắt tại dấu " - " ĐẦU TIÊN nếu phần sau nó dài (> 15 ký tự),
    tức là đó là description chứ không phải một phần tên sách.
    """
    # Tìm " - " đầu tiên
    dash_pos = name.find(' - ')
    if dash_pos != -1:
        suffix = name[dash_pos + 3:].strip()
        # Nếu phần sau dấu " - " dài → đó là description, cắt bỏ
        if len(suffix) > 15:
            return name[:dash_pos].strip()
    return name.strip()


def _parse_item_block(items_block: str) -> list:
    """Parse item block thành list, hỗ trợ 3 format:
      1. Đánh số dòng :  "1. Item"  hoặc  "1) Item"
      2. Dấu gạch đầu :  "- Item"
      3. Phẳng (comma) :  "Item1, Item2, Item3"
    """
    items_block = items_block.strip()

    # Format 1: numbered lines
    numbered = re.findall(r'^\d+[.)]\s*(.+)', items_block, re.MULTILINE)
    if numbered:
        return [_clean_item_name(i) for i in numbered if i.strip()]

    # Format 2: bullet lines
    bulleted = re.findall(r'^[-*•]\s*(.+)', items_block, re.MULTILINE)
    if bulleted:
        return [_clean_item_name(i) for i in bulleted if i.strip()]

    # Format 3: comma-separated (kể cả cùng 1 dòng hoặc nhiều dòng)
    flat = [_clean_item_name(i) for i in items_block.replace('\n', ',').split(',') if i.strip()]
    return flat


# ---------------------------------------------------------------------------
# Main split functions
# ---------------------------------------------------------------------------

def split_rec_reponse_top_n(response):
    """Parse rec-agent response dạng:
        Reason: <text>
        Items: <item1>, <item2>, ...   (hoặc dạng đánh số, bullet)
    Trả về (reason: str, item_list: list) hoặc (None, None) nếu thất bại.
    """
    if response is None:
        return None, None

    response = str(response).strip()

    # --- Tách Reason ---
    reason = None
    reason_match = re.search(r'Reason:\s*(.*?)(?=\nItems?:)', response, re.DOTALL | re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()

    # --- Tách Items block ---
    items_match = re.search(r'Items?:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    item_list = []
    if items_match:
        item_list = _parse_item_block(items_match.group(1))

    if not reason or not item_list:
        print("[split_rec_reponse_top_n] Cannot split, response =", response)
        return None, None

    return reason, item_list


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


def split_user_rec_reponse(response):
    if response is None:
        print("[split_user_rec_reponse] response is None")
        return None, None
    response = str(response) + '\n'
    pattern = r'Reason: (.*?)\nItem: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_rec_reponse] cannot split, response =", response)
        return None, None
    return matches[0][0].strip(), matches[0][1].strip()


def split_user_ab_response(response):
    if response is None:
        print("[split_user_ab_response] response is None")
        return None, None
    response = str(response) + '\n'
    pattern = r'Reason: (.*?)\nDecision: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_user_ab_response] cannot split, response =", response)
        return None, None
    reason, decision = matches[0][0].strip(), matches[0][1].strip().lower()
    if decision.startswith('yes'):
        return reason, 1
    elif decision.startswith('no'):
        return reason, 0
    print("[split_user_ab_response] cannot find flag, response =", response)
    return None, None


def split_prior_rec_response(response):
    if response is None:
        print("[split_prior_rec_response] response is None")
        return None
    response = str(response) + '\n'
    pattern = r'Item: (.*?)\n'
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_rec_response] cannot split, response =", response)
        return None
    return matches[0].strip()


def split_prior_llama3_response(response):
    if response is None:
        print("[split_prior_llama3_response] response is None")
        return None
    pattern = r'Item: (.*?)<\|eot_id\|>'
    matches = re.findall(str(response), re.DOTALL)
    if len(matches) != 1:
        print("[split_prior_llama3_response] cannot split via eot, trying fallback")
        return split_prior_rec_response(response)
    return matches[0].strip()


# ---------------------------------------------------------------------------
# Strategy 3: Dynamic Sequence Augmentation helper
# ---------------------------------------------------------------------------

def extract_positive_mentions(user_reason, rec_item_list, max_items=2):
    """Trích xuất items mà user có thái độ tích cực trong lý do từ chối.

    Trả về list tên item (len <= max_items).
    """
    if not user_reason or not rec_item_list:
        return []

    NEGATIVE_QUALIFIERS = [
        "not what i", "don't want", "not interested", "completely wrong",
        "irrelevant", "not relevant", "nothing to do", "not related",
        "dislike", "hate",
    ]

    reason_lower = user_reason.lower()
    mentioned = []

    for item_name in rec_item_list:
        item_lower = item_name.lower().strip()
        if not item_lower or item_lower not in reason_lower:
            continue
        pos = reason_lower.index(item_lower)
        window = reason_lower[max(0, pos - 80): pos + len(item_lower) + 40]
        if not any(neg in window for neg in NEGATIVE_QUALIFIERS):
            mentioned.append(item_name)

    # Deduplicate
    seen, unique = set(), []
    for item in mentioned:
        key = item.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:max_items] if unique else [rec_item_list[0]]