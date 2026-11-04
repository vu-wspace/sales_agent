# Chuẩn hóa tin nhắn của khách trước khi luật từ khóa (conversation.py) và model đọc,
# và dọn câu trả lời của model trước khi gửi cho khách.
import re
import unicodedata

# Ký tự vô hình hay dính khi copy/paste (zero-width, BOM, soft hyphen)
INVISIBLE = re.compile('[​-‏⁠﻿­]')

# Bỏ dấu kiểu cũ -> kiểu mới ở cuối từ: "hoà" -> "hòa", "huỷ" -> "hủy", "khoẻ" -> "khỏe".
# Luật từ khóa viết theo kiểu mới; "hoàng", "toàn" (còn chữ phía sau) giữ nguyên.
OLD_TONES = {
    'oà': 'òa', 'oá': 'óa', 'oả': 'ỏa', 'oã': 'õa', 'oạ': 'ọa',
    'oè': 'òe', 'oé': 'óe', 'oẻ': 'ỏe', 'oẽ': 'õe', 'oẹ': 'ọe',
    'uỳ': 'ùy', 'uý': 'úy', 'uỷ': 'ủy', 'uỹ': 'ũy', 'uỵ': 'ụy',
}
OLD_TONE_RE = re.compile(rf"({'|'.join(OLD_TONES)})(?!\w)", re.I)

# Viết tắt, teencode hay gặp khi chat -> từ đầy đủ. Chỉ thay cả từ, không ngay sau số ("100k", "100 k" là tiền).
# Không thay "v", "z" (vậy) vì trùng "Z Fold", "Z Flip"; không thay "ip", "ss" vì tools.py đã xử lý tên máy.
ABBREVIATIONS = {
    'ko': 'không', 'k': 'không', 'kh': 'không', 'hok': 'không', 'hông': 'không', 'khum': 'không', 'hem': 'không',
    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được', 'dzc': 'được',
    'mik': 'mình', 'mk': 'mình', 'mìn': 'mình',
    'j': 'gì', 'gi': 'gì', 'e': 'em',
    'bn': 'bao nhiêu', 'bnh': 'bao nhiêu', 'bnhiu': 'bao nhiêu', 'bao nhiu': 'bao nhiêu',
    'ntn': 'như thế nào', 'vs': 'với', 'mún': 'muốn', 'cx': 'cũng', 'cg': 'cũng',
    'sp': 'sản phẩm', 'đt': 'điện thoại', 'dt': 'điện thoại', 'ib': 'nhắn tin',
    'trc': 'trước', 'r': 'rồi',
}
ABBREV_RE = re.compile(r'(?<!\w)(?<!\d )(' + '|'.join(sorted(map(re.escape, ABBREVIATIONS), key=len, reverse=True)) + r')(?!\w)',
                       re.I)
# "20tr", "20 tr" -> "20 triệu"; "15m" giữ nguyên (parse_budget đã hiểu)
MILLION_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*tr(?!\w)', re.I)
REPEAT_RE = re.compile(r'([^\W\d_])\1{2,}')  # "đẹppppp" -> "đẹp", "quááá" -> "quá"

def nfc(text: str) -> str:
    # Bàn phím iOS/macOS, một số bộ gõ gửi dấu rời (NFD): "ố" = "o" + 2 dấu -> regex có dấu không khớp
    return unicodedata.normalize('NFC', text)

def normalize_user_text(text: str) -> str:
    text = INVISIBLE.sub('', nfc(text))
    text = OLD_TONE_RE.sub(lambda m: _keep_case(m.group(0), OLD_TONES[m.group(0).lower()]), text)
    text = REPEAT_RE.sub(r'\1', text)
    text = MILLION_RE.sub(r'\1 triệu', text)
    text = ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(0).lower()], text)
    return re.sub(r'\s+', ' ', text).strip()

def _keep_case(src: str, dst: str) -> str:
    return dst.upper() if src.isupper() else dst[0].upper() + dst[1:] if src[0].isupper() else dst

# Chữ "ạ" đứng riêng ("không ạ?", "vâng ạ"), không đụng "Dạ", "lạ", "hạ"
A_PARTICLE_RE = re.compile(r'[ \t]*(?<!\w)ạ(?!\w)', re.I)
NHE_RE = re.compile(r'(?<!\w)nhé(?!\w)')  # giọng miền Nam: "nha" thay "nhé"

# Nhãn che PII (pii.PHONE_TOKEN, pii.EMAIL_TOKEN) mà model chép lại vào câu trả lời ("liên hệ qua số [SỐ ĐIỆN THOẠI]")
PII_TOKEN_RE = re.compile(r'(?:(?:qua |theo )?(?:số(?: điện thoại)?|sđt) )?\[SỐ ĐIỆN THOẠI\]|'
                          r'(?:(?:qua |theo )?(?:địa chỉ )?email )?\[EMAIL\]', re.I)

def _pii_phrase(m: re.Match) -> str:
    via = 'qua ' if m.group(0).lower().startswith(('qua ', 'theo ')) else ''
    what = 'email' if 'EMAIL' in m.group(0).upper() else 'số điện thoại'
    return f'{via}{what} anh/chị đã cung cấp'

def clean_answer(text: str) -> str:
    # Một số model (Nemotron, Qwen) để lộ phần suy nghĩ <think>...</think> trong câu trả lời
    text = re.sub(r'<think>.*?</think>', '', nfc(text), flags=re.S | re.I)
    text = INVISIBLE.sub('', text)
    text = PII_TOKEN_RE.sub(_pii_phrase, text)
    # Cửa hàng không dùng "ạ" (prompt đã dặn, đây là chốt chặn khi model vẫn viết)
    text = A_PARTICLE_RE.sub('', text)
    text = NHE_RE.sub('nha', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()
