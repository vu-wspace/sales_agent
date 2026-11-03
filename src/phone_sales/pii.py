# Thông tin cá nhân của khách (số điện thoại, email):
#   - che trước khi gửi LLM / LangSmith / lịch sử hội thoại (redact)
#   - che khi hiển thị trên thẻ xác nhận, tóm tắt đơn (mask_phone, mask_email)
#   - che trong log (RedactFilter)
#   - mã hóa khi lưu database (encrypt/decrypt), tra cứu bằng blind index HMAC (phone_key, email_key)
import hashlib
import hmac
import logging
import re
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from phone_sales import config

PHONE_TOKEN = '[SỐ ĐIỆN THOẠI]'
EMAIL_TOKEN = '[EMAIL]'
ENC_PREFIX = 'enc:'  # giá trị đã mã hóa trong database (phân biệt với dữ liệu cũ chưa mã hóa)

EMAIL_RE = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}(?![\w-])')
# Ứng viên số điện thoại: +84/84/0 rồi 9-10 chữ số, có thể cách bằng dấu cách, chấm, gạch ("0912 345 678").
# Kiểm tra lại bằng _is_phone: giá tiền ("28.999.000đ"), mã đơn không bị che nhầm
PHONE_CANDIDATE_RE = re.compile(r'(?<![\w+.,])(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?![\w.,]?\d)')
# 10 số (giống kiểm tra trên form: 0 + 9 số, kể cả đầu số lạ như 0123456789), số cố định 11 số (02x)
VN_PHONE_RE = re.compile(r'0(?:\d{9}|2\d{9})')


class PiiKeyError(RuntimeError):
    """Thiếu hoặc sai PII_KEY: không mã hóa/giải mã được thông tin khách."""


def normalize_phone(phone: str) -> str:
    # "+84 912.345.678" -> "0912345678"
    digits = re.sub(r'\D', '', phone)
    return '0' + digits[2:] if digits.startswith('84') and len(digits) in (11, 12) else digits


def _is_phone(text: str) -> bool:
    return bool(VN_PHONE_RE.fullmatch(normalize_phone(text)))


def redact(text: str) -> str:
    """Thay số điện thoại, email trong chữ bằng nhãn ([SỐ ĐIỆN THOẠI], [EMAIL])."""
    if not text:
        return text
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    return PHONE_CANDIDATE_RE.sub(lambda m: PHONE_TOKEN if _is_phone(m.group(0)) else m.group(0), text)


def mask_phone(phone: str | None) -> str | None:
    # 0912345678 -> 0912 *** 678: đủ để khách nhận ra số của mình, không lộ cả số
    if not phone:
        return phone
    p = normalize_phone(phone)
    return f'{p[:4]} *** {p[-3:]}' if len(p) >= 9 else '***'


def mask_email(email: str | None) -> str | None:
    # nguyenvana@gmail.com -> ng***@gmail.com (không lộ độ dài tên)
    if not email or '@' not in email:
        return email
    user, domain = email.rsplit('@', 1)
    return f'{user[:2] if len(user) > 2 else user[:1]}***@{domain}'


# ---------------------------------------------------------------- mã hóa trong database

@lru_cache
def _fernet(key: str) -> Fernet:
    try:
        return Fernet(key)
    except (ValueError, TypeError) as e:
        raise PiiKeyError('PII_KEY không hợp lệ (cần khóa Fernet base64 32 byte)') from e


def _key() -> str:
    if not config.PII_KEY:
        raise PiiKeyError('Chưa đặt PII_KEY trong .env: không mã hóa được số điện thoại, email của khách')
    return config.PII_KEY


def encrypt(value: str | None) -> str | None:
    if value is None:
        return None
    return ENC_PREFIX + _fernet(_key()).encrypt(value.encode()).decode()


def decrypt(value: str | None) -> str | None:
    # Giá trị cũ chưa mã hóa (không có tiền tố) trả nguyên
    if not value or not value.startswith(ENC_PREFIX):
        return value
    try:
        return _fernet(_key()).decrypt(value[len(ENC_PREFIX):].encode()).decode()
    except InvalidToken as e:
        raise PiiKeyError('Không giải mã được: PII_KEY khác khóa đã dùng để mã hóa') from e


def is_encrypted(value: str | None) -> bool:
    return bool(value) and value.startswith(ENC_PREFIX)


def _blind(value: str) -> str:
    # Fernet mỗi lần mã hóa ra chuỗi khác -> không tra cứu được; HMAC cố định theo khóa thì tra được mà
    # không lưu giá trị gốc. Khóa HMAC tách khỏi khóa mã hóa (cùng sinh từ PII_KEY)
    mac_key = hashlib.sha256(b'phone-sales-blind-index:' + _key().encode()).digest()
    return hmac.new(mac_key, value.encode(), hashlib.sha256).hexdigest()


def phone_key(phone: str) -> str:
    return _blind('phone:' + normalize_phone(phone))


def email_key(email: str | None) -> str | None:
    return _blind('email:' + email.strip().lower()) if email else None


# ---------------------------------------------------------------- mã khách cho phiên chat mới

def customer_token(customer_id: str) -> str:
    """Mã khách kèm chữ ký HMAC, trình duyệt giữ để phiên chat sau nhận ra khách quen (api.py: POST /session).
    Có chữ ký nên không đoán/đổi được mã khách khác để xem tên, đơn cũ của họ."""
    return f"{customer_id}.{_blind('customer:' + customer_id)[:32]}"


def verify_customer_token(token: str | None) -> str | None:
    """Trả mã khách nếu chữ ký đúng, sai/thiếu thì None."""
    customer_id, _, sig = (token or '').rpartition('.')
    if not customer_id or not sig:
        return None
    return customer_id if hmac.compare_digest(sig, customer_token(customer_id).rpartition('.')[2]) else None


# ---------------------------------------------------------------- che trong log

class RedactFilter(logging.Filter):
    """Che số điện thoại, email trong mọi dòng log, kể cả traceback (lỗi psycopg có thể in tham số SQL)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        # Che từng tham số, giữ nguyên dạng tuple/dict: formatter của uvicorn.access tách args theo vị trí
        # (địa chỉ, method, path, ...), gộp thành chuỗi sẽ làm hỏng log
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def install_log_filter() -> None:
    # Gắn vào handler (không phải logger): filter của logger không áp dụng cho log từ logger con truyền lên
    handlers = [h for name in ('', 'uvicorn', 'uvicorn.error', 'uvicorn.access')
                for h in logging.getLogger(name).handlers]
    if logging.lastResort:
        handlers.append(logging.lastResort)  # log của phone_sales.* khi root chưa có handler
    for h in handlers:
        if not any(isinstance(f, RedactFilter) for f in h.filters):
            h.addFilter(RedactFilter())
