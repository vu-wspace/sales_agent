# Kiểm tra câu trả lời có bịa số liệu không (giá, pin, camera, thời hạn bảo hành...), bằng luật, không gọi thêm LLM.
# Mỗi con số kèm đơn vị trong câu trả lời phải xuất hiện trong nguồn: kết quả tool, trạng thái hội thoại
# (giỏ hàng, ngân sách) hoặc tin nhắn của khách. graph.py cho model viết lại 1 lần nếu có số không rõ nguồn.
import re
from langchain_core.messages import AnyMessage

# Số kèm đơn vị: "28.999.000đ", "4.900 mAh", "6,3 inch", "45W", "30 ngày", "10%", "12GB", "29,9 triệu"
UNITS = r'đ|₫|vnđ|vnd|đồng|triệu|tr|mah|mp|inch|hz|w|nits|gb|tb|giờ|ngày|tháng|năm|%'
CLAIM_RE = re.compile(rf'(?<![\w.,])(\d[\d.,]*)\s*({UNITS})(?!\w)', re.I)
NUMBER_RE = re.compile(r'\d[\d.,]*\d|\d')
MILLION = 1_000_000

def _digits(n: str) -> str:
    # "4.900" và "4900", "6,3" và "6.3" coi là một: so bằng dãy chữ số
    return re.sub(r'\D', '', n)

def source_numbers(texts: list[str]) -> set[str]:
    return {_digits(n) for t in texts for n in NUMBER_RE.findall(t)}

def _as_million(n: str) -> float | None:
    try:
        return float(n.replace('.', '').replace(',', '.')) if n.count(',') <= 1 else None
    except ValueError:
        return None

def ungrounded(answer: str, sources: list[str]) -> list[str]:
    """Các số liệu trong câu trả lời không có trong nguồn (giữ thứ tự, bỏ trùng)."""
    known = source_numbers(sources)
    prices = [int(d) for d in known if len(d) >= 6]
    bad = []
    for m in CLAIM_RE.finditer(answer):
        number, unit = m.group(1).rstrip('.,'), m.group(2).lower()
        if _digits(number) in known:
            continue
        if unit in ('triệu', 'tr'):
            # "khoảng 29,99 triệu" làm tròn từ giá 29.990.001đ: chấp nhận nếu lệch dưới 1 triệu
            value = _as_million(number)
            if value is not None and any(abs(p - value * MILLION) < MILLION for p in prices):
                continue
        bad.append(m.group(0).strip())
    return list(dict.fromkeys(bad))

def sources_of(messages: list[AnyMessage], extra: str = '') -> list[str]:
    # Không lấy câu trả lời cũ của model làm nguồn: số đã bịa ở lượt trước không thành "có nguồn".
    # Không lấy web_search: giá, thông số, chính sách chỉ theo dữ liệu cửa hàng (quy tắc 5, 7 trong prompts.py);
    # máy chưa có thông số mà model lấy số từ web thì phải viết lại thành "chưa có thông tin"
    return [extra, *(m.text for m in messages
                     if m.type == 'human' or (m.type == 'tool' and getattr(m, 'name', None) != 'web_search'))]

def correction(bad: list[str]) -> str:
    return ('Câu trả lời vừa rồi có số liệu không có trong kết quả công cụ hay tin nhắn của khách: '
            + ', '.join(bad) + '. Viết lại câu trả lời cho khách: chỉ dùng số liệu đúng như kết quả công cụ; '
            'cần số liệu chưa tra thì gọi công cụ; công cụ không có thì nói chưa có thông tin, không ước đoán. '
            'Không nhắc tới việc viết lại.')
