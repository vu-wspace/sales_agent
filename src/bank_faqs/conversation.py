# Trạng thái hội thoại bán hàng (dùng trong graph.py) và các luật cập nhật nó.
# Cập nhật bằng luật (từ khóa + kết quả tool), không gọi thêm LLM: không tốn quota, chạy tức thì, test được.
import re
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

Stage = Literal['discovery', 'recommending', 'comparing', 'closing', 'handoff']
Sentiment = Literal['positive', 'neutral', 'negative']

MAX_SHOWN = 20          # nhớ tối đa ngần này sản phẩm đã giới thiệu
NEGATIVE_STREAK = 2     # khách bực liên tiếp ngần này lượt thì chuyển nhân viên

def merge_slots(old: dict | None, new: dict | None) -> dict:
    # Giá trị mới ghi đè giá trị cũ; riêng list (vd mục đích dùng) thì cộng dồn, bỏ trùng
    out = dict(old or {})
    for k, v in (new or {}).items():
        if isinstance(v, list):
            out[k] = list(dict.fromkeys([*out.get(k, []), *v]))
        elif v is not None:
            out[k] = v
    return out

def append_unique(old: list | None, new: list | None) -> list:
    return list(dict.fromkeys([*(old or []), *(new or [])]))[-MAX_SHOWN:]

def append(old: list | None, new: list | None) -> list:
    return [*(old or []), *(new or [])]

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # lịch sử tin nhắn (nối thêm, không ghi đè)
    stage: Stage                                          # đang ở bước nào của quy trình bán hàng
    slots: Annotated[dict, merge_slots]                   # nhu cầu đã thu: budget_max, brand, purposes, storage...
    shown_products: Annotated[list[str], append_unique]   # sản phẩm đã giới thiệu
    cart: list[dict]                                      # giỏ hàng (tool giỏ hàng trả về cả giỏ mới)
    objections: Annotated[list[str], append_unique]       # khách đang chê/lo ngại gì
    sentiment_history: Annotated[list[str], append]       # cảm xúc của khách qua từng lượt
    handoff: bool                                         # có cần chuyển nhân viên không (đã bật thì giữ)

# ---------------------------------------------------------------- trích từ tin nhắn của khách

# Từ khóa so trên chữ thường CÓ dấu: bỏ dấu thì "đắt" (giá) trùng "đặt" (đặt hàng)
BRANDS = {'Apple': r'iphone|apple|ip|táo', 'Samsung': r'samsung|galaxy|ss'}
PURPOSES = {
    'chụp ảnh': r'chụp|camera|quay video|selfie|sống ảo',
    'chơi game': r'game|liên quân|pubg|chơi game',
    'pin lâu': r'pin trâu|pin lâu|pin khỏe|pin tốt|dùng cả ngày',
    'học tập': r'học sinh|sinh viên|đi học|học online',
    'công việc': r'công việc|văn phòng|làm việc|họp',
    'cho người lớn tuổi': r'bố|mẹ|ông|bà|người già|người lớn tuổi',
    'màn hình lớn': r'màn hình (to|lớn)|màn to',
    'nhỏ gọn': r'nhỏ gọn|gọn nhẹ|cầm vừa tay|nhỏ nhẹ',
    'máy gập': r'gập|fold|flip',
}
OBJECTIONS = {
    'chê giá cao': r'đắt|mắc quá|mắc vậy|giá cao|chát|hơi mắc',
    'lo pin yếu': r'pin yếu|hao pin|tụt pin|pin kém',
    'lo máy nóng': r'nóng máy|máy nóng|bị nóng',
    'chê máy nặng/to': r'nặng quá|to quá|cồng kềnh|hơi nặng|hơi to',
    'chê camera': r'camera (xấu|kém|dở)|chụp (xấu|kém|dở)',
    'lo máy lỗi': r'hay lỗi|dễ lỗi|sợ lỗi|lo lỗi',
    'không thích màu': r'không thích màu|màu xấu',
}
NEGATIVE = r'bực|tức (quá|ghê|thật|chết)|tệ|chán|thất vọng|khó chịu|lừa đảo|vô lý|làm ăn kiểu|quá đáng|sao lâu vậy|chậm quá'
POSITIVE = r'cảm ơn|cám ơn|thanks|tuyệt|ok|oke|okela|thích|được đó|ưng|hài lòng|chốt'
HANDOFF = (r'gặp nhân viên|người thật|nói chuyện với người|tư vấn viên|khiếu nại|giao sai|giao thiếu|giao chậm|'
           r'chưa nhận được hàng|chưa giao|hủy đơn|huỷ đơn|đòi tiền|hoàn tiền ngay')
COMPARE = r'so sánh|khác gì|khác nhau|nên chọn|nên lấy cái nào|hơn không|với .+ thì sao'
CLOSING = r'chốt|lấy máy|lấy cái|mua luôn|đặt hàng|đặt mua|thanh toán|giao cho|ship cho|lên đơn'

def parse_budget(text: str) -> int | None:
    # "15 triệu", "15tr", "15 củ", "dưới 20 triệu", "từ 10 đến 15 triệu" -> ngân sách tối đa (VND)
    t = text.lower().replace(',', '.')
    nums = [float(n) for n, _ in re.findall(r'(\d+(?:\.\d+)?)\s*(triệu|tr\b|củ|m\b)', t)]
    if nums:
        return int(max(nums) * 1_000_000)
    m = re.search(r'\b(\d{7,9})\b', t.replace('.', ''))  # "15000000" hoặc "15.000.000"
    return int(m.group(1)) if m else None

def _has(pattern: str, text: str) -> bool:
    # Ranh giới từ: tránh "ông" khớp trong "không", "ip" khớp trong "chip"
    return re.search(rf'(?<!\w)(?:{pattern})(?!\w)', text) is not None

def sentiment_of(text: str) -> Sentiment:
    t = text.lower()
    if _has(NEGATIVE, t) or t.count('!') >= 2:
        return 'negative'
    t = re.sub(r'(không|chẳng|chưa) (thích|ưng|ok)', '', t)  # "không thích màu" không phải khen
    return 'positive' if _has(POSITIVE, t) else 'neutral'

def track_user_message(text: str) -> dict:
    """Cập nhật state từ 1 tin nhắn của khách (chưa tính stage, xem decide_stage)."""
    t = text.lower()
    slots: dict = {}
    budget = parse_budget(t)
    if budget:
        slots['budget_max'] = budget
    brands = [b for b, pat in BRANDS.items() if _has(pat, t)]
    if len(brands) == 1:
        slots['brand'] = brands[0]
    purposes = [p for p, pat in PURPOSES.items() if _has(pat, t)]
    if purposes:
        slots['purposes'] = purposes
    storage = re.search(r'\b(\d+)\s*(gb|tb)\b', t)
    if storage:
        slots['storage'] = f'{storage.group(1)}{storage.group(2).upper()}'
    return {
        'slots': slots,
        'objections': [o for o, pat in OBJECTIONS.items() if _has(pat, t)],
        'sentiment_history': [sentiment_of(t)],
        'wants_handoff': _has(HANDOFF, t),
        'wants_compare': _has(COMPARE, t),
        'wants_close': _has(CLOSING, t),
    }

# ---------------------------------------------------------------- trích từ kết quả tool

def _current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1:]

def _tool_results(messages: list[AnyMessage]):
    # (tool_call, nội dung kết quả) của các tool chạy trong lượt hiện tại
    recent = _current_turn(messages)
    calls = {tc['id']: tc for m in recent if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    for m in recent:
        if isinstance(m, ToolMessage) and m.tool_call_id in calls:
            yield calls[m.tool_call_id], str(m.content)

def track_tool_results(messages: list[AnyMessage]) -> dict:
    """Nhu cầu suy ra từ tham số tool model đã gọi trong lượt hiện tại (ngân sách, hãng, dung lượng, màu)."""
    slots = {}
    for call, _ in _tool_results(messages):
        args = call['args']
        if call['name'] in ('get_product_details', 'add_to_cart'):
            slots.update({k: args[k] for k in ('storage', 'color') if args.get(k)})
        elif call['name'] == 'search_products':
            if args.get('max_price'):
                slots.setdefault('budget_max', args['max_price'])
            if args.get('brand'):
                slots.setdefault('brand', args['brand'])
    return {'slots': slots}

def _candidate_products(messages: list[AnyMessage]) -> list[str]:
    # Tên sản phẩm xuất hiện trong kết quả tool của lượt này: danh sách search_products ("- Tên (Hãng): ..."),
    # dòng đầu get_product_details ("Tên (Hãng)"), dòng phụ kiện find_accessories ("- Tên: giá | kho")
    names = []
    for call, content in _tool_results(messages):
        if call['name'] == 'search_products':
            names += re.findall(r'^- (.+?) \((?:Apple|Samsung)\):', content, flags=re.M)
        elif call['name'] == 'get_product_details':
            m = re.fullmatch(r'(.+) \((?:Apple|Samsung)\)', content.splitlines()[0].strip()) if content else None
            names += [m.group(1)] if m else []
        elif call['name'] == 'find_accessories':
            names += re.findall(r'^- (.+?): [\d.]+đ', content, flags=re.M)
    return list(dict.fromkeys(names))

def products_mentioned(messages: list[AnyMessage]) -> list[str]:
    """Sản phẩm thật sự được giới thiệu: có trong kết quả tool của lượt này VÀ được nhắc trong câu trả lời cuối.
    (search_products trả cả chục máy cho model, nhưng khách chỉ thấy vài máy model chọn.)"""
    reply = next((m.text for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), '')
    text = reply.lower()
    found = []
    # "Galaxy S26" không được tính khi câu trả lời chỉ nói "Galaxy S26 Ultra"/"S26+"; "iPhone 17" không tính trong "iPhone 17e"
    for name in sorted(_candidate_products(messages), key=len, reverse=True):
        pat = re.escape(name.lower()) + r'(?![\w+])(?! (ultra|pro|plus|fe|max|edge|air)\b)'
        if re.search(pat, text):
            found.append(name)
    return found

def decide_stage(state: dict, wants_compare: bool = False, wants_close: bool = False) -> Stage:
    if state.get('handoff'):
        return 'handoff'
    if state.get('cart') or (wants_close and state.get('shown_products')):
        return 'closing'
    shown = state.get('shown_products') or []
    if len(shown) >= 2 and (wants_compare or state.get('stage') == 'comparing'):
        return 'comparing'
    return 'recommending' if shown else 'discovery'

def needs_handoff(state: dict, wants_handoff: bool) -> bool:
    recent = (state.get('sentiment_history') or [])[-NEGATIVE_STREAK:]
    angry = len(recent) == NEGATIVE_STREAK and all(s == 'negative' for s in recent)
    return bool(state.get('handoff') or wants_handoff or angry)

# ---------------------------------------------------------------- đưa state vào prompt

STAGE_GUIDE = {
    'discovery': 'Khách chưa xem máy nào. Nếu thiếu ngân sách hoặc nhu cầu chính, hỏi tối đa 1 câu ngắn trước khi gợi ý.',
    'recommending': 'Đã giới thiệu sản phẩm. Trả lời câu hỏi về máy đó; khách có vẻ ưng thì hỏi màu/dung lượng và mời thêm vào giỏ.',
    'comparing': 'Khách đang phân vân giữa các máy. So sánh ngắn theo đúng nhu cầu đã biết, đưa ra 1 gợi ý rõ ràng kèm lý do.',
    'closing': 'Khách đã chọn hàng. Xác nhận giỏ hàng và tổng tiền, hỏi hình thức nhận hàng và thanh toán (trả thẳng/trả góp).',
    'handoff': 'Cần chuyển nhân viên. Xin lỗi nếu khách không hài lòng, nêu bước cần làm theo chính sách (nếu có), '
               'rồi thông báo đã chuyển cho nhân viên hỗ trợ. Không cố bán thêm.',
}

def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'

def describe(state: dict) -> str:
    slots = state.get('slots') or {}
    need = []
    if slots.get('budget_max'):
        need.append(f"ngân sách tối đa {_price(slots['budget_max'])}")
    for k, label in (('brand', 'hãng'), ('storage', 'dung lượng'), ('color', 'màu')):
        if slots.get(k):
            need.append(f'{label} {slots[k]}')
    if slots.get('purposes'):
        need.append('nhu cầu: ' + ', '.join(slots['purposes']))
    cart = state.get('cart') or []
    cart_text = '; '.join(f"{i['quantity']} x {i['name']} {i.get('storage') or ''} {i.get('color') or ''}".strip()
                          + f" ({_price(i['price'])})" for i in cart)
    total = sum(i['price'] * i['quantity'] for i in cart)
    stage = state.get('stage') or 'discovery'
    lines = [
        f'Bước hiện tại: {stage}. {STAGE_GUIDE[stage]}',
        'Nhu cầu đã biết: ' + ('; '.join(need) if need else 'chưa có') + ' (không hỏi lại điều đã biết).',
        'Đã giới thiệu: ' + (', '.join(state.get('shown_products') or []) or 'chưa có'),
        'Giỏ hàng: ' + (f'{cart_text}. Tổng {_price(total)}' if cart else 'trống'),
    ]
    if state.get('objections'):
        lines.append('Khách đang lo ngại: ' + ', '.join(state['objections'])
                     + '. Trả lời thẳng vào lo ngại này bằng dữ liệu công cụ (vd gợi ý máy rẻ hơn, pin lớn hơn).')
    if (state.get('sentiment_history') or [])[-1:] == ['negative']:
        lines.append('Khách đang không vui: trả lời ngắn, xin lỗi, không quảng cáo thêm.')
    return '\n'.join(lines)
