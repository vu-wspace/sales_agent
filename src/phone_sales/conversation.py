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

def wanted_update(old: dict | None, new: dict | None) -> dict:
    # Phụ kiện khách đã yêu cầu mua {loại: tin nhắn của khách}. None: làm lại từ đầu (đơn đã đặt/hủy);
    # {'drop': [loại]}: khách bỏ món đó khỏi giỏ (remove_from_cart), không tự thêm lại
    if new is None:
        return {}
    out = dict(old or {})
    for kind in new.get('drop', []):
        out.pop(kind, None)
    out.update({k: v for k, v in new.items() if k != 'drop'})
    return out

def steps(old: list | None, new: list | None) -> list:
    # None: làm lại từ đầu (đơn đã đặt/đã hủy, giỏ trống -> lần mua sau bán kèm lại từ bước 1)
    return [] if new is None else append_unique(old, new)

def cart_update(old: list | None, new: list | dict | None) -> list:
    """Giỏ hàng. list: thay cả giỏ (api.py làm trống sau khi đặt/hủy đơn). dict: thao tác của tool giỏ hàng,
    {'add': món (quantity = số lượng thêm)} hoặc {'remove': [variant_id]}. Model dự phòng có thể gọi 2 tool giỏ hàng
    song song trong 1 bước (vd thêm máy và củ sạc): mỗi thao tác áp lên kết quả trước, không món nào bị mất
    (khóa không có reducer thì LangGraph báo InvalidUpdateError: Can receive only one value per step)."""
    if new is None:
        return list(old or [])
    if isinstance(new, list):
        return [dict(i) for i in new]
    cart = [dict(i) for i in (old or [])]
    if 'add' in new:
        item = new['add']
        same = next((i for i in cart if i['variant_id'] == item['variant_id']), None)
        if same:
            same['quantity'] += item['quantity']
        else:
            cart.append(dict(item))
    if 'remove' in new:
        cart = [i for i in cart if i['variant_id'] not in set(new['remove'])]
    return cart

def last(old, new):
    # Lấy giá trị mới nhất; khác khóa thường ở chỗ cho phép ghi nhiều lần trong 1 bước (tool chạy song song)
    return new

class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # lịch sử tin nhắn (nối thêm, không ghi đè)
    stage: Stage                                          # đang ở bước nào của quy trình bán hàng
    slots: Annotated[dict, merge_slots]                   # nhu cầu đã thu: budget_max, brand, purposes, storage...
    shown_products: Annotated[list[str], append_unique]   # sản phẩm đã giới thiệu
    cart: Annotated[list[dict], cart_update]              # giỏ hàng (tool trả thao tác thêm/bỏ, xem cart_update)
    objections: Annotated[list[str], append_unique]       # khách đang chê/lo ngại gì
    sentiment_history: Annotated[list[str], append]       # cảm xúc của khách qua từng lượt
    handoff: bool                                         # có cần chuyển nhân viên không (đã bật thì giữ)
    cross_sell: Annotated[list[str], steps]               # bước bán kèm đã làm: 'charger', 'accessories', 'confirm'
    declined: bool                                        # tin nhắn mới nhất của khách là từ chối ("không cần")
    order_form: Annotated[bool, last]                     # lượt này hiện form đặt hàng (show_order_form)
    orders: Annotated[list[str], append]                  # mã các đơn đã đặt trong cuộc trò chuyện (api.py: /order)
    objection: str | None                                 # phản đối trong tin nhắn lượt này: price, competitor, hesitation
    objection_type: str | None                            # loại phản đối đang xử lý (đổi loại thì làm lại từ bước đầu)
    objection_step: int                                   # đã xử lý loại đó bao nhiêu lượt (bậc thang khi khách do dự)
    offered_alternatives: Annotated[list[str], append_unique]  # máy rẻ hơn đã gợi ý (find_cheaper_alternatives không lặp lại)
    wanted: Annotated[dict, wanted_update]                # phụ kiện khách đã yêu cầu mua {loại: tin nhắn}, xem wanted_of

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
# Phản đối cần xử lý theo chiến lược (OBJECTION_GUIDE). Khác OBJECTIONS ở trên (lo ngại cụ thể để trả lời thẳng).
# So với nơi khác: không khớp "bên em" (chính cửa hàng)
COMPETITOR = (r'bên (?:kia|khác|đó|ngoài)|chỗ khác|shop khác|cửa hàng khác|tiệm khác|nơi khác|ngoài tiệm|'
              r'thế giới di động|tgdđ|tgdd|fpt shop|fpt|cellphones|cellphone s|hoàng hà|di động việt|'
              r'shopee|lazada|tiki|online rẻ hơn|trên mạng rẻ hơn')
# Do dự. Không dùng "để sau" (trùng câu từ chối bán kèm "củ sạc để sau")
HESITATION = (r'để (?:anh|chị|em|mình|tui|tôi|tớ) (?:suy nghĩ|tính|cân nhắc|xem lại|về hỏi|hỏi lại)|'
              r'suy nghĩ thêm|cân nhắc thêm|tính thêm|chưa chắc|phân vân|lưỡng lự|chưa quyết|đắn đo|'
              r'chưa biết có nên')

# Từ chối lời mời bán kèm. Không dùng "thôi" đứng riêng: "lấy luôn thôi" là đồng ý
DECLINE = (r'không cần|ko cần|k cần|khỏi|không lấy|không mua|chưa cần|để sau|thôi khỏi|thôi không|không đâu|'
           r'không ạ|không nhé|không em|vậy được rồi|vậy thôi')

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
        'declined': _has(DECLINE, t),
        'objection': objection_of(t),
        'wanted': wanted_of(text),
    }

# Khách liệt kê phụ kiện muốn mua ("đặt ip17, củ sạc 40W, ốp lưng silicon, kính cường lực belkin"): nhớ lại để
# thêm thẳng vào giỏ và không mời lại như gợi ý bán kèm. Chỉ tính khi có ý mua: "iphone 17 có củ sạc nào" là hỏi
BUY = r'mua|lấy|đặt|chốt|thêm|order|cho (?:tui|tôi|mình|em|anh|chị) (?:xin|luôn)?'
WANTED_KINDS = {
    'charger': r'củ sạc|cục sạc|bộ sạc|sạc nhanh|sạc \d+\s*w|adapter|tiếp hợp',
    'case': r'ốp|bao da',
    'screen-protector': r'cường lực|kính|miếng dán|dán màn',
    'cable': r'cáp|dây sạc',
    'power-bank': r'sạc dự phòng|pin dự phòng',
}
WANTED_LABELS = {'charger': 'củ sạc', 'case': 'ốp lưng', 'screen-protector': 'cường lực/miếng dán',
                 'cable': 'cáp sạc', 'power-bank': 'sạc dự phòng'}

def wanted_of(text: str) -> dict:
    t = text.lower()
    if not _has(BUY, t) or _has(DECLINE, t):
        return {}
    return {k: text.strip()[:300] for k, pat in WANTED_KINDS.items() if _has(pat, t)}

def _cart_kinds(cart: list[dict]) -> set:
    return {i.get('accessory_type') for i in cart if i.get('category') == 'accessory'}

def wanted_hint(state: dict) -> str | None:
    """Phụ kiện khách đã yêu cầu mua mà giỏ chưa có: thêm thẳng, không hỏi lại như lời mời bán kèm."""
    pending = {k: v for k, v in (state.get('wanted') or {}).items() if k not in _cart_kinds(state.get('cart') or [])}
    if not pending:
        return None
    asked = '; '.join(dict.fromkeys(pending.values()))
    labels = ', '.join(WANTED_LABELS.get(k, k) for k in pending)
    return (f"Khách đã yêu cầu mua kèm: {labels} (khách nói: \"{asked}\"). Đây là yêu cầu của khách, không phải gợi ý: "
            "gọi find_accessories cho máy khách chọn với đúng loại đó (có thể gọi song song), chọn món khớp mô tả của "
            "khách (công suất, chất liệu, hãng...) rồi add_to_cart luôn, kể cả khi máy chưa đủ dung lượng, màu (khi đó "
            "chỉ hỏi phiên bản máy). Chỉ hỏi lại khi có nhiều món khớp ngang nhau (vd nhiều màu) hoặc không có món nào "
            "khớp (nói thật, gợi ý món gần nhất). Không mời lại các loại này như gợi ý bán kèm.")

def objection_of(text: str) -> str | None:
    """Loại phản đối trong tin nhắn (đã viết thường): so với nơi khác > chê giá > do dự.
    "Phân vân giữa A và B" là so sánh (comparing), không phải do dự."""
    if _has(COMPETITOR, text):
        return 'competitor'
    if _has(OBJECTIONS['chê giá cao'], text):
        return 'price'
    if _has(HESITATION, text) and not _has(COMPARE, text):
        return 'hesitation'
    return None

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

def added_to_cart_this_turn(messages: list[AnyMessage]) -> bool:
    # Cả khi add_to_cart nằm cùng 1 message với tool đang chạy (model gọi song song)
    return any(tc['name'] == 'add_to_cart' for m in _current_turn(messages) if isinstance(m, AIMessage)
               for tc in (m.tool_calls or []))

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
    # Không kèm phí/thời gian giao, kỳ hạn trả góp: model tự bịa ("trả góp 3-24 tháng", chính sách là tối đa 12 tháng)
    'closing': 'Khách đã chọn hàng. Làm theo mục "Bán kèm" bên dưới nếu có; không có thì xác nhận giỏ hàng và tổng '
               'tiền, hỏi khách có chốt đơn không. Khách xác nhận đặt hàng thì gọi show_order_form (form hỏi tên, số điện thoại, địa chỉ, thanh toán), không hỏi các thông tin '
               'đó bằng chữ. Không tự nêu phí, thời gian giao, kỳ hạn hay đối tác trả góp; '
               'khách hỏi chi tiết thì gọi search_policy.',
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
    unhappy = (state.get('sentiment_history') or [])[-1:] == ['negative']
    if unhappy:
        lines.append('Khách đang không vui: trả lời ngắn, xin lỗi, không quảng cáo thêm.')
    wanted = wanted_hint(state)
    if wanted:
        lines.append(wanted)
    objection = objection_hint(state)
    if objection:
        lines.append(objection)
    added = added_hint(state)
    if added:
        lines.append(added)
    hint = cross_sell_hint(state)
    # Đã liệt kê các món vừa thêm kèm tổng giỏ: không cần thêm mẫu "Đơn hàng của anh/chị" của bước xác nhận
    if hint and not unhappy and not (added and cross_sell_step(state) == 'confirm'):
        lines.append(hint)
    return '\n'.join(lines)

ADDED_RE = re.compile(r'^Đã thêm (\d+) x (.+?) \((\d[\d.]*đ)\)\.')

def added_this_turn(messages: list[AnyMessage]) -> list[dict]:
    """Các món add_to_cart thêm thành công trong lượt này (đọc từ kết quả tool: đúng tên, phiên bản, giá)."""
    items = []
    for call, content in _tool_results(messages):
        m = ADDED_RE.match(content) if call['name'] == 'add_to_cart' else None
        if m:
            items.append({'qty': int(m.group(1)), 'name': ' '.join(m.group(2).split()), 'price': m.group(3)})
    return items

def added_hint(state: dict) -> str | None:
    """Khách mua nhiều món cùng lúc ("bán cho tui ip17 512gb đen, củ sạc 40w, kính belkin"): liệt kê từng món
    vừa thêm rồi mới chốt 1 câu tổng, không chỉ nói gộp "đã thêm 3 món"."""
    items = added_this_turn(state.get('messages') or [])
    if len(items) < 2:
        return None
    cart = state.get('cart') or []
    total = _price(sum(i['price'] * i['quantity'] for i in cart))
    missing = [WANTED_LABELS.get(k, k) for k in (state.get('wanted') or {}) if k not in _cart_kinds(cart)]
    lines = '\n'.join(f"- {i['qty']} x **{i['name']}**: {i['price']}" if i['qty'] > 1 else
                      f"- **{i['name']}**: {i['price']}" for i in items)
    done = (f"Đã thêm {len(items)} món anh/chị yêu cầu vào giỏ, còn {', '.join(missing)} em chưa thêm được "
            "[nói ngắn lý do theo kết quả tool]." if missing else
            f"Đã thêm đủ {len(items)} món anh/chị yêu cầu vào giỏ.")
    return ("Lượt này đã thêm nhiều món vào giỏ: liệt kê đúng từng món như dưới (không gộp, không đổi tên, giá) rồi "
            "mới chốt 1 câu tổng. Trả lời theo mẫu:\n"
            f"Dạ, em đã thêm vào giỏ:\n{lines}\n\n{done} Tổng cộng **{total}**.\n\n"
            "[1 câu tiếp theo: gợi ý theo mục \"Bán kèm\" nếu có, không thì hỏi khách xác nhận đặt hàng để mở form]")

# Khoảng giá máy thay thế theo % (không dùng số tiền cố định: "rẻ hơn 5-10 triệu" với máy 7 triệu là tụt phân khúc
# hoặc không có máy nào). Khách đã nói ngân sách thì find_cheaper_alternatives lấy ngân sách làm trần
DROP_PCT = {'price': (0.15, 0.35), 'hesitation': (0.07, 0.15)}
HESITATION_STEPS = 3  # trả góp 0% -> giá trị máy -> máy rẻ hơn; quá số này thì thôi thuyết phục

def _cheaper_call(kind: str) -> str:
    low, high = DROP_PCT[kind]
    return (f"find_cheaper_alternatives(product=<máy khách đang chê/phân vân>, min_drop_pct={low}, "
            f"max_drop_pct={high})")

def _value_text(state: dict) -> str:
    purposes = (state.get('slots') or {}).get('purposes') or []
    need = f" gắn với nhu cầu khách đã nói ({', '.join(purposes)})" if purposes else ''
    return (f"nêu 1-2 giá trị của máy khách đang xem{need}, chỉ dùng thông số từ get_product_details (máy chưa có "
            "thông số thì chỉ nói về dung lượng, dòng máy, chính sách bảo hành); không tự liệt kê ưu điểm theo trí nhớ")

def objection_hint(state: dict) -> str | None:
    """Hướng dẫn xử lý phản đối của lượt này (state['objection'] do graph.track_user đặt mỗi lượt).
    price: ghi nhận + giá trị + 2-3 máy rẻ hơn 15-35%. competitor: ưu điểm cửa hàng theo chính sách thật.
    hesitation: bậc thang qua nhiều lượt (trả góp 0% -> giá trị -> máy rẻ hơn 7-15%), sau đó thôi thuyết phục."""
    kind = state.get('objection')
    if not kind:
        return None
    step = state.get('objection_step') or 0
    if kind == 'price':
        return ("Xử lý phản đối (chê giá): ghi nhận ngắn băn khoăn về giá, không phủ nhận, không tự giảm giá. "
                f"Rồi {_value_text(state)}. Sau đó gọi {_cheaper_call('price')} và giới thiệu 2-3 máy kèm giá, số "
                "tiền rẻ hơn và điểm đánh đổi trung thực (dung lượng thấp hơn, dòng thấp hơn; thông số chỉ theo dữ "
                "liệu). Tool không có máy nào thì nói thật, không gợi ý máy quá rẻ; có thể nhắc dung lượng thấp "
                "hơn của chính máy đó (nếu tool liệt kê), khuyến mãi đang áp dụng (get_promotions) hoặc trả góp "
                "0% (search_policy), chỉ đúng theo kết quả tool."
                + (" Khách vẫn chê giá: tool tự bỏ máy đã gợi ý, không giới thiệu lại máy cũ." if step else ''))
    if kind == 'competitor':
        return ("Xử lý phản đối (so với nơi khác): không chê bai, không bình luận về bên kia, không xác nhận hay "
                "phủ nhận giá của bên đó (không kiểm chứng được; cần thì nói nhẹ 1 câu kiểu \"giá bên đó em không nắm "
                "được\", không nói \"không xác nhận hay phủ nhận\"), không hứa khớp giá hay giảm giá. Gọi "
                "search_policy về bảo hành, đổi trả (và trả góp 0% nếu khách quan tâm giá), rồi nêu 2-3 điểm của "
                "cửa hàng đúng theo kết quả (hàng chính hãng, đổi trả, bảo hành, hậu mãi); không nói chung chung "
                "kiểu \"uy tín nhất\". Giá ở cửa hàng chỉ nêu theo dữ liệu công cụ.")
    if step >= HESITATION_STEPS:
        return ("Xử lý phản đối (do dự, đã gợi ý đủ 3 bước): không thuyết phục thêm, không gợi ý thêm máy. Tôn "
                "trọng quyết định, nói khách cứ thong thả, cần gì cứ nhắn; có thể hỏi khách có muốn gặp nhân viên "
                "tư vấn trực tiếp không.")
    steps = [
        # Khuyến mãi chỉ lấy từ bảng promotions (get_promotions); không có thì bỏ qua, không bịa
        "bước 1/3: gọi get_promotions(product=<máy khách đang xem>); có khuyến mãi thì nêu ngắn 1 khuyến mãi phù "
        "hợp nhất đúng theo kết quả (mức giảm, điều kiện, hạn, mã nhập ở form đặt hàng); không có thì không nhắc "
        "khuyến mãi, giảm giá, quà tặng nào. Muốn nhắc trả góp 0% thì phải gọi search_policy trong lượt này rồi "
        "nêu đúng theo kết quả. Nói khách cứ thong thả, không ép",
        f"bước 2/3: {_value_text(state)}; không ép",
        f"bước 3/3: gọi {_cheaper_call('hesitation')}, giới thiệu 2-3 máy kèm giá, số tiền rẻ hơn và điểm đánh "
        "đổi trung thực; tool không có máy nào thì nói thật. Không ép",
    ]
    return f"Xử lý phản đối (khách do dự), {steps[step]}."

CROSS_SELL_ACCESSORIES = {'case', 'screen-protector'}

def _phone_in_cart(cart: list[dict]) -> dict | None:
    # Giỏ tạo trước khi có trường category thì coi là điện thoại
    return next((i for i in cart if i.get('category', 'phone') == 'phone'), None)

def _needs_charger(cart: list[dict]) -> bool:
    phone = _phone_in_cart(cart)
    in_cart = {i.get('accessory_type') for i in cart if i.get('category') == 'accessory'}
    return phone.get('charger_in_box') is False and 'charger' not in in_cart

def cross_sell_step(state: dict) -> str | None:
    """Bước bán kèm của lượt này khi giỏ có điện thoại (bước closing), mỗi lượt mời 1 loại:
    'charger': lượt chọn máy, máy không kèm củ sạc -> mời củ sạc (chưa mời ốp lưng, cường lực).
    'accessories': khách lấy củ sạc hoặc không -> mời ốp lưng, cường lực (máy đã kèm củ sạc thì bắt đầu từ đây;
                   khách đã tự lấy ốp/cường lực thì bỏ qua).
    'confirm': khách lấy phụ kiện hoặc không -> xác nhận đơn (giỏ hàng, tổng tiền), không mời thêm.
    Bước đã làm lưu trong state['cross_sell'] (graph.track_reply ghi cuối mỗi lượt)."""
    cart = state.get('cart') or []
    phone = _phone_in_cart(cart)
    unhappy = (state.get('sentiment_history') or [])[-1:] == ['negative']
    # Lượt khách chê giá, so giá, do dự: không mời bán kèm (không ghi là đã mời), lượt sau mời tiếp
    if state.get('stage') != 'closing' or not phone or unhappy or state.get('objection'):
        return None
    done = set(state.get('cross_sell') or [])
    in_cart = {i.get('accessory_type') for i in cart if i.get('category') == 'accessory'}
    # Loại khách đã tự yêu cầu mua (wanted) thì không mời như gợi ý: thêm thẳng theo wanted_hint
    wanted = set(state.get('wanted') or {})
    if not done & {'charger', 'accessories'} and _needs_charger(cart) and 'charger' not in wanted:
        return 'charger'
    if 'accessories' not in done and not (in_cart | wanted) & CROSS_SELL_ACCESSORIES:
        return 'accessories'
    if 'confirm' not in done:
        return 'confirm'
    return None

# Bán kèm chỉ là gợi ý nhẹ: khách không lấy cũng được, không thúc giục
LIGHT_OFFER = ('Gợi ý nhẹ, không ép: không thúc giục, không nói "nên mua", "đừng bỏ lỡ", không khen quá lời; '
               'khách không lấy thì thôi, không hỏi lại.')

def cross_sell_hint(state: dict) -> str | None:
    """Hướng dẫn + mẫu câu trả lời cho bước bán kèm của lượt này (xem cross_sell_step)."""
    step = cross_sell_step(state)
    if step is None:
        return None
    cart = state.get('cart') or []
    phone = _phone_in_cart(cart)
    name = phone['name']
    # Mẫu trả lời: chỉ dẫn cho model đặt TRƯỚC "Trả lời theo mẫu", trong mẫu chỉ có chữ gửi khách và [chỗ điền];
    # chỉ dẫn nằm trong dòng mẫu thì model chép nguyên văn cho khách. Định dạng theo quy tắc 11b (prompts.py)
    added = "Dạ, em đã thêm **[máy, dung lượng, màu]** ([giá]) vào giỏ rồi nha."
    item = "- **[tên món]**: [giá] – [ghi chú ngắn nếu tool có, vd đủ công suất sạc nhanh]"
    if step == 'charger':
        if phone.get('cable_in_box'):
            # Hộp đã có cáp: chỉ mời củ sạc (tool bỏ loại kèm cáp), vài lựa chọn từ rẻ tới đắt
            call, box = 'củ sạc', f"Hộp {name} đã có cáp sạc nhưng chưa kèm củ sạc."
            pick = ("Mời 2-3 củ sạc chính hãng từ rẻ tới đắt (tool chỉ có 1 thì mời 1), không mời cáp; dòng tool có "
                    "ghi công suất thì giữ ghi chú ngắn củ nào đủ sạc nhanh, củ nào chỉ sạc thường.")
            group = 'củ sạc chính hãng'
        else:
            call, box = 'sạc, cáp', f"{name} không kèm củ sạc trong hộp."
            pick = ("Mời 1 củ sạc và 1 cáp chính hãng (củ sạc đã kèm cáp thì không cần cáp); nói đúng như dòng "
                    "'Trong hộp'.")
            group = 'củ sạc, cáp chính hãng'
        return (f"Bán kèm (bước 1/3, củ sạc): gọi find_accessories(phone='{name}', accessory_type='{call}', "
                f"genuine_only=True). {pick} {LIGHT_OFFER} Chỉ gợi ý củ sạc trong lượt này, chưa gợi ý ốp lưng, "
                "cường lực; chưa gọi show_order_form. Trả lời theo mẫu:\n"
                f"{added}\n\n{box} Nếu anh/chị cần thì bên em có {group}:\n{item}\n\n"
                "Anh/chị không cần thì cũng không sao, cứ nói em một tiếng nha.")
    if step == 'accessories':
        done = set(state.get('cross_sell') or [])
        first = 'charger' not in done  # máy đã kèm củ sạc: lượt chọn máy mời luôn ốp lưng, cường lực
        opening = (f"{added}\n\n" if first else
                   "[1 câu: đã thêm món khách chọn và tổng giỏ; khách không lấy gì thì bỏ câu này]\n\n")
        return (f"Bán kèm (bước {'1/2' if first else '2/3'}, ốp lưng, cường lực): "
                + ('' if first else "khách chọn củ sạc trong tin nhắn thì add_to_cart món đó. ")
                + f"Gọi find_accessories(phone='{name}', accessory_type='ốp lưng, cường lực', genuine_only=True), "
                f"gợi ý 2-3 món. {LIGHT_OFFER} Món không ghi 'chính hãng' (vd miếng dán Belkin cho iPhone) thì ghi "
                "đúng tên hãng trong ngoặc, không gọi là chính hãng; không nhắc lại củ sạc. Tool không có món nào thì "
                "nói ngắn chưa có ốp lưng, cường lực cho máy này rồi hỏi khách xác nhận đặt hàng. Chưa gọi "
                "show_order_form, kể cả khi khách nói xác nhận. Trả lời theo mẫu:\n" + opening
                + f"Anh/chị muốn giữ {name} kỹ hơn thì có thể tham khảo thêm ốp lưng, cường lực bên em:\n{item}\n\n"
                "Không cần thì anh/chị xác nhận đặt hàng giúp em luôn nha.")
    return ("Bán kèm đã xong, không mời thêm phụ kiện. Khách chọn món nào trong tin nhắn thì add_to_cart món "
            "đó. Khách đã nói xác nhận/chốt đơn trong tin nhắn này mà không thêm món mới thì gọi show_order_form. "
            "Giỏ hàng, tổng tiền lấy từ trạng thái hoặc kết quả tool. Trả lời theo mẫu:\n"
            "Đơn hàng của anh/chị:\n"
            "- [số lượng] x **[tên, dung lượng, màu]**: [giá]\n\n"
            "Tổng cộng: **[tổng tiền]**\n\n"
            "Anh/chị xác nhận đặt hàng giúp em, em mở form để mình điền thông tin nhận hàng nha.")
