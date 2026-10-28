import re
import unicodedata
import psycopg
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from psycopg.rows import dict_row
from typing import Annotated, Any
from bank_faqs.config import DATABASE_URL, POLICY_MIN_SCORE, POLICY_SCORE_MARGIN, POLICY_TOP_K
from bank_faqs.prompts import POLICY_GUIDE, POLICY_PROMPTS
from bank_faqs.vector_db import get_vector_store
from tavily import TavilyClient
from functools import lru_cache

tavily_client = TavilyClient()

MAX_LINES = 15  # số sản phẩm tối đa trả về cho model, tránh kết quả quá dài

# Cách khách hay gọi tắt -> từ có trong tên sản phẩm (đã chuẩn hóa bằng _norm_text)
ALIASES = {'ip': 'iphone', 'ss': 'samsung', 'promax': 'pro max', 'zfold': 'z fold', 'zflip': 'z flip'}
# Bỏ khi so khớp tên: khách nói "samsung s26", "điện thoại galaxy a57" nhưng tên trong DB là "Galaxy S26", "Galaxy A57 5G"
STOP_WORDS = {'galaxy', 'samsung', 'apple', '5g', 'dien', 'thoai', 'may', 'cho', 'cua'}

# Loại phụ kiện (attributes->>'accessory_type') và các từ khách hay dùng
ACCESSORY_TYPES = {
    'case': ['op', 'case', 'bao da', 'op lung'],
    'screen-protector': ['dan', 'cuong luc', 'mieng dan', 'film'],
    'charger': ['sac', 'cu sac', 'adapter', 'sac khong day'],
    'cable': ['cap', 'day sac'],
    'power-bank': ['pin du phong', 'sac du phong', 'power bank'],
    's-pen': ['s pen', 'but'],
    'wallet': ['vi'],
    'holder': ['gia do', 'tok'],
}

# Mục thông số đưa vào câu trả lời (tên mục khác nhau giữa trang Apple và Samsung)
SPEC_KEYS = [
    'Màn Hình', 'Kích cỡ (Màn hình chính)', 'Công nghệ màn hình (màn hình chính)', 'Chip', 'Bộ vi xử lý',
    'Camera', 'Camera sau - Độ phân giải (Multiple)', 'Camera Trước', 'Camera trước - Độ phân giải',
    'Pin Và Nguồn Điện', 'Dung lượng pin (mAh, Typical)', 'Chống Tia Nước, Chống Nước, Và Chống Bụi',
    'Trọng lượng (g)',
]

DB_ERROR = ('Lỗi hệ thống khi tra cứu kho hàng ({}). Không tự đưa ra giá hay tình trạng hàng; '
            'xin lỗi anh/chị và đề nghị liên hệ hotline hoặc cửa hàng gần nhất.')

@lru_cache
def _policy_store():
    return get_vector_store()

def _relevant_policy_hits(scored: list[tuple]) -> list[tuple]:
    # scored: [(Document, điểm cosine)] giảm dần. Bỏ đoạn dưới ngưỡng, và đoạn kém đoạn tốt nhất quá MARGIN
    if not scored or scored[0][1] < POLICY_MIN_SCORE:
        return []
    best = scored[0][1]
    return [(d, s) for d, s in scored if s >= POLICY_MIN_SCORE and best - s <= POLICY_SCORE_MARGIN]

def _norm_text(text: str) -> str:
    text = unicodedata.normalize('NFD', text.lower().replace('đ', 'd').replace('+', ' plus '))
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()

def _tokens(text: str) -> set[str]:
    words = ' '.join(ALIASES.get(w, w) for w in _norm_text(text).split())
    words = re.sub(r'\b(\d+)(pro|max|plus|ultra|fe)\b', r'\1 \2', words)  # "17pro" -> "17 pro"
    return {w for w in words.split() if w not in STOP_WORDS and not re.fullmatch(r'\d+(gb|tb)', w)}

def _query(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row) as conn:
        return conn.execute(sql, params).fetchall()

def _match(products: list[dict], name: str, prefer_exact: bool = True) -> list[dict]:
    # Mọi từ khách gõ phải có trong tên sản phẩm. prefer_exact: có tên trùng khớp hoàn toàn thì chỉ lấy tên đó
    # ("iPhone 18 Pro" không kéo theo "iPhone 18 Pro Max"); tắt khi tìm cả dòng máy ("s26" -> S26, S26+, S26 Ultra...)
    want = _tokens(name)
    if not want:
        return []
    hits = [p for p in products if want <= _tokens(p['name'])]
    exact = [p for p in hits if _tokens(p['name']) == want]
    return (exact if prefer_exact else []) or hits

def _storage_key(s: str) -> tuple:
    # "64GB" < "256GB" < "1TB" < "2TB"; nhãn kèm RAM ("256GB - RAM 12GB") xếp theo dung lượng rồi tới RAM
    nums = [(int(n) * (1024 if u.upper() == 'TB' else 1)) for n, u in re.findall(r'(\d+)\s*(GB|TB)', s, re.I)]
    return tuple(nums) or (0,)

def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'

def _stock(n: int) -> str:
    if n <= 0:
        return 'hết hàng'
    return f'sắp hết (còn {n})' if n <= 5 else f'còn {n}'

def _accessory_type(text: str) -> str | None:
    t = _norm_text(text)
    if t in ACCESSORY_TYPES:
        return t
    # So từ dài trước: "sac du phong" phải ra power-bank chứ không phải charger
    pairs = sorted(((w, k) for k, ws in ACCESSORY_TYPES.items() for w in ws), key=lambda p: -len(p[0]))
    return next((k for w, k in pairs if re.search(rf'\b{w}\b', t)), None)

@tool
def search_products(query: str = '', brand: str = '', max_price: int = 0, category: str = 'phone') -> str:
    """Tìm sản phẩm đang bán tại cửa hàng (dữ liệu kho nội bộ): khoảng giá, các dung lượng, tình trạng hàng.
    query: tên/dòng máy (vd 'iphone 18', 's26', 'z fold'), để trống để liệt kê tất cả.
    brand: 'Apple' hoặc 'Samsung'. max_price: ngân sách tối đa (VND, vd 15000000), 0 = không giới hạn.
    category: 'phone' (điện thoại) hoặc 'accessory' (phụ kiện).
    Dùng khi khách hỏi có bán máy nào, máy nào trong tầm giá, so sánh giá các dòng."""
    try:
        rows = _query(
            """SELECT p.product_id, p.name, p.brand, min(v.price) AS min_price, max(v.price) AS max_price,
                      sum(v.stock) AS stock, array_agg(DISTINCT v.storage) AS storages
               FROM products p JOIN variants v USING (product_id)
               WHERE p.category = %s AND (%s = '' OR p.brand ILIKE %s)
               GROUP BY p.product_id, p.name, p.brand ORDER BY min(v.price)""",
            (category, brand.strip(), brand.strip()))
    except Exception as e:
        return DB_ERROR.format(type(e).__name__)
    if query.strip():
        rows = _match(rows, query, prefer_exact=False)
    if max_price > 0:
        rows = [r for r in rows if r['min_price'] <= max_price]
    if not rows:
        return (f"Không có sản phẩm khớp (tên: '{query}', hãng: '{brand}', giá tối đa: "
                f"{_price(max_price) if max_price else 'không giới hạn'}).")
    lines = []
    for r in rows[:MAX_LINES]:
        price = _price(r['min_price']) if r['min_price'] == r['max_price'] else \
            f"{_price(r['min_price'])} - {_price(r['max_price'])}"
        storages = ', '.join(sorted((s for s in r['storages'] if s), key=_storage_key))
        lines.append(f"- {r['name']} ({r['brand']}): {price}"
                     + (f' | {storages}' if storages else '') + f" | {_stock(r['stock'])}")
    more = f'\n(còn {len(rows) - MAX_LINES} sản phẩm khác, hãy hỏi cụ thể hơn)' if len(rows) > MAX_LINES else ''
    return 'Sản phẩm đang bán (giá đã gồm VAT):\n' + '\n'.join(lines) + more

@tool
def get_product_details(product: str, storage: str = '', color: str = '') -> str:
    """Xem chi tiết 1 điện thoại: giá và tồn kho từng phiên bản (màu, dung lượng), RAM, thông số chính.
    product: tên máy (vd 'iPhone 18 Pro Max', 'Galaxy S26 Ultra', 'A57').
    storage (vd '256GB') và color (vd 'Đen') để lọc phiên bản, có thể để trống.
    Dùng khi khách hỏi giá chính xác, còn màu/dung lượng nào, thông số máy."""
    try:
        products = _query("SELECT product_id, name, brand, description, attributes FROM products WHERE category = 'phone'")
        hits = _match(products, product)
        if len(hits) != 1:
            names = ', '.join(p['name'] for p in hits[:MAX_LINES])
            return (f"Có nhiều máy khớp '{product}': {names}. Hỏi lại khách muốn xem máy nào." if hits else
                    f"Không tìm thấy điện thoại '{product}'. Dùng search_products để xem danh sách máy đang bán.")
        p = hits[0]
        variants = _query('SELECT variant_id, color, storage, price, stock FROM variants '
                          'WHERE product_id = %s', (p['product_id'],))
        variants.sort(key=lambda v: (_storage_key(v['storage'] or ''), v['price'], v['color'] or ''))
    except Exception as e:
        return DB_ERROR.format(type(e).__name__)

    attrs = p['attributes'] or {}
    if storage.strip():
        # "256gb" khớp cả "256GB - RAM 8GB" và "256GB - RAM 12GB"
        want = _storage_key(storage)[:1]
        variants = [v for v in variants if _storage_key(v['storage'] or '')[:1] == want]
    if color.strip():
        variants = [v for v in variants if _norm_text(color) in _norm_text(v['color'] or '')]
    estimated = set(attrs.get('estimated_price_skus', []))
    lines = [f"{p['name']} ({p['brand']})", p['description'] or '']
    if attrs.get('ram_by_storage'):
        lines.append('RAM: ' + ', '.join(f'{s}: {r}' for s, r in attrs['ram_by_storage'].items()))
    specs = attrs.get('specs') or {}
    for k in SPEC_KEYS:
        if specs.get(k):
            lines.append(f'{k}: {specs[k][:220]}')
    lines.append('Phiên bản (dung lượng | màu | giá | kho):' if variants else
                 f"Không có phiên bản khớp (dung lượng: '{storage}', màu: '{color}').")
    lines += [f"- {v['storage']} | {v['color']} | {_price(v['price'])}"
              f"{' (giá tham khảo)' if v['variant_id'] in estimated else ''} | {_stock(v['stock'])}"
              for v in variants]
    return '\n'.join(line for line in lines if line)

@tool
def find_accessories(phone: str, accessory_type: str = '') -> str:
    """Tìm phụ kiện tương thích với 1 điện thoại: ốp lưng, miếng dán màn hình, củ sạc, cáp, sạc dự phòng,
    S Pen, ví, giá đỡ. phone: tên máy (vd 'iPhone 17', 'Galaxy S26 Ultra').
    accessory_type: loại phụ kiện (vd 'ốp lưng', 'sạc', 'cáp', 'cường lực'), để trống để xem tất cả loại."""
    kind = _accessory_type(accessory_type) if accessory_type.strip() else None
    if accessory_type.strip() and not kind:
        return f"Không rõ loại phụ kiện '{accessory_type}'. Các loại có: ốp lưng, miếng dán, sạc, cáp, sạc dự phòng, S Pen, ví, giá đỡ."
    try:
        phones = _query("SELECT product_id, name FROM products WHERE category = 'phone'")
        hits = _match(phones, phone)
        if len(hits) != 1:
            names = ', '.join(p['name'] for p in hits[:MAX_LINES])
            return (f"Có nhiều máy khớp '{phone}': {names}. Hỏi lại khách dùng máy nào." if hits else
                    f"Không tìm thấy điện thoại '{phone}' trong danh sách máy cửa hàng đang bán.")
        rows = _query(
            """SELECT a.name, a.attributes->>'accessory_type' AS kind, v.price, v.stock
               FROM compatibility c JOIN products a ON a.product_id = c.accessory_id
               JOIN variants v ON v.product_id = a.product_id
               WHERE c.phone_id = %s AND (%s::text IS NULL OR a.attributes->>'accessory_type' = %s)
               ORDER BY v.stock = 0, v.price""",
            (hits[0]['product_id'], kind, kind))
    except Exception as e:
        return DB_ERROR.format(type(e).__name__)
    if not rows:
        return f"Chưa có phụ kiện {accessory_type or ''} tương thích với {hits[0]['name']}.".replace('  ', ' ')
    lines = [f"- {r['name']}: {_price(r['price'])} | {_stock(r['stock'])}" for r in rows[:MAX_LINES]]
    more = f'\n(còn {len(rows) - MAX_LINES} phụ kiện khác)' if len(rows) > MAX_LINES else ''
    return f"Phụ kiện tương thích với {hits[0]['name']}:\n" + '\n'.join(lines) + more

@tool
def search_policy(query: str) -> str:
    """Tra cứu chính sách của cửa hàng từ dữ liệu nội bộ: đổi trả, hoàn tiền, giao hàng, nhận tại
    cửa hàng, trả góp (thẻ tín dụng, công ty tài chính), bảo hành Apple và Samsung.
    Dùng cho câu hỏi về điều kiện, thời hạn, phí, thủ tục. Không dùng để tra giá hay tồn kho sản phẩm."""
    try:
        scored = _policy_store().similarity_search_with_score(query, k=POLICY_TOP_K)
    except Exception as e:
        # ToolNode chỉ bắt lỗi sai tham số; lỗi khác (Qdrant sập...) sẽ làm hỏng cả lượt hội thoại
        return (f'Lỗi hệ thống khi tra cứu dữ liệu nội bộ ({type(e).__name__}). '
                'Không tự đưa ra chính sách; xin lỗi anh/chị và đề nghị liên hệ hotline hoặc cửa hàng.')
    hits = _relevant_policy_hits(scored)
    if not hits:
        return ('Không tìm thấy chính sách liên quan tới câu hỏi trong dữ liệu nội bộ. Không tự trả lời về chính sách; '
                'nói chưa có thông tin và đề nghị liên hệ hotline hoặc cửa hàng.')
    # Gom đoạn theo file (thứ tự theo đoạn liên quan nhất), mỗi file kèm prompt riêng để model trả lời đúng phạm vi
    by_source: dict[str, list] = {}
    for doc, _ in hits:
        by_source.setdefault(doc.metadata['source'], []).append(doc)
    parts = [POLICY_GUIDE]
    for source, docs in by_source.items():
        guide = POLICY_PROMPTS.get(source)
        parts.append(f'=== [{source}] ===' + (f'\nHướng dẫn trả lời: {guide}' if guide else '')
                     + '\n' + '\n\n'.join(d.page_content for d in docs))
    return '\n\n'.join(parts)

@tool
def web_search(query: str) -> dict[str, Any] | str:
    """Tìm kiếm thông tin trên web: đánh giá, so sánh chi tiết, tin tức về điện thoại, máy sắp ra mắt.
    Chỉ dùng khi các công cụ dữ liệu nội bộ không có câu trả lời.
    Không dùng để tra giá, tồn kho (dùng search_products / get_product_details) hay chính sách cửa hàng (dùng search_policy)."""
    try:
        return tavily_client.search(query, max_results=3, timeout=30)
    except Exception as e:
        return (f'Lỗi hệ thống khi tìm kiếm trên web ({type(e).__name__}). '
                'Không tự đưa ra số liệu; xin lỗi anh/chị và đề nghị hỏi lại sau.')

# ---------------------------------------------------------------- giỏ hàng (chỉ graph.py có state 'cart')

def _cart_reply(tool_call_id: str, text: str, cart: list | None = None) -> Command:
    update = {'messages': [ToolMessage(text, tool_call_id=tool_call_id)]}
    if cart is not None:
        update['cart'] = cart
    return Command(update=update)

@tool
def add_to_cart(product: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
                storage: str = '', color: str = '', quantity: int = 1) -> Command:
    """Thêm 1 sản phẩm (điện thoại hoặc phụ kiện) vào giỏ hàng khi khách đồng ý mua.
    product: tên sản phẩm (vd 'iPhone 17', 'Ốp Lưng Trong Suốt Galaxy S26 Ultra').
    storage, color: phiên bản điện thoại khách chọn (bắt buộc nếu máy có nhiều phiên bản). quantity: số lượng.
    Chỉ gọi khi khách đã nói muốn mua/lấy/chốt; không tự thêm khi khách mới hỏi giá."""
    if quantity < 1:
        return _cart_reply(tool_call_id, 'Số lượng phải từ 1 trở lên.')
    try:
        products = _query('SELECT product_id, name, category FROM products')
        hits = _match(products, product)
        if len(hits) != 1:
            names = ', '.join(p['name'] for p in hits[:MAX_LINES])
            return _cart_reply(tool_call_id, f"Có nhiều sản phẩm khớp '{product}': {names}. Hỏi lại khách." if hits
                               else f"Không tìm thấy sản phẩm '{product}'.")
        p = hits[0]
        variants = _query('SELECT variant_id, color, storage, price, stock FROM variants WHERE product_id = %s',
                          (p['product_id'],))
    except Exception as e:
        return _cart_reply(tool_call_id, DB_ERROR.format(type(e).__name__))
    if storage.strip():
        want = _storage_key(storage)[:1]
        variants = [v for v in variants if _storage_key(v['storage'] or '')[:1] == want]
    if color.strip():
        variants = [v for v in variants if _norm_text(color) in _norm_text(v['color'] or '')]
    if len(variants) != 1:
        options = ', '.join(sorted({f"{v['storage'] or ''} {v['color'] or ''}".strip() for v in variants})[:MAX_LINES])
        return _cart_reply(tool_call_id, f"Cần khách chọn rõ phiên bản của {p['name']}: {options}." if variants
                           else f"{p['name']} không có phiên bản dung lượng '{storage}', màu '{color}'.")
    v = variants[0]
    cart = [dict(i) for i in (state.get('cart') or [])]
    item = next((i for i in cart if i['variant_id'] == v['variant_id']), None)
    new_qty = quantity + (item['quantity'] if item else 0)
    if v['stock'] < new_qty:
        return _cart_reply(tool_call_id, f"{p['name']} {v['storage'] or ''} {v['color'] or ''} "
                           f"{'hết hàng' if v['stock'] <= 0 else 'không đủ số lượng'}. Gợi ý phiên bản khác.")
    if item:
        item['quantity'] = new_qty
    else:
        cart.append({'variant_id': v['variant_id'], 'name': p['name'], 'storage': v['storage'],
                     'color': v['color'], 'price': v['price'], 'quantity': quantity})
    total = sum(i['price'] * i['quantity'] for i in cart)
    return _cart_reply(tool_call_id, f"Đã thêm {quantity} x {p['name']} {v['storage'] or ''} {v['color'] or ''} "
                       f"({_price(v['price'])}). Giỏ có {len(cart)} sản phẩm, tổng {_price(total)}.", cart)

@tool
def remove_from_cart(product: str, state: Annotated[dict, InjectedState],
                     tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Bỏ 1 sản phẩm khỏi giỏ hàng khi khách đổi ý. product: tên sản phẩm trong giỏ."""
    cart = state.get('cart') or []
    want = _tokens(product)
    keep = [i for i in cart if not (want and want <= _tokens(f"{i['name']} {i['storage'] or ''} {i['color'] or ''}"))]
    if len(keep) == len(cart):
        return _cart_reply(tool_call_id, f"Không có '{product}' trong giỏ hàng.")
    total = sum(i['price'] * i['quantity'] for i in keep)
    return _cart_reply(tool_call_id, f'Đã bỏ {len(cart) - len(keep)} sản phẩm. Giỏ còn {len(keep)} sản phẩm, '
                       f'tổng {_price(total)}.', keep)

# Tool của từng agent. Mô tả tool mà model nhìn thấy lấy từ docstring của hàm @tool ở trên.
TOOLS = {t.name: t for t in (search_products, get_product_details, find_accessories, search_policy, web_search,
                             add_to_cart, remove_from_cart)}
AGENT_TOOLS = {
    'agent_main': ['search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search'],
    # graph.py có trạng thái hội thoại (giỏ hàng...) nên thêm tool giỏ hàng
    'agent_graph': ['search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search',
                    'add_to_cart', 'remove_from_cart'],
}

def get_agent_tools(agent: str) -> list:
    return [TOOLS[name] for name in AGENT_TOOLS[agent]]
