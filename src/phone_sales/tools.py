import re
import psycopg
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from psycopg.rows import dict_row
from typing import Annotated, Any
from phone_sales import promotions
from phone_sales.conversation import _current_turn, added_to_cart_this_turn, cart_update, cross_sell_step
from phone_sales.config import DATABASE_URL, POLICY_MIN_SCORE, POLICY_SCORE_MARGIN, POLICY_TOP_K
from phone_sales.prompts import POLICY_GUIDE, POLICY_PROMPTS
from phone_sales.textnorm import strip_accents
from phone_sales.vector_db import get_vector_store
from tavily import TavilyClient
from functools import lru_cache

tavily_client = TavilyClient()

MAX_LINES = 15  # số sản phẩm tối đa trả về cho model, tránh kết quả quá dài
# Phụ kiện mỗi trang: prompt dặn tối đa 5 dòng nhưng tool trả 8-15 dòng thì model vẫn liệt kê hết (eval acc-01)
ACCESSORIES_PER_PAGE = 5
PER_KIND = 2  # tìm nhiều loại phụ kiện một lúc ("sạc, cáp"): mỗi loại lấy ngần này món
NOT_WALL_CHARGER = r'không dây|magsafe|magnet|qi2|ô tô'
THIRD_PARTY_BRANDS = ('beats', 'mophie', 'belkin', 'anker', 'otterbox', 'zagg', 'spigen')
# Máy không có mục "Trong Hộp Sản Phẩm" trong dữ liệu (web Samsung không ghi): theo chính sách chung của hãng,
# Galaxy bán tại Việt Nam chỉ kèm cáp USB-C, không kèm củ sạc. GIẢ ĐỊNH, cần kiểm tra lại với từng model.
IN_BOX_BY_BRAND = {'Samsung': {'cable': True, 'charger': False}}
# Up-sell trong search_products: kèm tối đa 2 máy có giá thấp nhất vượt ngân sách không quá 10%
UPSELL_BUDGET_MARGIN = 0.1
UPSELL_ABOVE_BUDGET = 2

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

NO_SPECS_IN_LIST = ('(Danh sách chỉ có giá, dung lượng, tình trạng hàng, không có thông số. '
                    'Cần nêu chip, camera, pin, màn hình thì gọi get_product_details, không tự nêu.)')
NO_SPECS = ('Thông số: chưa có thông số chi tiết của máy này trong dữ liệu. Không tự nêu chip, camera, pin, màn hình; '
            'khách hỏi thì nói chưa có thông tin chi tiết.')

DB_ERROR = ('Lỗi hệ thống khi tra cứu kho hàng ({}). Không tự đưa ra giá hay tình trạng hàng; '
            'xin lỗi anh/chị và hỏi khách có cần nhân viên cửa hàng hỗ trợ thêm không.')

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
    text = strip_accents(text.lower().replace('+', ' plus '))
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

def _same_storage(label: str, storage: str) -> bool:
    """Phiên bản có khớp dung lượng khách/agent nêu không. "256gb" khớp mọi RAM ("256GB - RAM 8GB", "... RAM 12GB");
    nêu cả RAM ("256GB - RAM 12GB", "12GB/256GB", "256GB 12GB RAM") thì chỉ khớp đúng bản đó."""
    have, want = _storage_key(label or ''), _storage_key(storage)
    if len(want) < 2 or len(have) < 2:
        return have[:1] == want[:1]
    return have[:2] in (want[:2], want[1::-1])   # nhận cả kiểu viết RAM trước ("12GB/256GB")

def _filter_storage(variants: list[dict], storage: str) -> list[dict]:
    # Nêu cả RAM mà không khớp bản nào (vd RAM ghi sai) thì quay về chỉ so dung lượng
    if not storage.strip():
        return variants
    exact = [v for v in variants if _same_storage(v['storage'], storage)]
    if exact:
        return exact
    want = _storage_key(storage)
    same = [v for v in variants if _storage_key(v['storage'] or '')[:1] in (want[:1], want[1:2])]
    if same or len(want) != 1:
        return same
    # Chỉ nêu RAM ("RAM 8GB", "8GB"): khớp phần RAM của nhãn "128GB - RAM 8GB"
    return [v for v in variants if _storage_key(v['storage'] or '')[1:2] == want]

def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'

def _stock(n: int) -> str:
    # Không đưa số lượng cho model: có số là model hay đọc ra cho khách ("còn 9 máy"). add_to_cart vẫn kiểm tra số lượng thật
    if n <= 0:
        return 'hết hàng'
    return 'sắp hết' if n <= 5 else 'còn hàng'

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
    above = []
    if max_price > 0:
        # Up-sell: vài máy chỉ nhỉnh hơn ngân sách một chút, để model nhắc khi hợp nhu cầu hơn
        above = [r for r in rows if max_price < r['min_price'] <= max_price * (1 + UPSELL_BUDGET_MARGIN)]
        rows = [r for r in rows if r['min_price'] <= max_price]
    if not rows:
        return (f"Không có sản phẩm khớp (tên: '{query}', hãng: '{brand}', giá tối đa: "
                f"{_price(max_price) if max_price else 'không giới hạn'}).")

    def line(r: dict) -> str:
        price = _price(r['min_price']) if r['min_price'] == r['max_price'] else \
            f"{_price(r['min_price'])} - {_price(r['max_price'])}"
        storages = ', '.join(sorted((s for s in r['storages'] if s), key=_storage_key))
        return (f"- {r['name']} ({r['brand']}): {price}"
                + (f' | {storages}' if storages else '') + f" | {_stock(r['stock'])}")

    lines = [line(r) for r in rows[:MAX_LINES]]
    more = f'\n(còn {len(rows) - MAX_LINES} sản phẩm khác, hãy hỏi cụ thể hơn)' if len(rows) > MAX_LINES else ''
    if above:
        more += ('\nNhỉnh hơn ngân sách (chỉ nhắc tối đa 1 máy, khi hợp nhu cầu hơn rõ rệt):\n'
                 + '\n'.join(f"{line(r)} | vượt {_price(r['min_price'] - max_price)}"
                             for r in above[:UPSELL_ABOVE_BUDGET]))
    # "muốn mua Galaxy S26": model hay coi là cả dòng máy, chỉ đưa khoảng giá thay vì các phiên bản của đúng máy đó
    exact = [r for r in rows if _tokens(r['name']) == _tokens(query)] if query.strip() else []
    if len(exact) == 1:
        more += (f"\nKhách nêu đúng tên máy {exact[0]['name']}: nếu khách muốn mua/xem máy này thì gọi "
                 f"get_product_details('{exact[0]['name']}') và liệt kê các phiên bản, không chỉ đưa khoảng giá.")
    # Không có dòng này thì model tự thêm thông số theo hiểu biết riêng (vd "S26 FE pin 4.500mAh", dữ liệu là 4900)
    return ('Sản phẩm đang bán (giá đã gồm VAT):\n' + '\n'.join(lines) + more
            + f'\n{NO_SPECS_IN_LIST}' * (category == 'phone'))

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
    bigger = []
    if storage.strip():
        # "256gb" khớp cả "256GB - RAM 8GB" và "256GB - RAM 12GB"; nêu cả RAM thì chỉ bản đó
        want = _storage_key(storage)[:1]
        bigger = [v for v in variants if _storage_key(v['storage'] or '')[:1] > want]
        variants = _filter_storage(variants, storage)
    if color.strip():
        variants = [v for v in variants if _norm_text(color) in _norm_text(v['color'] or '')]
    estimated = set(attrs.get('estimated_price_skus', []))
    lines = [f"{p['name']} ({p['brand']})", p['description'] or '']
    if attrs.get('ram_by_storage'):
        lines.append('RAM: ' + ', '.join(f'{s}: {r}' for s, r in attrs['ram_by_storage'].items()))
    specs = attrs.get('specs') or {}
    spec_lines = [f'{k}: {specs[k][:220]}' for k in SPEC_KEYS if specs.get(k)]
    lines += spec_lines or [NO_SPECS]  # dòng Galaxy S/Z chưa có thông số (web Samsung tải bằng JavaScript)
    box = _in_box(p)
    if box:
        lines.append(f"Trong hộp: {box['text']}")
    lines.append('Phiên bản (dung lượng | màu | giá | kho):' if variants else
                 f"Không có phiên bản khớp (dung lượng: '{storage}', màu: '{color}').")
    lines += [f"- {v['storage']} | {v['color']} | {_price(v['price'])}"
              f"{' (giá tham khảo)' if v['variant_id'] in estimated else ''} | {_stock(v['stock'])}"
              for v in variants]
    lines.append(_bigger_storages(variants, bigger))
    return '\n'.join(line for line in lines if line)

def _bigger_storages(chosen: list[dict], bigger: list[dict]) -> str:
    # Up-sell: giá từng dung lượng lớn hơn bản khách chọn, kèm số tiền chênh tính sẵn (model hay cộng trừ sai)
    base = min((v['price'] for v in chosen), default=None)
    in_stock = [v for v in bigger if v['stock'] > 0]
    if base is None or not in_stock:
        return ''
    cheapest: dict[str, int] = {}
    for v in in_stock:
        key = v['storage'] or ''
        cheapest[key] = min(cheapest.get(key, v['price']), v['price'])
    parts = [f'{s} từ {_price(price)} (thêm {_price(price - base)})'
             for s, price in sorted(cheapest.items(), key=lambda kv: _storage_key(kv[0]))]
    return 'Dung lượng lớn hơn còn hàng: ' + ', '.join(parts)

@tool
def find_accessories(phone: str, accessory_type: str = '', page: int = 1, genuine_only: bool = False) -> str:
    """Tìm phụ kiện tương thích với 1 điện thoại: ốp lưng, miếng dán màn hình, củ sạc, cáp, sạc dự phòng,
    S Pen, ví, giá đỡ. phone: tên máy (vd 'iPhone 17', 'Galaxy S26 Ultra').
    accessory_type: loại phụ kiện (vd 'ốp lưng', 'sạc', 'củ sạc', 'cáp', 'cường lực'), nhiều loại cách nhau dấu phẩy
    (vd 'sạc, cáp' hoặc 'ốp lưng, cường lực': mỗi loại 2 món), để trống để xem tất cả loại.
    'củ sạc' chỉ lấy củ sạc cắm tường (bỏ sạc không dây, MagSafe, ô tô; hộp máy đã có cáp thì bỏ loại kèm cáp).
    Mỗi lần trả 5 phụ kiện (còn hàng trước, chính hãng trước, rẻ trước); tìm sạc/cáp thì kèm máy có sẵn gì trong hộp.
    Khách muốn xem thêm thì gọi lại với page=2, 3...
    genuine_only=True: loại nào có hàng chính hãng (cùng hãng với máy) thì chỉ lấy hàng chính hãng, dùng khi mời bán kèm."""
    # Nhiều loại cách nhau dấu phẩy ("sạc, cáp", "ốp lưng, cường lực"): mỗi loại lấy vài món, không bị 1 loại chiếm hết
    asked = [s.strip() for s in re.split(r',|\+|\bvà\b', accessory_type) if s.strip()]
    kinds = list(dict.fromkeys(_accessory_type(s) for s in asked))
    if None in kinds:
        return f"Không rõ loại phụ kiện '{accessory_type}'. Các loại có: ốp lưng, miếng dán, sạc, cáp, sạc dự phòng, S Pen, ví, giá đỡ."
    try:
        phones = _query("SELECT product_id, name, brand, attributes FROM products WHERE category = 'phone'")
        hits = _match(phones, phone)
        if len(hits) != 1:
            names = ', '.join(p['name'] for p in hits[:MAX_LINES])
            return (f"Có nhiều máy khớp '{phone}': {names}. Hỏi lại khách dùng máy nào." if hits else
                    f"Không tìm thấy điện thoại '{phone}' trong danh sách máy cửa hàng đang bán.")
        rows = _query(
            """SELECT a.name, a.brand, a.attributes->>'accessory_type' AS kind, v.price, v.stock
               FROM compatibility c JOIN products a ON a.product_id = c.accessory_id
               JOIN variants v ON v.product_id = a.product_id
               WHERE c.phone_id = %s AND (%s::text[] IS NULL OR a.attributes->>'accessory_type' = ANY(%s))
               ORDER BY v.stock = 0, v.price""",
            (hits[0]['product_id'], kinds or None, kinds or None))
    except Exception as e:
        return DB_ERROR.format(type(e).__name__)
    p = hits[0]
    box = _in_box(p)
    if any(re.search(r'\bcu sac\b', _norm_text(s)) for s in asked):
        # Khách/bước bán kèm cần củ sạc cắm tường: sạc không dây vẫn phải có củ sạc mới dùng được.
        # Hộp máy đã có cáp thì bỏ bộ củ sạc kèm cáp, chỉ lấy củ sạc
        skip = NOT_WALL_CHARGER + (r'|(?<!không )kèm cáp' if box and box['cable'] else '')
        rows = [r for r in rows if not (r['kind'] == 'charger' and re.search(skip, r['name'].lower()))]
    if genuine_only:
        # Bán kèm ưu tiên hàng chính hãng: lọc ở đây, prompt dặn thì model vẫn hay lấy hàng hãng khác cho đủ số món
        # (vd củ sạc Apple, Belkin cho Galaxy). Lọc theo từng loại: loại không có hàng chính hãng thì giữ hàng hãng
        # khác, vì Apple không bán miếng dán màn hình, cường lực cho iPhone chỉ có Belkin
        has_genuine = {r['kind'] for r in rows if _genuine(r, p['brand'])}
        rows = [r for r in rows if r['kind'] not in has_genuine or _genuine(r, p['brand'])]
    header = f"Phụ kiện tương thích với {p['name']}:"
    if box and {'charger', 'cable'} & set(kinds):
        header = f"Trong hộp {p['name']}: {box['text']}.\n" + header
    if not rows:
        return (f"Chưa có phụ kiện {accessory_type or ''} tương thích với {p['name']}."
                .replace('  ', ' '))
    # Còn hàng trước, chính hãng trước, củ sạc cắm tường trước sạc không dây/ô tô (máy thiếu là thiếu củ sạc), rẻ trước
    need = _charging(p).get('wired') if 'charger' in kinds or not kinds else None
    rows.sort(key=lambda r: (r['stock'] <= 0, not _genuine(r, p['brand']),
                             r['kind'] == 'charger' and bool(re.search(NOT_WALL_CHARGER, r['name'].lower())),
                             bool(need) and r['kind'] == 'charger' and 'dưới' in _charger_fit(r, {'wired': need}),
                             r['price']))
    start = max(page, 1) - 1
    if len(kinds) > 1:
        by_kind = [[r for r in rows if r['kind'] == k] for k in kinds]
        shown = [r for group in by_kind for r in group[start * PER_KIND:(start + 1) * PER_KIND]]
        left = sum(max(len(group) - (start + 1) * PER_KIND, 0) for group in by_kind)
    else:
        shown = rows[start * ACCESSORIES_PER_PAGE:(start + 1) * ACCESSORIES_PER_PAGE]
        left = len(rows) - start * ACCESSORIES_PER_PAGE - len(shown)
    if not shown:
        return f"Đã xem hết {len(rows)} phụ kiện tương thích với {p['name']}."
    charging = _charging(p) if any(r['kind'] == 'charger' for r in shown) else {}
    lines = [f"- {r['name']}: {_price(r['price'])}" + (f" | chính hãng {p['brand']}" if _genuine(r, p['brand']) else '')
             + _charger_fit(r, charging) + f" | {_stock(r['stock'])}" for r in shown]
    # 1 loại: hàng chính hãng xếp trước, tách rõ phần hãng khác để model không trộn vào lời mời hàng chính hãng
    genuine = [_genuine(r, p['brand']) for r in shown]
    if len(kinds) <= 1 and genuine[0] and not all(genuine):
        lines.insert(genuine.index(False), f"Hãng khác (không phải chính hãng {p['brand']}):")
    more = f'\n(còn {left} phụ kiện khác, xem tiếp bằng page={page + 1})' if left > 0 else ''
    if charging.get('text'):
        header = f"{charging['text']}\n" + header
    return header + '\n' + '\n'.join(lines) + more

def _charging(phone: dict) -> dict:
    """Công suất sạc theo thông số máy (chỉ dòng có thông số, hiện là iPhone):
    wired: bộ tiếp hợp tối thiểu để sạc nhanh qua cáp USB-C; wireless: sạc không dây MagSafe tối đa."""
    specs = (phone.get('attributes') or {}).get('specs') or {}
    text = ' '.join(specs.values())
    wired = re.search(r'Sạc lên đến (\d+%) trong (?:khoảng )?(\d+ phút) với bộ tiếp hợp (\d+)\s*W trở lên'
                      r'[^.]{0,30}?sử dụng cáp sạc USB', text)
    wireless = re.search(r'Sạc không dây MagSafe lên đến (\d+)\s*W', text)
    out: dict = {}
    parts = []
    if wired:
        out['wired'] = int(wired.group(3))
        parts.append(f"sạc nhanh {wired.group(1)} trong {wired.group(2)} cần bộ tiếp hợp từ {out['wired']}W trở lên "
                     "(qua cáp USB-C)")
    if wireless:
        out['wireless'] = int(wireless.group(1))
        parts.append(f"sạc không dây MagSafe tối đa {out['wireless']}W")
    if parts:
        out['text'] = (f"Công suất sạc của {phone['name']} (theo thông số): " + '; '.join(parts) + '.'
                       + (' Nói rõ củ sạc nào phù hợp theo ghi chú từng dòng.' if wired else ''))
    return out

def _charger_fit(acc: dict, charging: dict) -> str:
    # Ghi chú củ sạc có đủ công suất sạc nhanh cho máy không (máy không có thông số sạc thì không ghi)
    if acc['kind'] != 'charger' or not charging:
        return ''
    name = acc['name'].lower()
    if re.search(NOT_WALL_CHARGER, name):
        return f" | sạc không dây tối đa {charging['wireless']}W trên máy này" \
            if charging.get('wireless') and re.search(r'magsafe|qi2|không dây', name) else ''
    watts = [int(w) for w in re.findall(r'(\d+)\s*w\b', name)]
    need = charging.get('wired')
    if not watts or not need:
        return ''
    if max(watts) >= need:
        return f' | đủ công suất sạc nhanh cho máy (từ {need}W)'
    return f' | dưới {need}W: sạc được nhưng không đạt sạc nhanh'

@tool
def find_cheaper_alternatives(product: str, state: Annotated[dict, InjectedState],
                              tool_call_id: Annotated[str, InjectedToolCallId], min_drop_pct: float = 0.15,
                              max_drop_pct: float = 0.35, storage: str = '') -> Command:
    """Tìm điện thoại khác rẻ hơn máy khách đang chê đắt/phân vân, trong khoảng % cho trước, còn hàng.
    product: máy khách đang xem (vd 'iPhone 17'); storage: dung lượng khách đang xem (để trống = bản rẻ nhất).
    min_drop_pct, max_drop_pct: rẻ hơn ít nhất / nhiều nhất bao nhiêu phần trăm (vd 0.15 và 0.35).
    Khách đã nói ngân sách thì không vượt ngân sách; máy đã gợi ý ở lượt trước không lặp lại."""
    def reply(text: str, offered: list[str] | None = None) -> Command:
        update = {'messages': [ToolMessage(text, tool_call_id=tool_call_id)]}
        if offered:
            update['offered_alternatives'] = offered
        return Command(update=update)

    low_pct, high_pct = sorted((max(min_drop_pct, 0.0), min(max_drop_pct, 0.9)))
    try:
        phones = _query("""SELECT p.product_id, p.name, p.brand, v.storage, v.color, v.price, v.stock
                           FROM products p JOIN variants v USING (product_id) WHERE p.category = 'phone'""")
    except Exception as e:
        return reply(DB_ERROR.format(type(e).__name__))
    products = list({r['product_id']: {'product_id': r['product_id'], 'name': r['name']} for r in phones}.values())
    hits = _match(products, product)
    if len(hits) != 1:
        names = ', '.join(p['name'] for p in hits[:MAX_LINES])
        return reply(f"Có nhiều máy khớp '{product}': {names}. Hỏi lại khách đang xem máy nào." if hits else
                     f"Không tìm thấy điện thoại '{product}' (tool chỉ áp dụng cho điện thoại).")
    base_id = hits[0]['product_id']
    own = [r for r in phones if r['product_id'] == base_id]
    chosen = _filter_storage(own, storage) or own
    base = min(chosen, key=lambda r: r['price'])
    budget = (state.get('slots') or {}).get('budget_max')
    low, high = base['price'] * (1 - high_pct), base['price'] * (1 - low_pct)
    if budget:
        high = min(high, budget)
    offered = set(state.get('offered_alternatives') or [])
    # Mỗi máy lấy 1 phiên bản còn hàng trong khoảng giá: dung lượng lớn nhất (ít đánh đổi nhất), rồi rẻ hơn
    best: dict[str, dict] = {}
    for r in phones:
        if (r['product_id'] == base_id or r['stock'] <= 0 or not low <= r['price'] <= high
                or r['name'] in offered):
            continue
        cur = best.get(r['product_id'])
        if cur is None or (_storage_key(r['storage'] or ''), -r['price']) > (_storage_key(cur['storage'] or ''), -cur['price']):
            best[r['product_id']] = r
    # Cùng hãng trước, rồi giá gần máy gốc nhất (ít tụt phân khúc)
    rows = sorted(best.values(), key=lambda r: (r['brand'] != base['brand'], -r['price']))[:3]
    base_text = f"{base['name']} {base['storage'] or ''} {_price(base['price'])}".replace('  ', ' ')
    # Dung lượng thấp hơn của chính máy đó: lựa chọn ít đánh đổi nhất
    smaller = sorted({r['storage']: r for r in own if r['stock'] > 0 and r['price'] < base['price']}.values(),
                     key=lambda r: -r['price'])
    lines = [f"Máy gốc: {base_text}. Khoảng tìm: {_price(int(low))} - {_price(int(high))}"
             f" (rẻ hơn {low_pct:.0%}-{high_pct:.0%}{', không vượt ngân sách ' + _price(budget) if budget else ''})."]
    lines += [f"- {r['name']} {r['storage'] or ''}: {_price(r['price'])} | rẻ hơn {_price(base['price'] - r['price'])}"
              f" | {_stock(r['stock'])}" for r in rows]
    if smaller:
        lines.append('Dung lượng thấp hơn của chính máy này: ' + ', '.join(
            f"{r['storage']} {_price(r['price'])} (rẻ hơn {_price(base['price'] - r['price'])})" for r in smaller))
    if not rows:
        lines.append('Không có máy khác còn hàng trong khoảng giá này'
                     + (' (đã bỏ máy gợi ý ở lượt trước)' if offered else '')
                     + '. Nói thật với khách, không gợi ý máy quá rẻ, không tự giảm giá.')
    else:
        lines.append('Thông số các máy này chưa có ở đây: muốn nêu đánh đổi về camera, pin, chip thì gọi '
                     'get_product_details, không tự đoán; đánh đổi chắc chắn là giá, dung lượng, dòng máy.')
    return reply('\n'.join(lines), [r['name'] for r in rows])

@tool
def get_promotions(product: str = '') -> str:
    """Khuyến mãi đang áp dụng tại cửa hàng (mã giảm giá, khuyến mãi tự động): mức giảm, phạm vi, điều kiện, hạn.
    product: tên sản phẩm khách đang xem để chỉ lấy khuyến mãi áp dụng cho sản phẩm đó (để trống = tất cả).
    Dùng khi khách hỏi khuyến mãi, ưu đãi, mã giảm giá, hoặc khi đang do dự. Chỉ nêu khuyến mãi có trong kết quả."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row) as conn:
            promos = promotions.active(conn)
        item = None
        if product.strip():
            hits = _match(_query('SELECT product_id, name, brand, category FROM products'), product)
            if len(hits) != 1:
                names = ', '.join(p['name'] for p in hits[:MAX_LINES])
                return (f"Có nhiều sản phẩm khớp '{product}': {names}. Hỏi lại khách." if hits else
                        f"Không tìm thấy sản phẩm '{product}'.")
            item = hits[0]
    except Exception as e:
        return DB_ERROR.format(type(e).__name__)
    if item:
        promos = [p for p in promos if promotions.applies_to(p, item)]
    target = f" cho {item['name']}" if item else ''
    if not promos:
        return (f'Hiện không có chương trình khuyến mãi nào đang áp dụng{target}. Không nêu khuyến mãi, giảm giá, '
                'quà tặng nào khác.')
    return (f'Khuyến mãi đang áp dụng{target} (khách nhập mã ở form đặt hàng; hệ thống kiểm tra điều kiện và số lần '
            'dùng khi đặt):\n' + '\n'.join('- ' + promotions.describe(p) for p in promos))

def _genuine(acc: dict, phone_brand: str) -> bool:
    # Chính hãng = cùng hãng với máy; cửa hàng Apple bán cả Beats, mophie, Belkin, Anker nhưng đó là hãng khác
    name = acc['name'].lower()
    return acc.get('brand') == phone_brand and not any(b in name for b in THIRD_PARTY_BRANDS)

def _in_box(phone: dict) -> dict | None:
    """Máy có kèm củ sạc, cáp không: {'charger': bool, 'cable': bool, 'text': ...}; None nếu không rõ."""
    box = ((phone.get('attributes') or {}).get('specs') or {}).get('Trong Hộp Sản Phẩm')
    if box:
        # Chỉ đọc phần đầu (danh sách món trong hộp); phía sau là đoạn giới thiệu môi trường của Apple
        head = box.lower()[:120]
        found = {'cable': 'cáp' in head, 'charger': bool(re.search(r'củ sạc|bộ sạc|bộ tiếp hợp|adapter', head))}
    else:
        found = IN_BOX_BY_BRAND.get(phone.get('brand'))
    if found is None:
        return None
    text = ', '.join(['có cáp sạc' if found['cable'] else 'không có cáp sạc',
                      'có củ sạc' if found['charger'] else 'không kèm củ sạc'])
    return {**found, 'text': text}

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
                'Không tự đưa ra chính sách; xin lỗi anh/chị và hỏi khách có cần nhân viên cửa hàng hỗ trợ thêm không.')
    hits = _relevant_policy_hits(scored)
    if not hits:
        return ('Không tìm thấy chính sách liên quan tới câu hỏi trong dữ liệu nội bộ. Không tự trả lời về chính sách; '
                'nói chưa có thông tin và hỏi khách có cần nhân viên cửa hàng hỗ trợ thêm không.')
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

def _cart_reply(tool_call_id: str, text: str, change: dict | None = None) -> Command:
    # change: thao tác trên giỏ ({'add': món} / {'remove': [variant_id]}), không phải cả giỏ: 2 tool giỏ hàng chạy
    # song song trong 1 bước thì conversation.cart_update áp lần lượt cả hai
    update = {'messages': [ToolMessage(text, tool_call_id=tool_call_id)]}
    if change is not None:
        update['cart'] = change
    return Command(update=update)

def _cart_extra(p: dict) -> dict:
    # Cho bước bán kèm (conversation.cross_sell_hint): máy có kèm củ sạc không, phụ kiện thuộc loại nào
    if p.get('category') == 'accessory':
        return {'accessory_type': (p.get('attributes') or {}).get('accessory_type')}
    box = _in_box(p)
    return {'charger_in_box': box['charger'] if box else None, 'cable_in_box': box['cable'] if box else None}

def _context_phones(state: dict, for_phone: str) -> list[str]:
    """Điện thoại khách đang mua: for_phone model truyền, máy trong giỏ, máy add_to_cart cùng lượt (gọi song song
    thì giỏ chưa kịp có máy). Dùng để chọn phụ kiện đúng máy khi tên phụ kiện khớp nhiều sản phẩm."""
    names = [for_phone] if for_phone.strip() else []
    names += [i['name'] for i in state.get('cart') or [] if i.get('category') == 'phone']
    names += [tc['args'].get('product', '') for m in _current_turn(state.get('messages') or [])
              if isinstance(m, AIMessage) for tc in (m.tool_calls or []) if tc['name'] == 'add_to_cart']
    return [n for n in dict.fromkeys(names) if n.strip()]

def _narrow_to_phone(hits: list[dict], phones: list[str]) -> list[dict]:
    """Tên phụ kiện khớp nhiều sản phẩm ('Miếng Dán Belkin UltraGlass 2' có bản cho iPhone 17, 17 Pro, 17 Pro Max):
    giữ món tương thích với máy khách đang mua. Không lọc được (không rõ máy, không món nào hợp) thì giữ nguyên."""
    if len(hits) < 2 or not phones or any(h['category'] == 'phone' for h in hits):
        return hits
    catalog = _query("SELECT product_id, name FROM products WHERE category = 'phone'")
    phone_ids = {p['product_id'] for n in phones for p in _match(catalog, n) if len(_match(catalog, n)) == 1}
    if not phone_ids:
        return hits
    fits = {r['accessory_id'] for r in _query(
        'SELECT accessory_id FROM compatibility WHERE phone_id = ANY(%s) AND accessory_id = ANY(%s)',
        (list(phone_ids), [h['product_id'] for h in hits]))}
    return [h for h in hits if h['product_id'] in fits] or hits

@tool
def add_to_cart(product: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId],
                storage: str = '', color: str = '', quantity: int = 1, for_phone: str = '') -> Command:
    """Thêm 1 sản phẩm (điện thoại hoặc phụ kiện) vào giỏ hàng khi khách đồng ý mua.
    product: tên sản phẩm (vd 'iPhone 17', 'Ốp Lưng Trong Suốt Galaxy S26 Ultra').
    storage, color: phiên bản điện thoại khách chọn (bắt buộc nếu máy có nhiều phiên bản). quantity: số lượng.
    for_phone: với phụ kiện, tên điện thoại khách dùng (vd 'iPhone 17') để chọn đúng bản cho máy đó.
    Chỉ gọi khi khách đã nói muốn mua/lấy/chốt; không tự thêm khi khách mới hỏi giá."""
    if quantity < 1:
        return _cart_reply(tool_call_id, 'Số lượng phải từ 1 trở lên.')
    try:
        products = _query('SELECT product_id, name, brand, category, attributes FROM products')
        hits = _narrow_to_phone(_match(products, product), _context_phones(state, for_phone))
        if len(hits) != 1:
            names = ', '.join(p['name'] for p in hits[:MAX_LINES])
            return _cart_reply(tool_call_id, f"Có nhiều sản phẩm khớp '{product}': {names}. Hỏi lại khách." if hits
                               else f"Không tìm thấy sản phẩm '{product}'.")
        p = hits[0]
        variants = _query('SELECT variant_id, color, storage, price, stock FROM variants WHERE product_id = %s',
                          (p['product_id'],))
    except Exception as e:
        return _cart_reply(tool_call_id, DB_ERROR.format(type(e).__name__))
    variants = _filter_storage(variants, storage)
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
    added = {'variant_id': v['variant_id'], 'name': p['name'], 'category': p['category'],
             'storage': v['storage'], 'color': v['color'], 'price': v['price'], 'quantity': quantity, **_cart_extra(p)}
    cart = cart_update(cart, {'add': added})
    total = sum(i['price'] * i['quantity'] for i in cart)
    return _cart_reply(tool_call_id, f"Đã thêm {quantity} x {p['name']} {v['storage'] or ''} {v['color'] or ''} "
                       f"({_price(v['price'])}). Giỏ có {len(cart)} sản phẩm, tổng {_price(total)}.", {'add': added})

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
    removed = [i for i in cart if i not in keep]
    reply = _cart_reply(tool_call_id, f'Đã bỏ {len(cart) - len(keep)} sản phẩm. Giỏ còn {len(keep)} sản phẩm, '
                        f'tổng {_price(total)}.', {'remove': [i['variant_id'] for i in removed]})
    # Khách bỏ phụ kiện đã yêu cầu: quên yêu cầu đó, không để wanted_hint tự thêm lại
    reply.update['wanted'] = {'drop': [i['accessory_type'] for i in removed if i.get('accessory_type')]}
    return reply

@tool
def show_order_form(state: Annotated[dict, InjectedState],
                    tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Hiện form đặt hàng (họ tên, số điện thoại, địa chỉ, cách thanh toán) ngay dưới câu trả lời.
    Gọi khi khách xác nhận đặt hàng/chốt đơn với giỏ hàng hiện tại. Không hỏi thông tin nhận hàng bằng chữ."""
    cart = state.get('cart') or []
    if not cart:
        return _cart_reply(tool_call_id, 'Giỏ hàng đang trống, chưa mở form đặt hàng. Hỏi khách muốn mua sản phẩm nào.')
    if added_to_cart_this_turn(state.get('messages') or []):
        # "Chốt S26 256GB đen" là chọn máy, chưa phải xác nhận đơn: bán kèm, hỏi chốt đơn trước, form để lượt sau
        return _cart_reply(tool_call_id, 'Chưa mở form: khách vừa thêm hàng vào giỏ ở lượt này. Làm theo mục "Bán kèm" '
                           'trong trạng thái (nếu có) rồi hỏi khách có xác nhận đặt hàng không; khách xác nhận ở '
                           'lượt sau thì mới gọi show_order_form.')
    if cross_sell_step(state) in ('charger', 'accessories'):
        # "Không cần sạc, xác nhận đặt hàng" cùng 1 tin: model hay mở form luôn, bỏ qua bước mời ốp lưng, cường lực
        return _cart_reply(tool_call_id, 'Chưa mở form: chưa xong bước bán kèm. Làm theo mục "Bán kèm" trong trạng '
                           'thái hội thoại (mời ốp lưng, cường lực), rồi hỏi khách lấy thêm hay xác nhận đặt hàng.')
    total = sum(i['price'] * i['quantity'] for i in cart)
    # Giao diện đọc order_form trong state để hiện form; khách gửi form thì api.py (POST /order) tạo đơn
    return Command(update={'order_form': True, 'messages': [ToolMessage(
        f'Đã hiện form đặt hàng cho {len(cart)} sản phẩm, tổng {_price(total)}. Trả lời 1 câu ngắn mời khách điền '
        'form bên dưới; không hỏi lại tên, số điện thoại, địa chỉ hay cách thanh toán.', tool_call_id=tool_call_id)]})

# Tool của từng agent. Mô tả tool mà model nhìn thấy lấy từ docstring của hàm @tool ở trên.
TOOLS = {t.name: t for t in (search_products, get_product_details, find_accessories, search_policy, web_search,
                             add_to_cart, remove_from_cart, show_order_form, find_cheaper_alternatives,
                             get_promotions)}
AGENT_TOOLS = {
    'agent_main': ['search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search',
                   'get_promotions'],
    # graph.py có trạng thái hội thoại (giỏ hàng...) nên thêm tool giỏ hàng và form đặt hàng
    'agent_graph': ['search_products', 'get_product_details', 'find_accessories', 'search_policy', 'web_search',
                    'get_promotions', 'add_to_cart', 'remove_from_cart', 'show_order_form',
                    'find_cheaper_alternatives'],
}

def get_agent_tools(agent: str) -> list:
    return [TOOLS[name] for name in AGENT_TOOLS[agent]]
