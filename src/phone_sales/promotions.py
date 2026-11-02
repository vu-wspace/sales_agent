# Khuyến mãi (bảng promotions, promotion_usages):
#   - khuyến mãi đang chạy cho agent tư vấn (tools.get_promotions)
#   - kiểm tra mã, tính giảm giá khi khách đặt hàng (POST /order/quote, orders.create_order)
# Phạm vi (scope / scope_value): 'all' cả đơn; 'brand' theo hãng ('Apple', 'Samsung'); 'product' theo product_id;
# 'category' theo loại ('phone', 'accessory'). Giảm trên phần giỏ thuộc phạm vi; min_order so với tạm tính cả đơn.
# Khuyến mãi không có code là tự động: khách không nhập mã thì áp mã tự động giảm nhiều nhất.
from datetime import datetime, timezone

DISCOUNT_TYPES = ('percent', 'fixed')
SCOPE_LABELS = {'brand': 'sản phẩm {}', 'product': 'sản phẩm {}', 'category': '{}'}
CATEGORY_LABELS = {'phone': 'điện thoại', 'accessory': 'phụ kiện'}


class PromoError(Exception):
    """Mã khuyến mãi không dùng được (hết hạn, chưa đủ điều kiện...): api.py trả nguyên văn cho giao diện."""


def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'


def _now() -> datetime:
    return datetime.now(timezone.utc)


def applies_to(promo: dict, item: dict) -> bool:
    scope, value = promo['scope'], (promo.get('scope_value') or '').strip().lower()
    if scope == 'all':
        return True
    if scope == 'brand':
        return (item.get('brand') or '').lower() == value
    if scope == 'product':
        return (item.get('product_id') or '').lower() == value
    if scope == 'category':
        return (item.get('category') or '').lower() == value
    return False  # scope lạ: không áp dụng thay vì giảm nhầm cả đơn


def check(promo: dict, subtotal: int, now: datetime | None = None, used_by_customer: int = 0) -> None:
    """Mã còn dùng được không: đang bật, trong thời hạn, còn lượt, khách chưa dùng quá số lần, đủ đơn tối thiểu."""
    now = now or _now()
    if not promo['is_active'] or now < promo['starts_at']:
        raise PromoError('Mã khuyến mãi này chưa được áp dụng.')
    if now > promo['ends_at']:
        raise PromoError('Mã khuyến mãi này đã hết hạn.')
    if promo.get('usage_limit') is not None and promo['used_count'] >= promo['usage_limit']:
        raise PromoError('Mã khuyến mãi này đã hết lượt sử dụng.')
    if promo.get('per_customer_limit') is not None and used_by_customer >= promo['per_customer_limit']:
        raise PromoError('Anh/chị đã dùng hết số lần cho mã khuyến mãi này.')
    if subtotal < (promo.get('min_order') or 0):
        raise PromoError(f"Mã khuyến mãi này áp dụng cho đơn từ {_price(promo['min_order'])}.")


def discount_for(promo: dict, items: list[dict]) -> int:
    """Số tiền giảm trên phần giỏ thuộc phạm vi của mã (items: price, quantity, brand, product_id, category)."""
    eligible = sum(i['price'] * i['quantity'] for i in items if applies_to(promo, i))
    if not eligible:
        raise PromoError('Mã khuyến mãi này không áp dụng cho sản phẩm trong giỏ hàng.')
    if promo['discount_type'] == 'percent':
        amount = eligible * promo['value'] // 100
        if promo.get('max_discount'):
            amount = min(amount, promo['max_discount'])
    else:
        amount = promo['value']
    return max(0, min(amount, eligible))


def describe(promo: dict) -> str:
    """1 dòng cho agent / giao diện: mức giảm, phạm vi, điều kiện, hạn (chỉ từ dữ liệu bảng promotions)."""
    if promo['discount_type'] == 'percent':
        amount = f"giảm {promo['value']}%" + (f" (tối đa {_price(promo['max_discount'])})" if promo.get('max_discount') else '')
    else:
        amount = f"giảm {_price(promo['value'])}"
    scope = promo['scope']
    value = promo.get('scope_value') or ''
    target = SCOPE_LABELS[scope].format(CATEGORY_LABELS.get(value, value)) if scope in SCOPE_LABELS else 'cả đơn hàng'
    parts = [f"{promo['description']}: {amount} cho {target}"]
    if promo.get('min_order'):
        parts.append(f"đơn từ {_price(promo['min_order'])}")
    parts.append(f"đến hết {promo['ends_at'].astimezone().strftime('%d/%m/%Y')}")
    code = f"mã {promo['code']}" if promo.get('code') else 'tự động áp dụng, không cần mã'
    return f"{', '.join(parts)} ({code})"


def label(promo: dict | None) -> str | None:
    if not promo:
        return None
    return promo['code'] or promo['description']


# ---------------------------------------------------------------- database (conn: psycopg, row_factory=dict_row)

ACTIVE = 'is_active AND starts_at <= now() AND ends_at >= now() AND (usage_limit IS NULL OR used_count < usage_limit)'


def active(conn) -> list[dict]:
    return conn.execute(f'SELECT * FROM promotions WHERE {ACTIVE} ORDER BY ends_at, promo_id').fetchall()


def cart_items(conn, cart: list[dict]) -> list[dict]:
    # Hãng, product_id, loại lấy từ database theo variant_id (giỏ hàng trong state không có đủ)
    rows = conn.execute('SELECT v.variant_id, p.product_id, p.brand, p.category FROM variants v '
                        'JOIN products p ON p.product_id = v.product_id WHERE v.variant_id = ANY(%s)',
                        ([i['variant_id'] for i in cart],)).fetchall()
    info = {r['variant_id']: r for r in rows}
    return [{**i, **{k: v for k, v in info.get(i['variant_id'], {}).items() if k != 'variant_id'}} for i in cart]


def customer_usage(conn, promo_id: str, customer_id: str | None) -> int:
    if not customer_id:
        return 0
    return conn.execute('SELECT count(*) AS n FROM promotion_usages WHERE promo_id = %s AND customer_id = %s',
                        (promo_id, customer_id)).fetchone()['n']


def quote(conn, cart: list[dict], code: str | None = None, customer_id: str | None = None,
          lock: bool = False) -> dict:
    """Tạm tính, giảm giá, tổng tiền cho giỏ hàng. Có mã: mã sai/không đủ điều kiện thì PromoError.
    Không mã: áp khuyến mãi tự động (không có code) giảm nhiều nhất, không có thì không giảm.
    lock=True (lúc đặt hàng): khóa dòng promotions để 2 đơn cùng lúc không vượt usage_limit."""
    items = cart_items(conn, cart)
    subtotal = sum(i['price'] * i['quantity'] for i in items)
    candidates = []
    if code:
        promo = conn.execute('SELECT * FROM promotions WHERE upper(code) = upper(%s)' + ' FOR UPDATE' * lock,
                             (code.strip(),)).fetchone()
        if not promo:
            raise PromoError(f'Mã khuyến mãi {code.strip().upper()} không tồn tại.')
        candidates = [promo]
    else:
        candidates = conn.execute(f'SELECT * FROM promotions WHERE code IS NULL AND {ACTIVE}' + ' FOR UPDATE' * lock
                                  ).fetchall()
    best, best_amount, error = None, 0, None
    for promo in candidates:
        try:
            check(promo, subtotal, used_by_customer=customer_usage(conn, promo['promo_id'], customer_id))
            amount = discount_for(promo, items)
        except PromoError as e:
            error = e
            continue
        if amount > best_amount:
            best, best_amount = promo, amount
    if code and best is None:
        raise error
    return {'subtotal': subtotal, 'discount': best_amount, 'total': subtotal - best_amount, 'promo': best}


def record_usage(conn, promo: dict, order_id: str, customer_id: str, amount: int) -> None:
    conn.execute('INSERT INTO promotion_usages (promo_id, order_id, customer_id, discount_amount) VALUES (%s, %s, %s, %s)',
                 (promo['promo_id'], order_id, customer_id, amount))
    conn.execute('UPDATE promotions SET used_count = used_count + 1 WHERE promo_id = %s', (promo['promo_id'],))
