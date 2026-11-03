# Đơn đặt hàng từ form trên khung chat: lưu khách hàng + đơn vào Postgres (bảng customers, orders, order_items),
# tóm tắt đơn bằng template lấy thẳng từ database, chỉ dùng LLM cho 1-2 câu cảm ơn cuối.
import logging
import uuid
from textwrap import dedent

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage
from psycopg.rows import dict_row

from phone_sales import llm, pii, promotions
from phone_sales.config import DATABASE_URL, ROOT
from phone_sales.fallback import FallbackMiddleware

log = logging.getLogger(__name__)
SCHEMA = ROOT / 'data_schema' / 'orders.sql'

FULFILLMENT = {'delivery': 'Giao tận nơi', 'pickup': 'Nhận tại cửa hàng'}
PAYMENT = {'cod': 'Thanh toán khi nhận hàng', 'bank_transfer': 'Chuyển khoản', 'installment': 'Trả góp'}
CALLBACK_NOTE = 'Nhân viên cửa hàng sẽ gọi điện xác nhận đơn trước khi giao.'
CANCEL_MESSAGE = ('Dạ, thật tiếc quá, em đã hủy yêu cầu đặt hàng của anh/chị rồi nha. '
                  'Anh/chị có cần em hỗ trợ hay tư vấn gì thêm nữa không?')
THANKS_FALLBACK = 'Dạ, em cảm ơn anh/chị đã tin tưởng chọn bên em nha. Chúc anh/chị một ngày vui vẻ!'
THANKS_PROMPT = dedent("""
    Bạn là nhân viên bán hàng của cửa hàng điện thoại. Khách vừa đặt hàng xong, phần tóm tắt đơn đã gửi cho khách.
    Viết 1-2 câu tiếng Việt cảm ơn khách đã lựa chọn cửa hàng, xưng "em", gọi khách là "anh/chị" kèm tên.
    Giọng miền Nam lịch sự, thân thiện (có thể dùng "Dạ", "nha", "bên em"); không dùng chữ "ạ", không suồng sã.
    Không nhắc lại sản phẩm, giá, mã đơn; không nói về phí, thời gian giao, khuyến mãi hay bảo hành; không đặt câu hỏi.
""").strip()

class OrderError(Exception):
    """Lỗi khách cần biết (hết hàng, giỏ trống...): api.py trả nguyên văn cho giao diện."""

def _connect():
    return psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row)

def _new_id(prefix: str) -> str:
    # Bảng dùng khóa text, không có sequence: DH3F9A1C2B (đơn), KH8E21D0F4 (khách)
    return prefix + uuid.uuid4().hex[:8].upper()

def ensure_tables() -> int:
    """Tạo/bổ sung bảng, rồi mã hóa số điện thoại, email còn ở dạng chữ thường (dữ liệu trước khi có mã hóa).
    Trả về số dòng vừa mã hóa."""
    with _connect() as conn:
        conn.execute(SCHEMA.read_text(encoding='utf-8'))
        return encrypt_existing(conn)

def encrypt_existing(conn) -> int:
    # Chạy lại nhiều lần được: chỉ đụng dòng chưa có tiền tố enc: hoặc chưa có phone_hash
    changed = 0
    with conn.transaction():
        for c in conn.execute("SELECT customer_id, phone, email FROM customers "
                              "WHERE phone_hash IS NULL OR phone NOT LIKE 'enc:%%' OR email NOT LIKE 'enc:%%'").fetchall():
            phone, email = pii.decrypt(c['phone']), pii.decrypt(c['email'])
            conn.execute('UPDATE customers SET phone = %s, phone_hash = %s, email = %s, email_hash = %s '
                         'WHERE customer_id = %s',
                         (pii.encrypt(phone), pii.phone_key(phone) if phone else None, pii.encrypt(email),
                          pii.email_key(email), c['customer_id']))
            changed += 1
        for o in conn.execute("SELECT order_id, recipient_phone, recipient_email FROM orders "
                              "WHERE recipient_phone NOT LIKE 'enc:%%' OR recipient_email NOT LIKE 'enc:%%'").fetchall():
            conn.execute('UPDATE orders SET recipient_phone = %s, recipient_email = %s WHERE order_id = %s',
                         (pii.encrypt(pii.decrypt(o['recipient_phone'])), pii.encrypt(pii.decrypt(o['recipient_email'])),
                          o['order_id']))
            changed += 1
    if changed:
        log.warning('Đã mã hóa số điện thoại, email của %d dòng khách hàng/đơn hàng cũ', changed)
    return changed

def quote(cart: list[dict], promo_code: str | None = None) -> dict:
    """Tạm tính, giảm giá, tổng tiền cho form đặt hàng (chưa biết khách nên chưa kiểm tra số lần khách đã dùng mã;
    create_order kiểm tra lại đầy đủ)."""
    if not cart:
        raise OrderError('Giỏ hàng đang trống, anh/chị chọn sản phẩm trước khi đặt hàng giúp em nha.')
    with _connect() as conn:
        return promotions.quote(conn, cart, promo_code)

def create_order(cart: list[dict], name: str, phone: str, address: str, fulfillment_type: str,
                 payment: str, thread_id: str | None = None, email: str | None = None,
                 promo_code: str | None = None) -> str:
    """Lưu đơn trong 1 transaction: trừ tồn kho (đủ hàng mới trừ), ghi khách hàng (khách cũ nhận theo số
    điện thoại), đơn và từng sản phẩm. Thiếu hàng thì không ghi gì cả. Trả về mã đơn.
    email (không bắt buộc) luôn ghi vào đơn; chỉ ghi vào khách hàng khi chưa thuộc khách khác (uq_customers_email_hash).
    Số điện thoại, email lưu dạng mã hóa (pii.encrypt), tra cứu bằng HMAC (pii.phone_key, pii.email_key).
    promo_code: mã khuyến mãi khách nhập (không nhập thì áp khuyến mãi tự động nếu có); mã không dùng được thì
    PromoError, không ghi gì cả. Dùng mã thì ghi promotion_usages và tăng used_count."""
    if not cart:
        raise OrderError('Giỏ hàng đang trống, anh/chị chọn sản phẩm trước khi đặt hàng giúp em nha.')
    order_id = _new_id('DH')
    # Mã hóa trước khi mở transaction: thiếu PII_KEY thì báo lỗi ngay, không trừ tồn kho
    phone_enc, email_enc = pii.encrypt(pii.normalize_phone(phone)), pii.encrypt(email)
    phone_h, email_h = pii.phone_key(phone), pii.email_key(email)
    with _connect() as conn, conn.transaction():
        for i in cart:
            done = conn.execute('UPDATE variants SET stock = stock - %s WHERE variant_id = %s AND stock >= %s',
                                (i['quantity'], i['variant_id'], i['quantity'])).rowcount
            if not done:
                raise OrderError(f'{_item_name(i)} không còn đủ hàng. Anh/chị chọn phiên bản khác giúp em nha.')
        # Khách cũ (cùng số điện thoại) thì cập nhật tên, địa chỉ mới nhất
        found = conn.execute('SELECT customer_id FROM customers WHERE phone_hash = %s ORDER BY created_at LIMIT 1',
                             (phone_h,)).fetchone()
        customer_email, customer_email_h = email_enc, email_h
        if email:
            # Email đã thuộc khách khác (số điện thoại khác): ghi vào khách sẽ vi phạm uq_customers_email_hash và
            # hỏng cả đơn -> chỉ ghi vào đơn, khách giữ email cũ
            owner = conn.execute('SELECT customer_id FROM customers WHERE email_hash = %s LIMIT 1',
                                 (email_h,)).fetchone()
            if owner and not (found and owner['customer_id'] == found['customer_id']):
                log.info('Email đơn mới đã thuộc khách %s, không gán cho khách đặt đơn này', owner['customer_id'])
                customer_email = customer_email_h = None
        if found:
            customer_id = found['customer_id']
            # Không nhập email thì giữ email cũ của khách
            conn.execute('UPDATE customers SET name = %s, default_address = %s, email = COALESCE(%s, email), '
                         'email_hash = COALESCE(%s, email_hash), updated_at = now() WHERE customer_id = %s',
                         (name, address, customer_email, customer_email_h, customer_id))
        else:
            customer_id = _new_id('KH')
            conn.execute('INSERT INTO customers (customer_id, name, phone, phone_hash, default_address, email, email_hash) '
                         'VALUES (%s, %s, %s, %s, %s, %s, %s)',
                         (customer_id, name, phone_enc, phone_h, address, customer_email, customer_email_h))
        # Tính giảm giá khi đã biết khách (giới hạn số lần mỗi khách); khóa dòng khuyến mãi tới hết transaction
        price = promotions.quote(conn, cart, promo_code, customer_id, lock=True)
        promo = price['promo']
        conn.execute(
            """INSERT INTO orders (order_id, customer_id, thread_id, recipient_name, recipient_phone, recipient_email,
                                   address, fulfillment_type, payment, subtotal, discount_amount, promo_id, total)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (order_id, customer_id, thread_id, name, phone_enc, email_enc, address, fulfillment_type, payment,
             price['subtotal'], price['discount'], promo['promo_id'] if promo else None, price['total']))
        with conn.cursor() as cur:
            cur.executemany('INSERT INTO order_items (order_id, variant_id, qty, unit_price) VALUES (%s, %s, %s, %s)',
                            [(order_id, i['variant_id'], i['quantity'], i['price']) for i in cart])
        if promo:
            promotions.record_usage(conn, promo, order_id, customer_id, price['discount'])
    return order_id

def get_order(order_id: str) -> dict:
    with _connect() as conn:
        # Người nhận lấy từ chính đơn: customers chỉ giữ tên mới nhất theo số điện thoại, đơn sau cùng số
        # với tên khác sẽ ghi đè (đơn cũ hiện sai tên nếu đọc từ customers)
        order = conn.execute(
            """SELECT o.order_id, o.customer_id, o.recipient_name AS name, o.recipient_phone AS phone, o.recipient_email AS email,
                      o.address, o.fulfillment_type, o.payment, o.subtotal, o.discount_amount, o.total, o.status,
                      p.code AS promo_code, p.description AS promo_description
               FROM orders o LEFT JOIN promotions p ON p.promo_id = o.promo_id
               WHERE o.order_id = %s""", (order_id,)).fetchone()
        items = conn.execute(
            """SELECT p.name, v.storage, v.color, oi.unit_price, oi.qty
               FROM order_items oi JOIN variants v ON v.variant_id = oi.variant_id
               JOIN products p ON p.product_id = v.product_id
               WHERE oi.order_id = %s ORDER BY oi.unit_price DESC""", (order_id,)).fetchall()
    # Giải mã ở đây; summary/view tự che khi hiển thị
    return {**order, 'phone': pii.decrypt(order['phone']), 'email': pii.decrypt(order['email']), 'items': items}

def customer_profile(customer_id: str) -> dict | None:
    """Khách quen cho phiên chat mới (api.py: POST /session): tên, số đơn, đơn gần nhất (mã, ngày, tên món).
    Không lấy số điện thoại, email, địa chỉ: thông tin này đưa vào trạng thái hội thoại, gửi cho LLM.
    Khách không tồn tại hoặc chưa có đơn nào thì None."""
    with _connect() as conn:
        c = conn.execute(
            """SELECT c.customer_id, c.name, count(o.order_id) AS order_count, max(o.created_at) AS last_at
               FROM customers c JOIN orders o ON o.customer_id = c.customer_id
               WHERE c.customer_id = %s GROUP BY c.customer_id, c.name""", (customer_id,)).fetchone()
        if not c:
            return None
        last = conn.execute(
            """SELECT o.order_id, o.created_at FROM orders o WHERE o.customer_id = %s
               ORDER BY o.created_at DESC LIMIT 1""", (customer_id,)).fetchone()
        items = conn.execute(
            """SELECT p.name, v.storage, v.color, oi.qty FROM order_items oi
               JOIN variants v ON v.variant_id = oi.variant_id JOIN products p ON p.product_id = v.product_id
               WHERE oi.order_id = %s ORDER BY oi.unit_price DESC""", (last['order_id'],)).fetchall()
    return {'customer_id': c['customer_id'], 'name': c['name'], 'order_count': c['order_count'],
            'last_order': {'order_id': last['order_id'],
                           'date': last['created_at'].strftime('%d/%m/%Y') if last['created_at'] else None,
                           'items': [_item_name(i) for i in items]}}

def welcome_back(profile: dict) -> str:
    """Lời chào khách quen (template, không gọi LLM): tên, máy/món đơn gần nhất."""
    items = profile['last_order']['items']
    bought = f" Lần trước anh/chị đã đặt {items[0]}{' và ' + str(len(items) - 1) + ' món khác' if len(items) > 1 else ''}." if items else ''
    return (f"Dạ, em chào anh/chị {profile['name']}, rất vui được gặp lại anh/chị nha.{bought} "
            'Hôm nay anh/chị cần em tư vấn thêm điện thoại, phụ kiện hay chính sách gì cứ nói em nha.')

EMAIL_STATUSES = ('pending', 'sent', 'failed', 'skipped')

def set_email_status(order_id: str, status: str) -> None:
    assert status in EMAIL_STATUSES
    with _connect() as conn:
        conn.execute('UPDATE orders SET email_status = %s WHERE order_id = %s', (status, order_id))

def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'

def _item_name(i: dict) -> str:
    return ' '.join(filter(None, [i['name'], i.get('storage'), i.get('color')]))

def summary(order: dict) -> str:
    """Tóm tắt đơn dạng chữ (ghi vào lịch sử hội thoại), cùng bố cục thẻ xác nhận .receipt trên giao diện.
    Số điện thoại, email đã che: tóm tắt nằm trong lịch sử, lượt sau sẽ được gửi cho LLM."""
    items = [f"- {i['qty']} x {_item_name(i)}: {_price(i['unit_price'] * i['qty'])}" for i in order['items']]
    return '\n'.join([
        f"Đặt hàng thành công. Mã đơn: **#{order['order_id']}**",
        '',
        'Thông tin nhận hàng:',
        f"- Người nhận: {order['name']}",
        f"- Điện thoại: {pii.mask_phone(order['phone'])}",
        *([f"- Email: {pii.mask_email(order['email'])}"] if order.get('email') else []),
        f"- Địa chỉ: {order['address']}",
        f"- Nhận hàng: {FULFILLMENT.get(order['fulfillment_type'], order['fulfillment_type'])}",
        f"- Thanh toán: {PAYMENT.get(order['payment'], order['payment'])}",
        '',
        f"Sản phẩm ({len(order['items'])}):",
        *items,
        '',
        *_discount_lines(order),
        f"Tổng cộng: **{_price(order['total'])}**",
        CALLBACK_NOTE,
    ])

def _promo_label(order: dict) -> str | None:
    return order.get('promo_code') or order.get('promo_description')

def _discount_lines(order: dict) -> list[str]:
    # Đơn có giảm giá: tạm tính + dòng giảm; đơn cũ (trước khi có khuyến mãi) hay không giảm thì chỉ có tổng
    if not order.get('discount_amount'):
        return []
    return [f"Tạm tính: {_price(order['subtotal'])}",
            f"Giảm giá ({_promo_label(order)}): -{_price(order['discount_amount'])}"]

def view(order: dict) -> dict:
    """Dữ liệu đơn cho thẻ xác nhận trên giao diện (cùng nguồn database với summary), số điện thoại, email đã che."""
    return {
        'order_id': order['order_id'], 'name': order['name'], 'phone': pii.mask_phone(order['phone']),
        'email': pii.mask_email(order.get('email')),
        'address': order['address'],
        'fulfillment': FULFILLMENT.get(order['fulfillment_type'], order['fulfillment_type']),
        'payment': PAYMENT.get(order['payment'], order['payment']),
        'items': [{'name': i['name'], 'variant': ' · '.join(filter(None, [i['storage'], i['color']])),
                   'qty': i['qty'], 'unit_price': i['unit_price']} for i in order['items']],
        'subtotal': order.get('subtotal') or order['total'],
        'discount': order.get('discount_amount') or 0,
        'promo': _promo_label(order) if order.get('discount_amount') else None,
        'total': order['total'],
        'note': CALLBACK_NOTE,
    }

_chain = FallbackMiddleware(*llm.FALLBACK_LLMS)

def thank_you(name: str) -> str:
    # Chỉ đưa tên khách cho LLM: thông tin đơn đã có trong phần tóm tắt, không để LLM viết lại số liệu
    messages = [SystemMessage(THANKS_PROMPT), HumanMessage(f'Tên khách: {name}')]
    try:
        text = _chain.call(llm.open_llm, lambda m: m.invoke(messages)).text.strip()
    except Exception as e:
        log.warning('Không tạo được lời cảm ơn bằng LLM (%s), dùng câu mặc định', type(e).__name__)
        return THANKS_FALLBACK
    return text or THANKS_FALLBACK
