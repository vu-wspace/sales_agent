# Đơn đặt hàng từ form trên khung chat: lưu khách hàng + đơn vào Postgres (bảng customers, orders, order_items),
# tóm tắt đơn bằng template lấy thẳng từ database, chỉ dùng LLM cho 1-2 câu cảm ơn cuối.
import logging
import uuid
from textwrap import dedent

import psycopg
from langchain_core.messages import HumanMessage, SystemMessage
from psycopg.rows import dict_row

from phone_sales import llm
from phone_sales.config import DATABASE_URL, ROOT
from phone_sales.fallback import FallbackMiddleware

log = logging.getLogger(__name__)
SCHEMA = ROOT / 'data_schema' / 'orders.sql'

FULFILLMENT = {'delivery': 'Giao tận nơi', 'pickup': 'Nhận tại cửa hàng'}
PAYMENT = {'cod': 'Thanh toán khi nhận hàng', 'bank_transfer': 'Chuyển khoản', 'installment': 'Trả góp'}
CALLBACK_NOTE = 'Nhân viên cửa hàng sẽ gọi điện xác nhận đơn trước khi giao.'
CANCEL_MESSAGE = ('Thật tiếc, em đã hủy yêu cầu đặt hàng của anh/chị. '
                  'Anh/chị có cần em hỗ trợ hay tư vấn gì thêm nữa không?')
THANKS_FALLBACK ='Cảm ơn anh/chị đã tin tưởng lựa chọn cửa hàng. Chúc anh/chị một ngày vui vẻ!'
THANKS_PROMPT = dedent("""
    Bạn là nhân viên bán hàng của cửa hàng điện thoại. Khách vừa đặt hàng xong, phần tóm tắt đơn đã gửi cho khách.
    Viết 1-2 câu tiếng Việt cảm ơn khách đã lựa chọn cửa hàng, xưng "em", gọi khách là "anh/chị" kèm tên.
    Không nhắc lại sản phẩm, giá, mã đơn; không nói về phí, thời gian giao, khuyến mãi hay bảo hành; không đặt câu hỏi.
""").strip()

class OrderError(Exception):
    """Lỗi khách cần biết (hết hàng, giỏ trống...): api.py trả nguyên văn cho giao diện."""

def _connect():
    return psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row)

def _new_id(prefix: str) -> str:
    # Bảng dùng khóa text, không có sequence: DH3F9A1C2B (đơn), KH8E21D0F4 (khách)
    return prefix + uuid.uuid4().hex[:8].upper()

def ensure_tables() -> None:
    with _connect() as conn:
        conn.execute(SCHEMA.read_text(encoding='utf-8'))

def create_order(cart: list[dict], name: str, phone: str, address: str, fulfillment_type: str,
                 payment: str, thread_id: str | None = None) -> str:
    """Lưu đơn trong 1 transaction: trừ tồn kho (đủ hàng mới trừ), ghi khách hàng (khách cũ nhận theo số
    điện thoại), đơn và từng sản phẩm. Thiếu hàng thì không ghi gì cả. Trả về mã đơn."""
    if not cart:
        raise OrderError('Giỏ hàng đang trống, anh/chị chọn sản phẩm trước khi đặt hàng nhé.')
    total = sum(i['price'] * i['quantity'] for i in cart)
    order_id = _new_id('DH')
    with _connect() as conn, conn.transaction():
        for i in cart:
            done = conn.execute('UPDATE variants SET stock = stock - %s WHERE variant_id = %s AND stock >= %s',
                                (i['quantity'], i['variant_id'], i['quantity'])).rowcount
            if not done:
                raise OrderError(f'{_item_name(i)} không còn đủ hàng. Anh/chị chọn phiên bản khác giúp em nhé.')
        # Khách cũ (cùng số điện thoại) thì cập nhật tên, địa chỉ mới nhất
        found = conn.execute('SELECT customer_id FROM customers WHERE phone = %s ORDER BY created_at LIMIT 1',
                             (phone,)).fetchone()
        if found:
            customer_id = found['customer_id']
            conn.execute('UPDATE customers SET name = %s, default_address = %s, updated_at = now() '
                         'WHERE customer_id = %s', (name, address, customer_id))
        else:
            customer_id = _new_id('KH')
            conn.execute('INSERT INTO customers (customer_id, name, phone, default_address) VALUES (%s, %s, %s, %s)',
                         (customer_id, name, phone, address))
        conn.execute(
            """INSERT INTO orders (order_id, customer_id, thread_id, recipient_name, recipient_phone, address,
                                   fulfillment_type, payment, total)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (order_id, customer_id, thread_id, name, phone, address, fulfillment_type, payment, total))
        with conn.cursor() as cur:
            cur.executemany('INSERT INTO order_items (order_id, variant_id, qty, unit_price) VALUES (%s, %s, %s, %s)',
                            [(order_id, i['variant_id'], i['quantity'], i['price']) for i in cart])
    return order_id

def get_order(order_id: str) -> dict:
    with _connect() as conn:
        # Người nhận lấy từ chính đơn: customers chỉ giữ tên mới nhất theo số điện thoại, đơn sau cùng số
        # với tên khác sẽ ghi đè (đơn cũ hiện sai tên nếu đọc từ customers)
        order = conn.execute(
            """SELECT order_id, recipient_name AS name, recipient_phone AS phone, address, fulfillment_type,
                      payment, total, status
               FROM orders WHERE order_id = %s""", (order_id,)).fetchone()
        items = conn.execute(
            """SELECT p.name, v.storage, v.color, oi.unit_price, oi.qty
               FROM order_items oi JOIN variants v ON v.variant_id = oi.variant_id
               JOIN products p ON p.product_id = v.product_id
               WHERE oi.order_id = %s ORDER BY oi.unit_price DESC""", (order_id,)).fetchall()
    return {**order, 'items': items}

def _price(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'

def _item_name(i: dict) -> str:
    return ' '.join(filter(None, [i['name'], i.get('storage'), i.get('color')]))

def summary(order: dict) -> str:
    items = '\n'.join(f"- {i['qty']} x {_item_name(i)}: {_price(i['unit_price'] * i['qty'])}" for i in order['items'])
    return '\n'.join([
        f"Đơn hàng **#{order['order_id']}** đã được tạo.",
        f"Khách hàng: {order['name']}",
        f"Số điện thoại: {order['phone']}",
        f"Địa chỉ giao hàng: {order['address']}",
        f"Cách nhận hàng: {FULFILLMENT.get(order['fulfillment_type'], order['fulfillment_type'])}",
        f"Thanh toán: {PAYMENT.get(order['payment'], order['payment'])}",
        f"Sản phẩm:\n{items}",
        f"Tổng cộng: **{_price(order['total'])}**",
        CALLBACK_NOTE,
    ])

def view(order: dict) -> dict:
    """Dữ liệu đơn cho thẻ xác nhận trên giao diện (cùng nguồn database với summary)."""
    return {
        'order_id': order['order_id'], 'name': order['name'], 'phone': order['phone'], 'address': order['address'],
        'fulfillment': FULFILLMENT.get(order['fulfillment_type'], order['fulfillment_type']),
        'payment': PAYMENT.get(order['payment'], order['payment']),
        'items': [{'name': i['name'], 'variant': ' · '.join(filter(None, [i['storage'], i['color']])),
                   'qty': i['qty'], 'unit_price': i['unit_price']} for i in order['items']],
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
