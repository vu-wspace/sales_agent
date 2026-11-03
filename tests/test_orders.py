from types import SimpleNamespace

import pytest

from phone_sales import orders, pii

CART = [{'variant_id': 'S26-256-DEN', 'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen Classic',
         'price': 29990001, 'quantity': 1},
        {'variant_id': 'SAC-60W', 'name': 'Củ Sạc Không Kèm Cáp 60W (Đen)', 'storage': None, 'color': None,
         'price': 1089720, 'quantity': 2}]
ADDRESS = '12 Lê Lợi, Quy Nhơn'


class FakeConn:
    """Giả lập kết nối psycopg: ghi lại câu SQL; UPDATE tồn kho thành công nếu variant còn đủ trong `stock`,
    `customer` là khách đã có cùng số điện thoại (None = khách mới), `email_owner` là khách đang giữ email,
    `promos` là các dòng bảng promotions, `used` là số lần khách đã dùng mã (promotion_usages)."""

    def __init__(self, stock: dict, customer: str | None = None, email_owner: str | None = None,
                 promos: list | None = None, used: int = 0):
        self.stock = stock
        self.customer = customer
        self.email_owner = email_owner
        self.promos = promos or []
        self.used = used
        self.sql = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def transaction(self):
        return self

    def execute(self, sql, params=()):
        self.sql.append((' '.join(sql.split()), params))
        if sql.startswith('UPDATE variants'):
            qty, vid, _ = params
            return SimpleNamespace(rowcount=int(self.stock.get(vid, 0) >= qty))
        if 'FROM variants v JOIN products' in sql:   # promotions.cart_items: hãng, loại theo variant
            rows = [{'variant_id': v, 'product_id': 'p-' + v, 'brand': 'Samsung',
                     'category': 'accessory' if v.startswith('SAC') else 'phone'} for v in params[0]]
            return SimpleNamespace(fetchall=lambda: rows)
        if 'FROM promotions WHERE upper(code)' in sql:
            found = next((p for p in self.promos if (p['code'] or '').upper() == params[0].upper()), None)
            return SimpleNamespace(fetchone=lambda: found)
        if 'FROM promotions WHERE code IS NULL' in sql:
            return SimpleNamespace(fetchall=lambda: [p for p in self.promos if p['code'] is None])
        if 'FROM promotion_usages' in sql:
            return SimpleNamespace(fetchone=lambda: {'n': self.used})
        owner = self.email_owner if 'email_hash =' in sql else self.customer
        found = {'customer_id': owner} if owner else None
        return SimpleNamespace(fetchone=lambda: found)

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def executemany(self, sql, rows):
                conn.sql.append((' '.join(sql.split()), rows))
        return Cur()

    def find(self, start: str):
        return [p for s, p in self.sql if s.startswith(start)]


def test_create_order_new_customer(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5})
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'Nguyễn Văn A', '0912 345 678', ADDRESS, 'delivery', 'cod', 't1')
    assert order_id.startswith('DH') and len(order_id) == 10
    assert len(conn.find('UPDATE variants')) == 2
    # Khách cũ tra bằng HMAC của số điện thoại, không tra bằng số gốc
    assert conn.find('SELECT customer_id FROM customers WHERE phone_hash') == [(pii.phone_key('0912345678'),)]
    (customer_id, name, phone, phone_h, address, email, email_h), = conn.find('INSERT INTO customers')
    assert customer_id.startswith('KH') and (name, address, email, email_h) == ('Nguyễn Văn A', ADDRESS, None, None)
    assert pii.is_encrypted(phone) and pii.decrypt(phone) == '0912345678' and phone_h == pii.phone_key('0912345678')
    (order,) = conn.find('INSERT INTO orders')
    total = 29990001 + 2 * 1089720
    assert order[:4] == (order_id, customer_id, 't1', 'Nguyễn Văn A') and order[5:] == (
        None, ADDRESS, 'delivery', 'cod', total, 0, None, total)       # subtotal, giảm 0, không mã, total
    assert pii.decrypt(order[4]) == '0912345678' and '0912345678' not in str(conn.sql)  # không lưu số gốc
    assert not conn.find('SELECT customer_id FROM customers WHERE email_hash')  # không nhập email: không tra
    assert conn.find('INSERT INTO order_items')[0] == [(order_id, 'S26-256-DEN', 1, 29990001),
                                                       (order_id, 'SAC-60W', 2, 1089720)]


def test_create_order_existing_customer_by_phone(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, customer='KH00000001')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'Nguyễn Văn B', '0912345678', ADDRESS, 'delivery', 'cod')
    assert not conn.find('INSERT INTO customers')
    assert conn.find('UPDATE customers') == [('Nguyễn Văn B', ADDRESS, None, None, 'KH00000001')]  # giữ email cũ
    assert conn.find('INSERT INTO orders')[0][:2] == (order_id, 'KH00000001')


def test_create_order_with_email(monkeypatch):
    # Email chưa ai dùng: ghi (mã hóa) vào khách mới và đơn
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5})
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='a@b.vn')
    *_, email, email_h = conn.find('INSERT INTO customers')[0]
    assert pii.decrypt(email) == 'a@b.vn' and email_h == pii.email_key('a@b.vn')
    assert pii.decrypt(conn.find('INSERT INTO orders')[0][5]) == 'a@b.vn' and 'a@b.vn' not in str(conn.sql)
    # Email là của chính khách cũ (cùng số điện thoại, khác hoa thường): cập nhật bình thường
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, customer='KH1', email_owner='KH1')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='A@B.vn')
    (_, _, email, email_h, customer_id), = conn.find('UPDATE customers')
    assert pii.decrypt(email) == 'A@B.vn' and email_h == pii.email_key('a@b.vn') and customer_id == 'KH1'


def test_create_order_email_of_other_customer_still_saved_on_order(monkeypatch):
    # uq_customers_email_hash: email đã thuộc khách khác thì không gán cho khách này (tránh lỗi cả đơn), đơn vẫn lưu
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, email_owner='KH_OTHER')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='a@b.vn')
    assert conn.find('INSERT INTO customers')[0][-2:] == (None, None)
    (order,) = conn.find('INSERT INTO orders')
    assert order[0] == order_id and pii.decrypt(order[5]) == 'a@b.vn'


def test_create_order_without_pii_key_writes_nothing(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5})
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    monkeypatch.setattr(pii.config, 'PII_KEY', '')
    with pytest.raises(pii.PiiKeyError, match='PII_KEY'):
        orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod')
    assert conn.sql == []  # báo lỗi trước khi trừ tồn kho


def test_create_order_out_of_stock_or_empty_cart(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 1})
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    with pytest.raises(orders.OrderError, match=r'Củ Sạc Không Kèm Cáp 60W \(Đen\) không còn đủ hàng'):
        orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod')
    assert not conn.find('INSERT')   # lỗi giữa transaction: không ghi khách, đơn
    with pytest.raises(orders.OrderError, match='Giỏ hàng đang trống'):
        orders.create_order([], 'A B', '0912345678', ADDRESS, 'delivery', 'cod')


def test_summary_template_from_database_row():
    order = {'order_id': 'DH3F9A1C2B', 'name': 'Nguyễn Văn A', 'phone': '0912345678', 'address': ADDRESS,
             'fulfillment_type': 'delivery', 'payment': 'cod', 'total': 32169441, 'status': 'pending',
             'items': [{'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen Classic', 'unit_price': 29990001, 'qty': 1},
                       {'name': 'Củ Sạc 60W', 'storage': None, 'color': None, 'unit_price': 1089720, 'qty': 2}]}
    # Số điện thoại, email đã che: tóm tắt nằm trong lịch sử hội thoại, lượt sau gửi cho LLM
    with_email = orders.summary({**order, 'email': 'nguyenvana@gmail.com'}).splitlines()
    assert with_email[5] == '- Email: ng***@gmail.com'
    assert orders.view({**order, 'email': 'nguyenvana@gmail.com'})['email'] == 'ng***@gmail.com'
    assert orders.summary(order).splitlines() == [
        'Đặt hàng thành công. Mã đơn: **#DH3F9A1C2B**',
        '',
        'Thông tin nhận hàng:',
        '- Người nhận: Nguyễn Văn A',
        '- Điện thoại: 0912 *** 678',
        '- Địa chỉ: 12 Lê Lợi, Quy Nhơn',
        '- Nhận hàng: Giao tận nơi',
        '- Thanh toán: Thanh toán khi nhận hàng',
        '',
        'Sản phẩm (2):',
        '- 1 x Galaxy S26 256GB Đen Classic: 29.990.001đ',
        '- 2 x Củ Sạc 60W: 2.179.440đ',
        '',
        'Tổng cộng: **32.169.441đ**',
        'Nhân viên cửa hàng sẽ gọi điện xác nhận đơn trước khi giao.',
    ]
    # Cùng dữ liệu cho thẻ xác nhận trên giao diện
    card = orders.view(order)
    assert card['phone'] == '0912 *** 678' and card['email'] is None
    assert card['fulfillment'] == 'Giao tận nơi' and card['payment'] == 'Thanh toán khi nhận hàng'
    assert card['items'] == [{'name': 'Galaxy S26', 'variant': '256GB · Đen Classic', 'qty': 1, 'unit_price': 29990001},
                             {'name': 'Củ Sạc 60W', 'variant': '', 'qty': 2, 'unit_price': 1089720}]
    assert card['total'] == 32169441 and card['note'] == orders.CALLBACK_NOTE


def test_thank_you_uses_llm_and_falls_back(monkeypatch):
    def call(primary, invoke):
        return invoke(SimpleNamespace(invoke=lambda messages: SimpleNamespace(text=' Cảm ơn anh A! ')))
    monkeypatch.setattr(orders._chain, 'call', call)
    assert orders.thank_you('A') == 'Cảm ơn anh A!'

    def boom(primary, invoke):
        raise RuntimeError('quota')
    monkeypatch.setattr(orders._chain, 'call', boom)
    assert orders.thank_you('A') == orders.THANKS_FALLBACK


def test_welcome_back_mentions_name_and_last_purchase():
    profile = {'name': 'Nguyễn Văn A', 'last_order': {'items': ['iPhone 17 512GB Đen', 'Ốp lưng', 'Củ sạc']}}
    text = orders.welcome_back(profile)
    assert text.startswith('Dạ, em chào anh/chị Nguyễn Văn A') and 'iPhone 17 512GB Đen và 2 món khác' in text
    assert ' ạ' not in text
