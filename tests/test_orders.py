from types import SimpleNamespace

import pytest

from phone_sales import orders

CART = [{'variant_id': 'S26-256-DEN', 'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen Classic',
         'price': 29990001, 'quantity': 1},
        {'variant_id': 'SAC-60W', 'name': 'Củ Sạc Không Kèm Cáp 60W (Đen)', 'storage': None, 'color': None,
         'price': 1089720, 'quantity': 2}]
ADDRESS = '12 Lê Lợi, Quy Nhơn'


class FakeConn:
    """Giả lập kết nối psycopg: ghi lại câu SQL; UPDATE tồn kho thành công nếu variant còn đủ trong `stock`,
    `customer` là khách đã có cùng số điện thoại (None = khách mới), `email_owner` là khách đang giữ email."""

    def __init__(self, stock: dict, customer: str | None = None, email_owner: str | None = None):
        self.stock = stock
        self.customer = customer
        self.email_owner = email_owner
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
        owner = self.email_owner if 'lower(email)' in sql else self.customer
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
    order_id = orders.create_order(CART, 'Nguyễn Văn A', '0912345678', ADDRESS, 'delivery', 'cod', 't1')
    assert order_id.startswith('DH') and len(order_id) == 10
    assert len(conn.find('UPDATE variants')) == 2
    (customer_id, *customer), = conn.find('INSERT INTO customers')
    assert customer_id.startswith('KH') and customer == ['Nguyễn Văn A', '0912345678', ADDRESS, None]
    assert conn.find('INSERT INTO orders') == [(order_id, customer_id, 't1', 'Nguyễn Văn A', '0912345678', None,
                                               ADDRESS, 'delivery', 'cod', 29990001 + 2 * 1089720)]
    assert not conn.find('SELECT customer_id FROM customers WHERE lower(email)')  # không nhập email: không tra
    assert conn.find('INSERT INTO order_items')[0] == [(order_id, 'S26-256-DEN', 1, 29990001),
                                                       (order_id, 'SAC-60W', 2, 1089720)]


def test_create_order_existing_customer_by_phone(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, customer='KH00000001')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'Nguyễn Văn B', '0912345678', ADDRESS, 'delivery', 'cod')
    assert not conn.find('INSERT INTO customers')
    assert conn.find('UPDATE customers') == [('Nguyễn Văn B', ADDRESS, None, 'KH00000001')]  # giữ email cũ
    assert conn.find('INSERT INTO orders')[0][:2] == (order_id, 'KH00000001')


def test_create_order_with_email(monkeypatch):
    # Email chưa ai dùng: ghi vào khách mới và đơn
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5})
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='a@b.vn')
    assert conn.find('INSERT INTO customers')[0][-1] == 'a@b.vn' and conn.find('INSERT INTO orders')[0][5] == 'a@b.vn'
    # Email là của chính khách cũ (cùng số điện thoại): cập nhật bình thường
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, customer='KH1', email_owner='KH1')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='A@B.vn')
    assert conn.find('UPDATE customers') == [('A B', ADDRESS, 'A@B.vn', 'KH1')]


def test_create_order_email_of_other_customer_still_saved_on_order(monkeypatch):
    # uq_customers_email: email đã thuộc khách khác thì không gán cho khách này (tránh lỗi cả đơn), đơn vẫn lưu email
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, email_owner='KH_OTHER')
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', email='a@b.vn')
    assert conn.find('INSERT INTO customers')[0][-1] is None
    assert conn.find('INSERT INTO orders')[0][:1] == (order_id,) and conn.find('INSERT INTO orders')[0][5] == 'a@b.vn'


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
    with_email = orders.summary({**order, 'email': 'a@b.vn'}).splitlines()
    assert with_email[5] == '- Email: a@b.vn' and orders.view({**order, 'email': 'a@b.vn'})['email'] == 'a@b.vn'
    assert orders.summary(order).splitlines() == [
        'Đặt hàng thành công. Mã đơn: **#DH3F9A1C2B**',
        '',
        'Thông tin nhận hàng:',
        '- Người nhận: Nguyễn Văn A',
        '- Điện thoại: 0912 345 678',
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
