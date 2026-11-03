import asyncio
from email import message_from_bytes
from email.header import decode_header, make_header

import pytest

from phone_sales import config, mailer

ORDER = {
    'order_id': 'DH00000001', 'name': 'Nguyễn <b>Văn</b> A', 'phone': '0912345678', 'email': 'khach@example.com',
    'address': '12 Lê Lợi, Quy Nhơn', 'fulfillment_type': 'delivery', 'payment': 'cod',
    'subtotal': 36_498_000, 'discount_amount': 500_000, 'total': 35_998_000, 'status': 'pending',
    'promo_code': 'DEMO-GIAM500K', 'promo_description': 'Giảm 500.000đ',
    'items': [{'name': 'iPhone 17', 'storage': '512GB', 'color': 'Đen', 'unit_price': 35_499_000, 'qty': 1},
              {'name': 'Bộ Tiếp Hợp Nguồn 40W', 'storage': None, 'color': None, 'unit_price': 999_000, 'qty': 1}],
}


def test_html_lists_items_discount_total_and_escapes_customer_input():
    html = mailer.render_html(mailer.email_data(ORDER))
    assert 'Xác nhận đơn hàng #DH00000001' in html
    assert 'iPhone 17 512GB Đen' in html and '35.499.000đ' in html and 'Bộ Tiếp Hợp Nguồn 40W' in html
    assert 'Giảm giá (DEMO-GIAM500K)' in html and '-500.000đ' in html and '35.998.000đ' in html
    assert 'Thanh toán khi nhận hàng' in html and '12 Lê Lợi, Quy Nhơn' in html
    assert '<b>Văn</b>' not in html and '&lt;b&gt;Văn&lt;/b&gt;' in html   # tên khách không chèn được HTML
    assert 'khach@example.com' not in html and '0912345678' not in html


def test_text_version_and_no_discount_lines_when_not_discounted():
    text = mailer.render_text(mailer.email_data({**ORDER, 'discount_amount': 0, 'subtotal': ORDER['total']}))
    assert '- iPhone 17 512GB Đen x1: 35.499.000đ' in text and 'Tổng cộng: 35.998.000đ' in text
    assert 'Giảm giá' not in text and 'Tạm tính' not in text


def test_pickup_order_has_no_address():
    data = mailer.email_data({**ORDER, 'fulfillment_type': 'pickup'})
    assert data['address'] is None and 'Địa chỉ' not in mailer.render_text(data)


def test_send_builds_multipart_html_and_text(monkeypatch):
    monkeypatch.setattr(config, 'SMTP_HOST', 'localhost')
    client = mailer.mail_client()
    client.config.SUPPRESS_SEND = 1
    monkeypatch.setattr(mailer, 'mail_client', lambda: client)
    with client.record_messages() as outbox:
        asyncio.run(mailer.send_order_confirmation(ORDER))
    msg = message_from_bytes(outbox[0].as_bytes())
    assert '<khach@example.com>' in msg['To'] and str(make_header(decode_header(msg['Subject']))) == 'Xác nhận đơn hàng #DH00000001 - PhoneShop Demo'
    types = [p.get_content_type() for p in msg.walk()]
    assert 'text/plain' in types and 'text/html' in types


@pytest.mark.parametrize('email, smtp_host, fail, expected', [
    ('khach@example.com', 'localhost', False, 'sent'),
    ('khach@example.com', 'localhost', True, 'failed'),
    (None, 'localhost', False, 'skipped'),          # khách không nhập email
    ('khach@example.com', '', False, 'skipped'),    # chưa cấu hình SMTP
])
def test_send_order_email_records_status(monkeypatch, email, smtp_host, fail, expected):
    statuses, sent = [], []
    monkeypatch.setattr(config, 'SMTP_HOST', smtp_host)
    monkeypatch.setattr(mailer.orders, 'get_order', lambda order_id: {**ORDER, 'email': email})
    monkeypatch.setattr(mailer.orders, 'set_email_status', lambda order_id, status: statuses.append((order_id, status)))
    async def send(order):
        if fail:
            raise ConnectionRefusedError('SMTP down')
        sent.append(order['order_id'])
    monkeypatch.setattr(mailer, 'send_order_confirmation', send)
    assert asyncio.run(mailer.send_order_email('DH00000001')) == expected
    assert statuses == [('DH00000001', expected)]
    assert sent == (['DH00000001'] if expected == 'sent' else [])
