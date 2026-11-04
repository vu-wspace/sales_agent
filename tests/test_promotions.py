from datetime import datetime, timedelta, timezone

import pytest

from phone_sales import orders, promotions
from test_orders import ADDRESS, CART, FakeConn

NOW = datetime.now(timezone.utc)


def promo(**kw):
    return {'promo_id': 'KM1', 'code': 'GIAM500K', 'description': 'Giảm cuối năm', 'discount_type': 'fixed',
            'value': 500_000, 'max_discount': None, 'min_order': 0, 'scope': 'all', 'scope_value': None,
            'starts_at': NOW - timedelta(days=1), 'ends_at': NOW + timedelta(days=30), 'usage_limit': None,
            'per_customer_limit': 1, 'used_count': 0, 'is_active': True, **kw}


ITEMS = [{'price': 30_000_000, 'quantity': 1, 'brand': 'Samsung', 'product_id': 'samsung-galaxy-s26', 'category': 'phone'},
         {'price': 500_000, 'quantity': 2, 'brand': 'Apple', 'product_id': 'acc-op', 'category': 'accessory'}]


@pytest.mark.parametrize('kw, expected', [
    ({}, 500_000),                                                                       # giảm cố định cả đơn
    ({'discount_type': 'percent', 'value': 10}, 3_100_000),                              # 10% của 31 triệu
    ({'discount_type': 'percent', 'value': 10, 'max_discount': 2_000_000}, 2_000_000),   # chạm mức tối đa
    ({'discount_type': 'percent', 'value': 10, 'scope': 'brand', 'scope_value': 'samsung'}, 3_000_000),
    ({'discount_type': 'percent', 'value': 10, 'scope': 'category', 'scope_value': 'accessory'}, 100_000),
    ({'scope': 'product', 'scope_value': 'acc-op', 'value': 5_000_000}, 1_000_000),      # không giảm quá phần áp dụng
])
def test_discount_for_scope_and_caps(kw, expected):
    assert promotions.discount_for(promo(**kw), ITEMS) == expected


def test_discount_outside_scope_or_unknown_scope():
    with pytest.raises(promotions.PromoError, match='không áp dụng cho sản phẩm trong giỏ'):
        promotions.discount_for(promo(scope='brand', scope_value='Xiaomi'), ITEMS)
    with pytest.raises(promotions.PromoError):
        promotions.discount_for(promo(scope='lạ'), ITEMS)   # scope lạ: không giảm nhầm cả đơn


@pytest.mark.parametrize('kw, used, match', [
    ({'is_active': False}, 0, 'chưa được áp dụng'),
    ({'starts_at': NOW + timedelta(days=1)}, 0, 'chưa được áp dụng'),
    ({'ends_at': NOW - timedelta(seconds=1)}, 0, 'đã hết hạn'),
    ({'usage_limit': 10, 'used_count': 10}, 0, 'hết lượt'),
    ({}, 1, 'đã dùng hết số lần'),                                  # per_customer_limit = 1
    ({'min_order': 50_000_000}, 0, 'đơn từ 50.000.000đ'),
])
def test_check_rejects(kw, used, match):
    with pytest.raises(promotions.PromoError, match=match):
        promotions.check(promo(**kw), 31_000_000, used_by_customer=used)


def test_check_accepts_and_describe():
    promotions.check(promo(usage_limit=10, used_count=9, per_customer_limit=None), 31_000_000, used_by_customer=5)
    text = promotions.describe(promo(discount_type='percent', value=5, max_discount=2_000_000, scope='brand',
                                     scope_value='Samsung', min_order=10_000_000))
    assert text.startswith('Giảm cuối năm: giảm 5% (tối đa 2.000.000đ) cho sản phẩm Samsung, đơn từ 10.000.000đ')
    assert text.endswith('(mã GIAM500K)')
    assert 'tự động áp dụng' in promotions.describe(promo(code=None, scope='category', scope_value='accessory'))
    assert 'cho phụ kiện' in promotions.describe(promo(scope='category', scope_value='accessory'))


def test_create_order_with_code_records_usage(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, promos=[promo(discount_type='percent', value=10,
                                                                    scope='category', scope_value='accessory')])
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    order_id = orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', promo_code='giam500k')
    subtotal = 29990001 + 2 * 1089720
    discount = 2 * 1089720 // 10                                        # 10% phần phụ kiện
    *_, sub, disc, promo_id, total = conn.find('INSERT INTO orders')[0]
    assert (sub, disc, promo_id, total) == (subtotal, discount, 'KM1', subtotal - discount)
    assert conn.find('INSERT INTO promotion_usages')[0][:2] == ('KM1', order_id)
    assert conn.find('UPDATE promotions SET used_count') == [('KM1',)]
    assert any('FOR UPDATE' in s for s, _ in conn.sql)                 # khóa dòng khuyến mãi khi đặt hàng


def test_create_order_bad_code_writes_no_order(monkeypatch):
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, promos=[promo()], used=1)
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    with pytest.raises(promotions.PromoError, match='đã dùng hết số lần'):
        orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', promo_code='GIAM500K')
    with pytest.raises(promotions.PromoError, match='KHONGCO không tồn tại'):
        orders.create_order(CART, 'A B', '0912345678', ADDRESS, 'delivery', 'cod', promo_code='khongco')
    assert not conn.find('INSERT INTO orders')


def test_auto_promotion_picks_biggest_discount(monkeypatch):
    autos = [promo(promo_id='A1', code=None, value=200_000),
             promo(promo_id='A2', code=None, discount_type='percent', value=5),             # 5% > 200k
             promo(promo_id='A3', code=None, value=9_000_000, min_order=99_000_000)]          # không đủ điều kiện
    conn = FakeConn({'S26-256-DEN': 3, 'SAC-60W': 5}, promos=autos)
    monkeypatch.setattr(orders, '_connect', lambda: conn)
    price = orders.quote(CART)
    assert price['promo']['promo_id'] == 'A2' and price['discount'] == price['subtotal'] * 5 // 100
    assert price['total'] == price['subtotal'] - price['discount']
    assert orders.quote(CART, None)['promo']['promo_id'] == 'A2'
    conn.promos = []
    assert orders.quote(CART) == {'subtotal': price['subtotal'], 'discount': 0, 'total': price['subtotal'], 'promo': None}


def test_summary_and_view_show_discount():
    order = {'order_id': 'DH1', 'name': 'A', 'phone': '0912345678', 'email': None, 'address': ADDRESS,
             'fulfillment_type': 'delivery', 'payment': 'cod', 'subtotal': 31_000_000, 'discount_amount': 500_000,
             'total': 30_500_000, 'promo_code': 'GIAM500K', 'promo_description': 'Giảm cuối năm',
             'items': [{'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen', 'unit_price': 31_000_000, 'qty': 1}]}
    lines = orders.summary(order).splitlines()
    assert lines[-4:-1] == ['Tạm tính: 31.000.000đ', 'Giảm giá (GIAM500K): -500.000đ', 'Tổng cộng: **30.500.000đ**']
    card = orders.view(order)
    assert (card['subtotal'], card['discount'], card['promo'], card['total']) == (31_000_000, 500_000, 'GIAM500K', 30_500_000)
    old = {**order, 'subtotal': None, 'discount_amount': 0}           # đơn trước khi có khuyến mãi
    assert 'Tạm tính' not in orders.summary(old) and orders.view(old)['promo'] is None


def test_describe_prints_end_date_in_vietnam_time():
    # Hết hạn 00:00 ngày 01/12 giờ VN = 17:00 ngày 30/11 UTC: phải in 01/12 dù máy chủ chạy UTC
    ends = datetime(2026, 11, 30, 17, 0, tzinfo=timezone.utc)
    assert 'đến hết 01/12/2026' in promotions.describe(promo(ends_at=ends))
