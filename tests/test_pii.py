import logging

import pytest

from phone_sales import pii


@pytest.mark.parametrize('text, expected', [
    ('sđt mình 0912 345 678 nha', 'sđt mình [SỐ ĐIỆN THOẠI] nha'),
    ('gọi +84 912.345.678 giúp', 'gọi [SỐ ĐIỆN THOẠI] giúp'),
    ('số 0123456789', 'số [SỐ ĐIỆN THOẠI]'),                       # form nhận mọi số 0 + 9 chữ số
    ('bàn 02838123456', 'bàn [SỐ ĐIỆN THOẠI]'),                    # số cố định 11 chữ số
    ('mail Nguyen.Van-A+shop@gmail.com.vn nhé', 'mail [EMAIL] nhé'),
    # Không che nhầm giá, mã đơn, dung lượng, số ngắn
    ('giá 28.999.000đ, 1.089.720đ, 84.999.000đ', 'giá 28.999.000đ, 1.089.720đ, 84.999.000đ'),
    ('đơn DH3F9A1C2B, iPhone 17 256GB, 01234', 'đơn DH3F9A1C2B, iPhone 17 256GB, 01234'),
])
def test_redact(text, expected):
    assert pii.redact(text) == expected


def test_mask_for_display():
    assert pii.mask_phone('0912345678') == pii.mask_phone('+84 912 345 678') == '0912 *** 678'
    assert pii.mask_email('nguyenvana@gmail.com') == 'ng***@gmail.com'
    assert pii.mask_email('ab@x.vn') == 'a***@x.vn' and pii.mask_email(None) is None


def test_encrypt_roundtrip_and_blind_index():
    enc = pii.encrypt('0912345678')
    assert enc.startswith('enc:') and '0912345678' not in enc and enc != pii.encrypt('0912345678')  # mỗi lần khác
    assert pii.decrypt(enc) == '0912345678' and pii.decrypt('0912345678') == '0912345678'      # dữ liệu cũ
    assert pii.encrypt(None) is None and pii.decrypt(None) is None
    # HMAC cố định: tra cứu được, không phân biệt cách viết số / hoa thường email
    assert pii.phone_key('0912 345 678') == pii.phone_key('+84912345678') != pii.phone_key('0912345679')
    assert pii.email_key('A@B.vn') == pii.email_key(' a@b.vn ') and pii.email_key('') is None


def test_missing_or_wrong_key(monkeypatch):
    enc = pii.encrypt('a@b.vn')
    monkeypatch.setattr(pii.config, 'PII_KEY', '')
    with pytest.raises(pii.PiiKeyError, match='Chưa đặt PII_KEY'):
        pii.encrypt('x')
    monkeypatch.setattr(pii.config, 'PII_KEY', 'không-phải-khóa')
    with pytest.raises(pii.PiiKeyError, match='không hợp lệ'):
        pii.encrypt('x')
    from cryptography.fernet import Fernet
    monkeypatch.setattr(pii.config, 'PII_KEY', Fernet.generate_key().decode())
    with pytest.raises(pii.PiiKeyError, match='khác khóa'):
        pii.decrypt(enc)


def test_log_filter_redacts_message_args_and_traceback():
    record = logging.LogRecord('x', logging.ERROR, __file__, 1, 'Khách %s, email %s', ('0912345678', 'a@b.vn'), None)
    try:
        raise ValueError('Key (lower(email))=(a@b.vn) already exists, phone 0912 345 678')
    except ValueError:
        import sys
        record.exc_info = sys.exc_info()
    assert pii.RedactFilter().filter(record)
    text = logging.Formatter().format(record)
    assert 'Khách [SỐ ĐIỆN THOẠI], email [EMAIL]' in text and 'ValueError' in text
    assert '0912' not in text and 'a@b.vn' not in text


def test_install_log_filter_is_idempotent():
    handler = logging.StreamHandler()
    logging.getLogger('uvicorn.error').addHandler(handler)
    try:
        pii.install_log_filter()
        pii.install_log_filter()
        assert sum(isinstance(f, pii.RedactFilter) for f in handler.filters) == 1
    finally:
        logging.getLogger('uvicorn.error').removeHandler(handler)


def test_log_filter_keeps_args_shape_for_uvicorn_access():
    # uvicorn.access tách record.args theo vị trí (client, method, path, version, status)
    record = logging.LogRecord('uvicorn.access', logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ('127.0.0.1:5000', 'GET', '/?phone=0912345678', '1.1', 200), None)
    pii.RedactFilter().filter(record)
    client, method, path, version, status = record.args
    assert path == '/?phone=[SỐ ĐIỆN THOẠI]' and status == 200 and client == '127.0.0.1:5000'
