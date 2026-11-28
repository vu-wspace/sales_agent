# Adapter Zalo OA: webhook có chữ ký, chat qua be, đặt hàng trong tin nhắn (thông tin nhận hàng không tới agent).
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from phone_zalo import app as zapp
from phone_zalo import config, zalo
from phone_zalo.store import Store


class FakeBe:
    def __init__(self, **responses):
        self.calls = []
        self.responses = responses   # path -> dict hoặc BeError

    def post(self, user_id, path, body):
        self.calls.append((user_id, path, {k: v for k, v in body.items() if v is not None}))
        res = self.responses[path]
        if isinstance(res, Exception):
            raise res
        return res(body) if callable(res) else res


class FakeZalo:
    def __init__(self):
        self.sent = []

    def send_text(self, user_id, text):
        self.sent.append((user_id, text))


SESSION = {'thread_id': 'th-1', 'greeting': 'Chào', 'returning': False, 'customer_token': None}
ANSWER = {'answer': 'iPhone 17 giá 24.990.000đ', 'handoff': False, 'order_form': False}


def adapter(**responses):
    be = FakeBe(**{'/session': SESSION, '/chat': ANSWER, **responses})
    return zapp.Adapter(Store(':memory:'), FakeZalo(), be), be


def paths(be):
    return [path for _, path, _ in be.calls]


def test_chat_opens_session_once_then_reuses_thread(monkeypatch):
    a, be = adapter()
    a.handle('u1', 'giá iphone 17')
    a.handle('u1', 'còn màu đen không')
    assert paths(be) == ['/session', '/chat', '/chat']
    assert all(c[0] == 'u1' for c in be.calls)                       # X-End-User = id người dùng Zalo
    assert be.calls[2][2] == {'message': 'còn màu đen không', 'thread_id': 'th-1'}
    assert a.zalo.sent[0] == ('u1', 'iPhone 17 giá 24.990.000đ')

    # Im quá 15 phút: mở phiên mới, kết thúc phiên cũ, gửi mã khách quen
    user = a.store.user('u1')
    a.store.save_user({**user, 'seen_at': user['seen_at'] - config.SESSION_TTL_S - 1, 'customer_token': 'KH1.sig'})
    be.responses['/session'] = {**SESSION, 'thread_id': 'th-2', 'returning': True, 'greeting': 'Chào anh A',
                                'customer_token': 'KH1.sig'}
    a.handle('u1', 'chào shop')
    assert be.calls[-2] == ('u1', '/session', {'end_thread_id': 'th-1', 'customer_token': 'KH1.sig'})
    assert a.zalo.sent[-1][1].startswith('Chào anh A\n\n')            # khách quen: chào theo tên
    assert a.store.user('u1')['thread_id'] == 'th-2'


def test_order_inside_zalo_without_sending_pii_to_agent():
    orders = []
    a, be = adapter(**{
        '/chat': {**ANSWER, 'answer': 'Em đã thêm vào giỏ.', 'order_form': True},
        '/order/quote': {'subtotal': 25_000_000, 'discount': 500_000, 'total': 24_500_000, 'promo': 'GIAM500K'},
        '/order': lambda body: orders.append(body) or {'answer': 'Đơn DH1 đã đặt', 'customer_token': 'KH9.sig'},
    })
    a.handle('u1', 'chốt iphone 17 256gb đen')
    assert 'tổng 24.500.000đ' in a.zalo.sent[-1][1] and 'Họ tên' in a.zalo.sent[-1][1]

    a.handle('u1', 'Nguyễn Văn A\n0912 345 678')                     # thiếu địa chỉ
    assert 'chưa đọc được đủ' in a.zalo.sent[-1][1]
    a.handle('u1', 'Nguyễn Văn A\n09123\n12 Lê Lợi, Q1, TP.HCM')       # sai số điện thoại
    assert 'số điện thoại' in a.zalo.sent[-1][1] and orders == []

    a.handle('u1', 'Nguyễn Văn A\n0912.345.678\n12 Lê Lợi, Quận 1, TP.HCM\nMã: giam500k')
    assert orders == [{'thread_id': 'th-1', 'name': 'Nguyễn Văn A', 'phone': '0912345678',
                       'address': '12 Lê Lợi, Quận 1, TP.HCM', 'promo_code': 'giam500k'}]
    assert a.zalo.sent[-1][1] == 'Đơn DH1 đã đặt'
    # Tin chứa họ tên, số điện thoại, địa chỉ không bao giờ đi qua /chat (agent, LLM)
    assert not any('0912' in json.dumps(body) for _, path, body in be.calls if path == '/chat')
    user = a.store.user('u1')
    assert user['checkout'] is None and user['customer_token'] == 'KH9.sig'


def test_order_errors_and_cancel_keep_customer_in_control():
    a, be = adapter(**{'/chat': {**ANSWER, 'order_form': True},
                       '/order/quote': {'subtotal': 1, 'discount': 0, 'total': 1, 'promo': None},
                       '/order': zapp.BeError(409, 'Mã khuyến mãi này đã hết hạn.'),
                       '/order/cancel': {'answer': 'Dạ em đã hủy đặt hàng.'}})
    a.handle('u1', 'chốt')
    a.handle('u1', 'Trần B\n0912345678\n34 Hai Bà Trưng, Hà Nội\nMã: CU')
    assert 'đã hết hạn' in a.zalo.sent[-1][1] and a.store.user('u1')['checkout'] == {}   # vẫn chờ khách sửa
    a.handle('u1', 'Hủy')
    assert a.zalo.sent[-1][1] == 'Dạ em đã hủy đặt hàng.' and a.store.user('u1')['checkout'] is None
    a.handle('u1', 'xem ốp lưng')
    assert paths(be)[-2:] == ['/chat', '/order/quote']                # hết chế độ đặt hàng: tin lại tới agent


def test_handoff_pauses_bot_and_be_errors_reach_customer():
    a, be = adapter(**{'/chat': {**ANSWER, 'answer': 'Em chuyển nhân viên', 'handoff': True}})
    a.handle('u1', 'cho gặp người thật')
    a.handle('u1', 'alo')
    assert paths(be).count('/chat') == 1 and len(a.zalo.sent) == 1   # nhân viên trả lời trong OA, bot im

    a, be = adapter(**{'/chat': zapp.BeError(429, 'Dạ, anh/chị nhắn hơi nhanh…')})
    a.handle('u2', 'hi')
    assert a.zalo.sent == [('u2', 'Dạ, anh/chị nhắn hơi nhanh…')]


def test_be_client_retries_busy_thread_and_maps_errors(monkeypatch):
    monkeypatch.setattr(zapp, 'BUSY_WAIT_S', 0)
    statuses = [409, 409, 200]

    def send(request):
        assert request.headers['x-end-user'] == 'u1' and request.headers['authorization'] == 'Bearer ak'
        code = statuses.pop(0)
        return httpx.Response(code, json={'detail': 'đang bận'} if code == 409 else {'answer': 'ok'})

    be = zapp.Be(httpx.Client(base_url='http://be', headers={'Authorization': 'Bearer ak'},
                              transport=httpx.MockTransport(send)))
    assert be.post('u1', '/chat', {'message': 'hi', 'thread_id': None}) == {'answer': 'ok'}

    def fail(request):
        return httpx.Response(500, json={'detail': 'Traceback nội bộ'})

    be = zapp.Be(httpx.Client(base_url='http://be', transport=httpx.MockTransport(fail)))
    with pytest.raises(zapp.BeError) as e:
        be.post('u1', '/chat', {'message': 'hi'})
    assert e.value.message == zapp.SORRY                               # lỗi nội bộ không lộ cho khách


def signed(body: dict) -> tuple[bytes, dict]:
    raw = json.dumps(body).encode()
    mac = hashlib.sha256((config.ZALO_APP_ID + raw.decode() + str(body['timestamp']) + config.ZALO_OA_SECRET)
                         .encode()).hexdigest()
    return raw, {'X-ZEvent-Signature': f'mac={mac}', 'Content-Type': 'application/json'}


def test_webhook_checks_signature_dedupes_and_ignores_other_events(monkeypatch):
    monkeypatch.setattr(config, 'ZALO_APP_ID', 'app1')
    monkeypatch.setattr(config, 'ZALO_OA_SECRET', 'oa-secret')
    a, be = adapter()
    zapp.app.state.adapter = a
    api = TestClient(zapp.app)
    event = {'app_id': 'app1', 'event_name': 'user_send_text', 'timestamp': '1764300000000',
             'sender': {'id': '7890'}, 'recipient': {'id': 'oa1'}, 'message': {'msg_id': 'm1', 'text': 'giá s26'}}
    raw, headers = signed(event)
    assert api.post('/webhook', content=raw, headers={**headers, 'X-ZEvent-Signature': 'mac=sai'}).status_code == 401
    assert api.post('/webhook', content=raw, headers=headers).status_code == 200
    assert api.post('/webhook', content=raw, headers=headers).status_code == 200   # Zalo gửi lại: bỏ qua
    assert paths(be) == ['/session', '/chat'] and a.zalo.sent[0][0] == '7890'

    raw, headers = signed({**event, 'event_name': 'user_send_image', 'message': {'msg_id': 'm2'}})
    api.post('/webhook', content=raw, headers=headers)
    assert a.zalo.sent[-1] == ('7890', zapp.ONLY_TEXT)
    raw, headers = signed({**event, 'event_name': 'follow', 'message': {}})
    api.post('/webhook', content=raw, headers=headers)
    assert len(a.zalo.sent) == 2

    monkeypatch.setattr(config, 'ZALO_OA_SECRET', '')                  # chưa cấu hình: từ chối hết
    assert api.post('/webhook', content=raw, headers=headers).status_code == 401
    del zapp.app.state.adapter


def test_zalo_text_and_token_refresh(monkeypatch):
    assert zalo.plain_text('# Giá\n**iPhone 17**: 24.990.000đ\n- 256GB') == 'Giá\niPhone 17: 24.990.000đ\n- 256GB'
    long = '\n'.join(f'- dòng {i} ' + 'x' * 50 for i in range(100))
    parts = zalo.chunks(long)
    assert all(len(p) <= zalo.MAX_TEXT for p in parts) and '\n'.join(parts) == long

    monkeypatch.setattr(config, 'ZALO_REFRESH_TOKEN', 'rt-0')
    calls = []

    def send(request):
        calls.append(request.url.path)
        if request.url.path == '/v4/oa/access_token':
            assert b'refresh_token=rt-' in request.content
            return httpx.Response(200, json={'access_token': f'at-{len(calls)}', 'refresh_token': f'rt-{len(calls)}',
                                             'expires_in': '90000'})
        if request.headers['access_token'] == 'at-1':                 # token đầu bị Zalo báo hết hạn
            return httpx.Response(200, json={'error': -216, 'message': 'Access token is invalid'})
        return httpx.Response(200, json={'error': 0})

    store = Store(':memory:')
    z = zalo.Zalo(store, httpx.Client(transport=httpx.MockTransport(send)))
    z.send_text('u1', 'xin chào')
    assert calls == ['/v4/oa/access_token', '/v3.0/oa/message/cs', '/v4/oa/access_token', '/v3.0/oa/message/cs']
    assert store.get('refresh_token') == 'rt-3'                        # refresh token mới được lưu, bản cũ bỏ
