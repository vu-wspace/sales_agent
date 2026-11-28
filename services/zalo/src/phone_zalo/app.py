# Adapter Zalo OA <-> be. Khách nhắn Zalo OA -> Zalo gọi POST /webhook -> adapter gọi be (/session, /chat, /order...)
# bằng key adapter -> gửi câu trả lời lại qua Zalo OA API. Không gọi thẳng agent: đặt hàng, khuyến mãi, email ở be.
#
# Đặt hàng trong Zalo không có form: agent báo order_form -> adapter hỏi họ tên, số điện thoại, địa chỉ (mỗi ý 1 dòng)
# -> gọi be /order. Tin chứa thông tin nhận hàng không gửi sang agent/LLM (như form trên web).
import json
import logging
import re
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from phone_zalo import config
from phone_zalo.store import Store
from phone_zalo.zalo import Zalo, verify_signature

log = logging.getLogger('uvicorn.error')

SORRY = 'Dạ, hệ thống đang gặp sự cố, anh/chị nhắn lại sau ít phút giúp em nha.'
ONLY_TEXT = 'Dạ, em hiện chỉ đọc được tin nhắn chữ, anh/chị nhắn giúp em bằng chữ nha.'
CHECKOUT_HELP = ('Để đặt hàng, anh/chị nhắn giúp em 1 tin gồm 3 dòng:\n'
                 'Họ tên\nSố điện thoại\nĐịa chỉ nhận hàng\n'
                 '(có mã khuyến mãi thì thêm dòng "Mã: <mã>"). Giao tận nơi, thanh toán khi nhận hàng.\n'
                 'Nhắn "hủy" nếu anh/chị chưa muốn đặt.')
CANCEL_WORDS = {'huy', 'hủy', 'huỷ', 'hủy đặt hàng', 'huỷ đặt hàng', 'huy dat hang', 'không đặt nữa'}
PHONE = r'^(0|\+84)\d{9}$'
BUSY_RETRIES, BUSY_WAIT_S = 5, 3


class BeError(Exception):
    def __init__(self, status: int, message: str, detail=None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail


class Be:
    """Gọi be bằng key adapter; X-End-User để be giới hạn tốc độ theo từng người dùng Zalo."""

    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(base_url=config.BE_URL, timeout=200,
                                         headers={'Authorization': f'Bearer {config.BE_ADAPTER_KEY}'})

    def post(self, user_id: str, path: str, body: dict) -> dict:
        body = {k: v for k, v in body.items() if v is not None}
        for attempt in range(BUSY_RETRIES + 1):
            try:
                r = self.http.post(path, json=body, headers={'X-End-User': user_id})
            except httpx.TransportError as e:
                log.warning('Không gọi được be %s (%s)', path, type(e).__name__)
                raise BeError(503, SORRY)
            # 409 ở /chat: lượt trước của khách chưa xong (vd tin trước hết thời gian chờ) -> chờ rồi gửi lại
            if r.status_code == 409 and path == '/chat' and attempt < BUSY_RETRIES:
                time.sleep(BUSY_WAIT_S)
                continue
            break
        if r.is_success:
            return r.json()
        try:
            detail = r.json().get('detail')
        except ValueError:
            detail = None
        # 409 (mã khuyến mãi, hết hàng...), 429 (nhắn quá nhanh), 503: be đã viết sẵn câu cho khách
        message = detail if isinstance(detail, str) and r.status_code in (409, 429, 503) else SORRY
        if r.status_code >= 500 or r.status_code in (401, 400):
            log.error('be trả %s cho %s: %s', r.status_code, path, detail)
        raise BeError(r.status_code, message, detail)


def money(v: int) -> str:
    return f'{v:,}'.replace(',', '.') + 'đ'


def parse_checkout(text: str) -> dict | str:
    """3 dòng họ tên / số điện thoại / địa chỉ (+ dòng "Mã: ..." tùy chọn) -> dữ liệu cho be /order, sai thì câu hướng dẫn."""
    lines, promo = [], None
    for line in (l.strip() for l in text.splitlines() if l.strip()):
        if m := re.match(r'(?i)^(mã|ma)\b[^:：]*[:：]\s*(\S+)$', line):
            promo = m.group(2)
        else:
            lines.append(line)
    if len(lines) != 3:
        return 'Dạ, em chưa đọc được đủ thông tin. ' + CHECKOUT_HELP
    name, phone, address = lines
    phone = ''.join(c for c in phone if c not in ' .-')
    if not re.fullmatch(PHONE, phone):
        return 'Dạ, số điện thoại ở dòng 2 chưa đúng (10 số, bắt đầu bằng 0). ' + CHECKOUT_HELP
    if len(address) < 10:
        return 'Dạ, địa chỉ ở dòng 3 hơi ngắn, anh/chị ghi rõ số nhà, đường, phường/xã, tỉnh/thành giúp em nha.'
    return {'name': name, 'phone': phone, 'address': address, 'promo_code': promo}


class Adapter:
    def __init__(self, store: Store, zalo: Zalo, be: Be):
        self.store, self.zalo, self.be = store, zalo, be
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)  # mỗi khách xử lý lần lượt

    def handle(self, user_id: str, text: str) -> None:
        with self._locks[user_id]:
            user = self.store.user(user_id)
            now = time.time()
            if user['checkout'] is not None and now - user['seen_at'] > config.SESSION_TTL_S:
                user['checkout'] = None   # giỏ trên agent đã hết hạn cùng cuộc trò chuyện: quay lại tư vấn
            try:
                if user['handoff_until'] and now < user['handoff_until']:
                    reply = None   # nhân viên đang trả lời trong Zalo OA, bot im
                elif user['checkout'] is not None:
                    reply = self._checkout(user, text)
                else:
                    reply = self._chat(user, text, now)
            except BeError as e:
                reply = e.message
            except Exception:
                log.exception('Lỗi khi xử lý tin Zalo')
                reply = SORRY
            user['seen_at'] = now
            self.store.save_user(user)
        if reply:
            self.send(user_id, reply)

    def send(self, user_id: str, text: str) -> None:
        try:
            self.zalo.send_text(user_id, text)
        except Exception:
            log.exception('Không gửi được tin Zalo')

    def _chat(self, user: dict, text: str, now: float) -> str:
        uid, greeting = user['user_id'], ''
        if not user['thread_id'] or now - user['seen_at'] > config.SESSION_TTL_S:
            # Phiên mới (lần đầu hoặc im quá 15 phút): thread_id do be cấp (uuid), khách quen được chào theo tên
            s = self.be.post(uid, '/session', {'end_thread_id': user['thread_id'],
                                               'customer_token': user['customer_token']})
            user['thread_id'], user['customer_token'] = s['thread_id'], s['customer_token']
            greeting = s['greeting'] + '\n\n' if s['returning'] else ''
        res = self.be.post(uid, '/chat', {'message': text[:1000], 'thread_id': user['thread_id']})
        reply = greeting + res['answer']
        if res.get('handoff'):
            user['handoff_until'] = now + config.HANDOFF_PAUSE_S
            log.info('Chuyển nhân viên: khách Zalo %s', uid)
        elif res.get('order_form'):
            user['checkout'] = {}
            reply += '\n\n' + self._quote_text(user) + CHECKOUT_HELP
        return reply

    def _quote_text(self, user: dict) -> str:
        try:
            q = self.be.post(user['user_id'], '/order/quote', {'thread_id': user['thread_id']})
        except BeError:
            return ''
        if q['discount']:
            return (f"Tạm tính {money(q['subtotal'])}, giảm {money(q['discount'])} ({q['promo']}), "
                    f"tổng {money(q['total'])}.\n")
        return f"Tổng tiền {money(q['total'])}.\n"

    def _checkout(self, user: dict, text: str) -> str:
        uid = user['user_id']
        if text.strip().lower().rstrip('.!') in CANCEL_WORDS:
            user['checkout'] = None
            return self.be.post(uid, '/order/cancel', {'thread_id': user['thread_id']})['answer']
        info = parse_checkout(text)
        if isinstance(info, str):
            return info
        try:
            res = self.be.post(uid, '/order', {'thread_id': user['thread_id'], **info})
        except BeError as e:
            if e.status == 422:
                return 'Dạ, thông tin chưa hợp lệ. ' + CHECKOUT_HELP
            if e.status == 409:   # mã khuyến mãi không dùng được, hết hàng...: sửa rồi gửi lại
                return f'{e.message}\nAnh/chị sửa lại thông tin (hoặc bỏ dòng mã) rồi gửi lại, hoặc nhắn "hủy".'
            raise
        user['checkout'] = None
        user['customer_token'] = res.get('customer_token') or user['customer_token']
        return res['answer']


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (config.ZALO_APP_ID and config.ZALO_OA_SECRET and config.BE_ADAPTER_KEY):
        log.warning('Thiếu ZALO_APP_ID / ZALO_OA_SECRET / ZALO_BE_ADAPTER_KEY: webhook sẽ từ chối mọi sự kiện')
    if not hasattr(app.state, 'adapter'):   # test gắn adapter giả trước
        store = Store(config.DB_PATH)
        app.state.adapter = Adapter(store, Zalo(store), Be())
    yield


app = FastAPI(title='Phone Shop Zalo adapter', version='1.0.0', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/webhook')
async def webhook(request: Request, background: BackgroundTasks):
    """Sự kiện Zalo OA. Trả 200 ngay, xử lý sau (Zalo chờ phản hồi ngắn); tin gửi lại (cùng msg_id) bỏ qua."""
    raw = await request.body()
    try:
        event = json.loads(raw)
    except ValueError:
        raise HTTPException(400, 'Body không phải JSON')
    if not verify_signature(raw, str(event.get('timestamp', '')), request.headers.get('x-zevent-signature', '')):
        raise HTTPException(401, 'Sai chữ ký Zalo')
    adapter: Adapter = request.app.state.adapter
    name = event.get('event_name', '')
    sender = str((event.get('sender') or {}).get('id') or '')
    message = event.get('message') or {}
    if not re.fullmatch(r'[\w-]{1,100}', sender) or not name.startswith('user_send_'):
        return {'ok': True}   # sự kiện khác (theo dõi OA, OA gửi tin...) không cần trả lời
    if not adapter.store.first_time(str(message.get('msg_id') or f"{sender}:{event.get('timestamp')}")):
        return {'ok': True}
    if name == 'user_send_text' and (message.get('text') or '').strip():
        background.add_task(adapter.handle, sender, message['text'])
    else:
        background.add_task(adapter.send, sender, ONLY_TEXT)
    return {'ok': True}
