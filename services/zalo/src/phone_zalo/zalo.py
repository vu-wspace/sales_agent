# Gọi Zalo Official Account Open API: kiểm tra chữ ký webhook, gửi tin tư vấn, đổi access token.
# Theo tài liệu Zalo OA API v3 (developers.zalo.me): kiểm tra lại đường dẫn, tên trường khi cấu hình ứng dụng thật.
import hashlib
import hmac
import logging
import re
import threading
import time

import httpx

from phone_zalo import config
from phone_zalo.store import Store

log = logging.getLogger(__name__)
MAX_TEXT = 2000            # Zalo giới hạn độ dài 1 tin nhắn chữ
TOKEN_INVALID = (-216, -124)  # access token sai / hết hạn


def verify_signature(raw_body: bytes, timestamp: str, signature: str) -> bool:
    """Header X-ZEvent-Signature = "mac=" + sha256(app_id + body + timestamp + OA secret key)."""
    if not (config.ZALO_APP_ID and config.ZALO_OA_SECRET):
        return False   # chưa cấu hình: từ chối hết, không nhận sự kiện giả
    data = config.ZALO_APP_ID + raw_body.decode('utf-8') + str(timestamp) + config.ZALO_OA_SECRET
    expected = 'mac=' + hashlib.sha256(data.encode('utf-8')).hexdigest()
    return hmac.compare_digest(expected, signature or '')


def plain_text(markdown: str) -> str:
    """Zalo không hiện markdown: bỏ **đậm**, # tiêu đề, giữ nguyên gạch đầu dòng và xuống dòng."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', markdown)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    return text.strip()


def chunks(text: str, size: int = MAX_TEXT) -> list[str]:
    # Cắt ở xuống dòng gần nhất để không đứt giữa 1 dòng giá
    parts = []
    while len(text) > size:
        cut = text.rfind('\n', 0, size)
        cut = cut if cut > 0 else size
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip('\n')
    return parts + [text] if text else parts


class Zalo:
    def __init__(self, store: Store, http: httpx.Client | None = None):
        self.store = store
        self.http = http or httpx.Client(timeout=15)
        self.lock = threading.Lock()   # 1 lần đổi token tại 1 thời điểm: refresh token chỉ dùng được 1 lần

    def _refresh(self) -> str:
        refresh_token = self.store.get('refresh_token') or config.ZALO_REFRESH_TOKEN
        if not refresh_token:
            raise RuntimeError('Chưa có ZALO_REFRESH_TOKEN')
        r = self.http.post(f'{config.ZALO_OAUTH}/v4/oa/access_token', headers={'secret_key': config.ZALO_APP_SECRET},
                           data={'app_id': config.ZALO_APP_ID, 'refresh_token': refresh_token,
                                 'grant_type': 'refresh_token'})
        r.raise_for_status()
        body = r.json()
        if 'access_token' not in body:
            raise RuntimeError(f'Không đổi được access token Zalo: {body.get("error_name") or body.get("error")}')
        # Refresh token mới thay bản cũ (bản cũ hết dùng được): lưu ngay, mất là phải lấy lại qua OAuth
        self.store.set('refresh_token', body['refresh_token'])
        self.store.set('access_token', body['access_token'])
        self.store.set('access_expires', str(time.time() + int(body.get('expires_in', 3600)) - 300))
        log.info('Đã đổi access token Zalo')
        return body['access_token']

    def token(self, force: bool = False) -> str:
        with self.lock:
            token, expires = self.store.get('access_token'), float(self.store.get('access_expires') or 0)
            if force or not token or time.time() >= expires:
                token = self._refresh()
            return token

    def send_text(self, user_id: str, text: str) -> None:
        for part in chunks(plain_text(text)):
            self._send(user_id, part)

    def _send(self, user_id: str, text: str, retried: bool = False) -> None:
        r = self.http.post(f'{config.ZALO_API}/v3.0/oa/message/cs', headers={'access_token': self.token(force=retried)},
                           json={'recipient': {'user_id': user_id}, 'message': {'text': text}})
        r.raise_for_status()
        error = r.json().get('error', 0)
        if error in TOKEN_INVALID and not retried:
            return self._send(user_id, text, retried=True)
        if error:
            raise RuntimeError(f'Zalo không gửi được tin (error {error}: {r.json().get("message")})')
