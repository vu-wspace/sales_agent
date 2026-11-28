# Chương trình mẫu: một kênh bán hàng (vd app, Messenger) đi qua be (cách A): be lo đơn hàng, khuyến mãi, email.
# Chạy: pip install httpx  rồi  python tich_hop_be.py
import sys
import time

import httpx

BE_URL = 'http://localhost:8002'      # địa chỉ be
ADAPTER_KEY = ''                      # key trong BE_ADAPTER_KEYS của be; be chưa bật key adapter thì để trống

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
be = httpx.Client(base_url=BE_URL, timeout=200,
                  headers={'Authorization': f'Bearer {ADAPTER_KEY}'} if ADAPTER_KEY else {})
sessions: dict[str, dict] = {}        # id người dùng -> {thread_id, customer_token, seen}; app thật lưu vào database


def call(user_id: str, path: str, body: dict) -> dict:
    # X-End-User: be giới hạn tốc độ theo từng người dùng (chỉ có tác dụng khi gửi kèm key adapter)
    for _ in range(5):
        r = be.post(path, json=body, headers={'X-End-User': user_id})
        if not (r.status_code == 409 and path == '/chat'):   # 409 ở /chat = tin trước chưa trả lời xong
            break
        time.sleep(3)
    if r.status_code in (409, 429, 503):  # be đã viết sẵn câu báo cho khách (mã hết hạn, nhắn nhanh quá...)
        raise ValueError(r.json()['detail'])
    r.raise_for_status()
    return r.json()


def thread_of(user_id: str) -> str:
    """Mỗi người dùng 1 cuộc trò chuyện do be cấp; im quá 15 phút thì mở cuộc mới."""
    s = sessions.get(user_id)
    if s is None or time.time() - s['seen'] > 15 * 60:
        res = call(user_id, '/session', {'end_thread_id': s and s['thread_id'],
                                         'customer_token': s and s['customer_token']})
        s = sessions[user_id] = {'thread_id': res['thread_id'], 'customer_token': res['customer_token']}
    s['seen'] = time.time()
    return s['thread_id']


def checkout(user_id: str, thread_id: str) -> None:
    q = call(user_id, '/order/quote', {'thread_id': thread_id})
    print(f"Shop: Tổng tiền {q['total']:,}đ. Nhập thông tin nhận hàng (để trống họ tên = hủy).".replace(',', '.'))
    name = input('Họ tên: ').strip()
    if not name:
        print('Shop:', call(user_id, '/order/cancel', {'thread_id': thread_id})['answer'])
        return
    order = {'thread_id': thread_id, 'name': name, 'phone': input('Số điện thoại: '),
             'address': input('Địa chỉ: '), 'promo_code': input('Mã khuyến mãi (có thể bỏ trống): ') or None}
    res = call(user_id, '/order', order)          # be tạo đơn từ giỏ hàng trên agent, gửi email xác nhận
    sessions[user_id]['customer_token'] = res['customer_token']   # lần sau be chào khách theo tên
    print('Shop:', res['answer'])


def main() -> None:
    user_id = 'khach-thu'
    print('Gõ tin nhắn, "thoát" để dừng.')
    while (text := input('Bạn: ').strip()) != 'thoát':
        if not text:
            continue
        try:
            thread_id = thread_of(user_id)
            res = call(user_id, '/chat', {'message': text, 'thread_id': thread_id})
            print('Shop:', res['answer'])
            if res.get('order_form'):
                checkout(user_id, thread_id)
        except ValueError as e:
            print('Shop:', e)


if __name__ == '__main__':
    main()
