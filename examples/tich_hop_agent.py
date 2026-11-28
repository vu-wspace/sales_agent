# Chương trình mẫu: một "hệ thống khác" chat với agent qua API /v1 (cách B).
# Chạy: pip install httpx  rồi  python tich_hop_agent.py
import sys
import time

import httpx

AGENT_URL = 'http://localhost:8001/v1'   # địa chỉ agent (thật: https://agent.cuahang.vn/v1)
AGENT_KEY = ''                           # key được cấp trong AGENT_CLIENT_KEYS; agent chưa bật key thì để trống

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
agent = httpx.Client(base_url=AGENT_URL, timeout=180,
                     headers={'Authorization': f'Bearer {AGENT_KEY}'} if AGENT_KEY else {})


def chat(user_id: str, text: str) -> dict:
    """Gửi 1 tin của khách cho agent, trả về câu trả lời và trạng thái."""
    for _ in range(5):
        r = agent.post('/chat', json={'message': text, 'thread_id': user_id})
        if r.status_code != 409:          # 409 = agent còn đang trả lời tin trước của khách này
            break
        time.sleep(3)                     # chờ rồi gửi lại
    if r.status_code in (429, 502, 503):  # quá giới hạn hoặc agent tạm lỗi
        return {'answer': 'Hệ thống đang bận, bạn nhắn lại sau ít phút nhé.', 'order_form': False, 'handoff': False}
    r.raise_for_status()                  # 401 sai key, 422 dữ liệu sai: lỗi của chương trình, dừng lại để sửa
    return r.json()


def place_order(user_id: str, cart: list[dict]) -> None:
    """Agent đã chốt giỏ: hỏi thông tin nhận hàng, tạo đơn trong hệ thống CỦA BẠN, rồi báo lại agent."""
    print('--- Đặt hàng (thông tin này không gửi cho agent) ---')
    name = input('Họ tên: ')
    phone = input('Số điện thoại: ')
    address = input('Địa chỉ: ')
    total = sum(item['price'] * item['quantity'] for item in cart)
    order_id = 'DH-THU-001'               # thay bằng lệnh tạo đơn thật trong hệ thống của bạn
    money = f'{total:,}'.replace(',', '.')   # 28999000 -> 28.999.000
    answer = f'Đã tạo đơn {order_id} cho {name}, tổng {money}đ. Cảm ơn bạn!'
    # Báo agent: ghi đơn vào lịch sử, làm trống giỏ để lần sau mua không bị cộng dồn
    agent.post(f'/threads/{user_id}/order-placed', json={'order_id': order_id, 'answer': answer}).raise_for_status()
    print('Shop:', answer)


def main() -> None:
    user_id = 'khach-thu'                 # id khách trong hệ thống của bạn (chữ, số, _ và -)
    print('Gõ tin nhắn, "thoát" để dừng.')
    while (text := input('Bạn: ').strip()) != 'thoát':
        if not text:
            continue
        res = chat(user_id, text)
        print('Shop:', res['answer'])
        if res.get('handoff'):
            print('--- Khách cần người thật: chuyển cuộc chat cho nhân viên ---')
        if res.get('order_form'):
            place_order(user_id, res['cart'])
    agent.delete(f'/threads/{user_id}')   # xóa hội thoại thử


if __name__ == '__main__':
    main()
