# Email xác nhận đơn hàng gửi cho khách sau khi đặt hàng (api.py: POST /order chạy send_order_email bằng
# BackgroundTasks, khách nhận kết quả ngay, email gửi sau). Đây là "xác nhận đơn hàng", không phải hóa đơn GTGT:
# hóa đơn điện tử phải phát hành qua nhà cung cấp dịch vụ hóa đơn (Nghị định 123/2020/NĐ-CP).
#
# Hạn chế: BackgroundTasks chạy trong tiến trình API, server khởi động lại đúng lúc đang gửi thì mất email và không tự
# gửi lại (đơn kẹt ở email_status = 'pending'). Làm thật nên dùng outbox (ghi yêu cầu gửi trong cùng transaction tạo
# đơn, worker riêng đọc bảng và gửi, có thử lại) hoặc hàng đợi (Celery, ARQ).
import asyncio
import logging
from email.utils import parseaddr
from pathlib import Path

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema, MessageType
from jinja2 import Environment, FileSystemLoader, select_autoescape

from phone_sales import config, orders

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / 'templates'

def vnd(n: int) -> str:
    return f'{n:,}'.replace(',', '.') + 'đ'

# autoescape: tên, địa chỉ do khách nhập -> không chèn được HTML vào email
_env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(['html']))
_env.filters['vnd'] = vnd

def shop_name() -> str:
    return parseaddr(config.MAIL_FROM)[0] or 'Cửa hàng'

def email_data(order: dict) -> dict:
    """Dữ liệu cho template, lấy từ đơn đã lưu (orders.get_order) qua orders.view: cùng tên món, giảm giá, tổng tiền
    với thẻ xác nhận trên khung chat."""
    v = orders.view(order)
    items = [{'name': ' '.join(filter(None, [i['name'], i['variant'].replace(' · ', ' ')])), 'qty': i['qty'],
              'unit_price': i['unit_price']} for i in v['items']]
    return {**v, 'items': items, 'address': order.get('address') if order['fulfillment_type'] == 'delivery' else None}

def subject(order: dict) -> str:
    return f"Xác nhận đơn hàng #{order['order_id']} - {shop_name()}"

def render_html(data: dict) -> str:
    return _env.get_template('order_confirmation.html').render(order=data, shop=shop_name())

def render_text(data: dict) -> str:
    """Bản chữ thuần cho trình đọc email không hiển thị HTML."""
    lines = [f"Xác nhận đơn hàng #{data['order_id']}", '', f"Chào {data['name']},",
             f'Cảm ơn anh/chị đã đặt hàng tại {shop_name()}.', '']
    lines += [f"- {i['name']} x{i['qty']}: {vnd(i['unit_price'] * i['qty'])}" for i in data['items']]
    lines.append('')
    if data['discount']:
        promo = f" ({data['promo']})" if data['promo'] else ''
        lines += [f"Tạm tính: {vnd(data['subtotal'])}", f"Giảm giá{promo}: -{vnd(data['discount'])}"]
    lines += [f"Tổng cộng: {vnd(data['total'])}", '', f"Nhận hàng: {data['fulfillment']}"]
    if data['address']:
        lines.append(f"Địa chỉ: {data['address']}")
    lines += [f"Thanh toán: {data['payment']}", '', data['note']]
    return '\n'.join(lines)

def mail_client() -> FastMail:
    name, address = parseaddr(config.MAIL_FROM)
    return FastMail(ConnectionConfig(
        MAIL_SERVER=config.SMTP_HOST, MAIL_PORT=config.SMTP_PORT, MAIL_FROM=address, MAIL_FROM_NAME=name or None,
        MAIL_USERNAME=config.SMTP_USER, MAIL_PASSWORD=config.SMTP_PASSWORD,
        USE_CREDENTIALS=bool(config.SMTP_USER), MAIL_STARTTLS=config.SMTP_STARTTLS, MAIL_SSL_TLS=config.SMTP_SSL,
        VALIDATE_CERTS=True, TIMEOUT=15,
    ))

async def send_order_confirmation(order: dict) -> None:
    data = email_data(order)
    message = MessageSchema(
        recipients=[order['email']], subject=subject(order),
        body=render_html(data), alternative_body=render_text(data),
        subtype=MessageType.html, multipart_subtype='alternative',
    )
    await mail_client().send_message(message)

async def send_order_email(order_id: str) -> str:
    """Việc nền sau khi tạo đơn: đọc lại đơn từ database (email luôn khớp đơn đã lưu, không dùng dữ liệu form),
    gửi email rồi ghi email_status. Không ném lỗi ra ngoài: đơn đã tạo xong, gửi email hỏng không ảnh hưởng khách."""
    try:
        order = await asyncio.to_thread(orders.get_order, order_id)
        if not order.get('email') or not config.SMTP_HOST:
            status = 'skipped'
        else:
            await send_order_confirmation(order)
            status = 'sent'
    except Exception:
        # Chỉ ghi mã đơn, không ghi email khách (pii.RedactFilter vẫn che nếu lỗi SMTP có kèm địa chỉ)
        log.exception('Gửi email xác nhận thất bại cho đơn %s', order_id)
        status = 'failed'
    try:
        await asyncio.to_thread(orders.set_email_status, order_id, status)
    except Exception:
        log.exception('Không ghi được email_status = %s cho đơn %s', status, order_id)
    return status
