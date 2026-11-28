import os
from pathlib import Path

# Thư mục gốc repo (data_schema/...). Tính từ vị trí file: services/be/src/phone_api/config.py
ROOT = Path(__file__).resolve().parents[4]
POLICY_DIR = ROOT / 'data_schema' / 'policies'   # chính sách mẫu, nạp vào bảng policies khi bảng rỗng

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/phone')

# Email xác nhận đơn hàng (mailer.py). Chạy trên máy dùng Mailpit trong docker-compose (SMTP localhost:1025, xem email
# tại http://localhost:8025, không gửi ra Internet). Chạy thật thì đổi sang Gmail (App Password), SES, SendGrid...
# SMTP_HOST để trống = tắt gửi email (đơn ghi email_status = 'skipped').
SMTP_HOST = os.getenv('SMTP_HOST', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '1025'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
SMTP_STARTTLS = os.getenv('SMTP_STARTTLS', 'false').lower() == 'true'
SMTP_SSL = os.getenv('SMTP_SSL', 'false').lower() == 'true'          # cổng 465
MAIL_FROM = os.getenv('MAIL_FROM', 'PhoneShop Demo <no-reply@phoneshop.example>')

# Agent chạy thành service riêng (services/agent). be gọi agent qua AGENT_URL, gửi AGENT_API_KEY nếu có.
AGENT_URL = os.getenv('AGENT_URL', 'http://localhost:8001')
AGENT_API_KEY = os.getenv('AGENT_API_KEY', '')
# Key agent dùng để đọc catalog, khuyến mãi, chính sách (để trống: không kiểm tra, chỉ dùng trong mạng nội bộ)
CATALOG_API_KEY = os.getenv('CATALOG_API_KEY', '')

# Key cho /admin/policies (header "Authorization: Bearer <key>"). Để trống thì tắt các endpoint quản trị
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', '')
CHAT_RATE_PER_MIN = int(os.getenv('CHAT_RATE_PER_MIN', '20'))  # số lượt chat, mở phiên, đặt hàng tối đa mỗi IP mỗi phút (0: tắt)
