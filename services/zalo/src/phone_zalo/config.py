import os

# be: adapter gọi /session, /chat, /order/quote, /order, /order/cancel bằng key adapter (BE_ADAPTER_KEYS bên be,
# mục "zalo:<key>"), kèm X-End-User = id người dùng Zalo để be giới hạn tốc độ theo từng người.
BE_URL = os.getenv('BE_URL', 'http://localhost:8002')
BE_ADAPTER_KEY = os.getenv('ZALO_BE_ADAPTER_KEY', '')

# Zalo Official Account (developers.zalo.me -> ứng dụng -> Official Account):
#   ZALO_APP_ID, ZALO_APP_SECRET: id, secret key của ứng dụng (dùng để lấy access token mới từ refresh token)
#   ZALO_OA_SECRET: "OA Secret Key" trong mục Webhook (dùng để kiểm tra chữ ký sự kiện Zalo gửi tới)
#   ZALO_REFRESH_TOKEN: refresh token ban đầu (lấy 1 lần qua OAuth); sau đó adapter tự đổi và lưu bản mới vào DB_PATH
ZALO_APP_ID = os.getenv('ZALO_APP_ID', '')
ZALO_APP_SECRET = os.getenv('ZALO_APP_SECRET', '')
ZALO_OA_SECRET = os.getenv('ZALO_OA_SECRET', '')
ZALO_REFRESH_TOKEN = os.getenv('ZALO_REFRESH_TOKEN', '')
ZALO_API = os.getenv('ZALO_API', 'https://openapi.zalo.me')
ZALO_OAUTH = os.getenv('ZALO_OAUTH', 'https://oauth.zaloapp.com')

# SQLite của adapter: người dùng Zalo -> thread_id của be, mã khách quen, trạng thái đặt hàng, token Zalo.
# Chỉ adapter đọc/ghi file này (1 process). Chạy nhiều bản thì chuyển sang Postgres/Redis.
DB_PATH = os.getenv('ZALO_DB_PATH', 'zalo.sqlite3')

SESSION_TTL_S = 15 * 60          # như agent: khách im 15 phút thì mở cuộc trò chuyện mới
HANDOFF_PAUSE_S = 60 * 60        # agent chuyển nhân viên: adapter ngừng tự trả lời 1 giờ để nhân viên nhắn trong OA
