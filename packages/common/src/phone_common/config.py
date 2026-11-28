import os

# Chỉ cấu hình của các tiện ích dùng chung (llm.py, pii.py). Cấu hình riêng của từng service nằm ở
# phone_agent/config.py và phone_api/config.py.

# Khóa mã hóa số điện thoại, email của khách trong database (Fernet, base64 32 byte). Tạo khóa:
#   uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Mất khóa là không đọc lại được thông tin khách đã lưu
PII_KEY = os.getenv('PII_KEY', '')

# 9Router (model chính của graph, xem llm.py). Chạy trên máy: http://localhost:20128/v1
ROUTER9_BASE_URL = os.getenv('ROUTER9_BASE_URL', 'http://localhost:20128/v1')
ROUTER9_API_KEY = os.getenv('ROUTER9_API_KEY', '')
ROUTER9_MODEL = os.getenv('ROUTER9_MODEL', 'agy/gemini-3.8-flash-medium')
