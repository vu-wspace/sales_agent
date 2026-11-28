import os
from pathlib import Path

# Thư mục gốc repo (data_schema/...). Tính từ vị trí file: services/agent/src/phone_agent/config.py
ROOT = Path(__file__).resolve().parents[4]
POLICY_DIR = ROOT / 'data_schema' / 'policies'   # chính sách mẫu (.md): chỉ dùng cho eval, test, chạy thử data_prep
CHUNK_MAX_CHARS = 900                            # model bkai cắt ở 256 token, chunk dài hơn sẽ mất phần cuối khi embed

# Vector DB
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
COLLECTION = 'phone_policies'
EMBEDDING_MODEL = 'bkai-foundation-models/vietnamese-bi-encoder'
VECTOR_SIZE = 768

# Lọc kết quả search_policy (điểm cosine, đo trên 19 câu policy của bộ eval 28/10/2026): câu đúng chủ đề có
# top 1 >= 0.30, câu lạc đề (bitcoin, thơ, thời tiết) 0.10-0.21. Đoạn kém đoạn tốt nhất quá MARGIN thì bỏ để
# tránh lan man. k=3, margin=0.08: vẫn ra đúng file 19/19, trung bình 2.2 đoạn / 1.3 file mỗi câu
# (margin 0.15 thì 2.4 đoạn / 1.3 file; k=4 thì kéo thêm file không liên quan).
POLICY_TOP_K = 3
POLICY_MIN_SCORE = 0.25
POLICY_SCORE_MARGIN = 0.08

# Agent service dùng bản nào: 'graph' (graph.py, có trạng thái hội thoại + giỏ hàng) hoặc 'agent' (agent.py)
AGENT_IMPL = os.getenv('AGENT_IMPL', 'graph').lower()

# be: agent đọc catalog, khuyến mãi, chính sách đã xuất bản qua HTTP (backend.py), gửi CATALOG_API_KEY.
BE_URL = os.getenv('BE_URL', 'http://localhost:8002')
CATALOG_API_KEY = os.getenv('CATALOG_API_KEY', '')
# AGENT_API_KEY: đặt thì mọi endpoint của agent (trừ /health) đòi header "Authorization: Bearer <key>" -
# bắt buộc khi mở agent cho hệ thống khác tích hợp; để trống chỉ dùng trong mạng nội bộ (docker compose).
AGENT_API_KEY = os.getenv('AGENT_API_KEY', '')

# Postgres riêng của agent (không dùng chung database của be): lịch sử hội thoại + giỏ hàng (checkpointer LangGraph),
# lượt đang chạy, lần tương tác cuối của từng thread. Có thì chạy được nhiều worker/replica, restart không mất hội thoại.
# Để trống: giữ trong RAM (chạy thử, test) - mất khi restart, chỉ 1 process.
AGENT_DATABASE_URL = os.getenv('AGENT_DATABASE_URL', '')
