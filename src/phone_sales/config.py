import os
from pathlib import Path

# Đường dẫn dữ liệu (dự án cài dạng editable nên tính từ vị trí file này: src/phone_sales/config.py)
ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / 'data_schema' / 'policies'   # chính sách đổi trả, giao hàng, trả góp, bảo hành (.md) -> nạp vào Qdrant
CHUNK_MAX_CHARS = 900                            # model bkai cắt ở 256 token, chunk dài hơn sẽ mất phần cuối khi embed

# Database sản phẩm (products, variants, compatibility). Mặc định trỏ tới container phone_db (cổng 5433)
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5433/phone')

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

# Web app dùng agent nào: 'graph' (graph.py, có trạng thái hội thoại + giỏ hàng) hoặc 'agent' (agent.py)
AGENT_IMPL = os.getenv('AGENT_IMPL', 'graph').lower()

# Web app: collection rỗng thì tự nạp dữ liệu khi khởi động
AUTO_INGEST = os.getenv('AUTO_INGEST', 'true').lower() == 'true'
