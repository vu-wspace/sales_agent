# Chính sách cửa hàng: bản gốc nằm ở be (bảng policies trong Postgres), be đẩy sang agent để chunk + embed vào Qdrant.
# Module này dùng chung cho 2 bên: cách tính hash để be biết agent đã có đúng bản chưa, và dữ liệu mẫu ban đầu.
import hashlib
import json
import re

from phone_common.config import POLICY_DIR

# slug = tên tài liệu, cũng là `source` của chunk trong Qdrant và tên nhóm trong kết quả search_policy (vd 'doi-tra.md')
SLUG = r'^[a-z0-9][a-z0-9-]*\.md$'
MAX_CONTENT = 100_000   # ký tự markdown mỗi tài liệu
MAX_PROMPT = 4_000


def policy_hash(content_md: str, prompt: str | None) -> str:
    """Hash của 1 tài liệu (nội dung + prompt). be so hash này với hash agent trả về để biết cần đẩy lại không."""
    return hashlib.sha256(f'{content_md}\0{prompt or ""}'.encode('utf-8')).hexdigest()


def load_seed(policy_dir=POLICY_DIR) -> list[dict]:
    """Dữ liệu mẫu: data_schema/policies/*.md + prompt riêng (prompts.json) + câu hỏi kiểm tra (checks.json, lấy từ
    các câu policy của evals/dataset.jsonl). be nạp vào database khi bảng rỗng; agent nạp thẳng vào Qdrant khi chạy
    không có be (AUTO_INGEST, python -m phone_agent.vector_db, eval)."""
    prompts = json.loads((policy_dir / 'prompts.json').read_text(encoding='utf-8'))
    checks = json.loads((policy_dir / 'checks.json').read_text(encoding='utf-8'))
    files = sorted(policy_dir.glob('*.md'))
    if not files:
        raise FileNotFoundError(f'Không có file chính sách .md trong {policy_dir}')
    docs = [{'slug': f.name, 'content_md': f.read_text(encoding='utf-8'), 'prompt': prompts.get(f.name),
             'checks': checks.get(f.name, [])} for f in files]
    assert all(re.fullmatch(SLUG, d['slug']) for d in docs), 'Tên file chính sách phải dạng chữ-thường-gạch-ngang.md'
    return docs
