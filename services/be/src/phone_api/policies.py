# Chính sách cửa hàng: be giữ bản gốc trong Postgres (bảng policies), agent chỉ chunk + embed + lưu Qdrant.
# Sửa qua /admin/policies (api.py) thì đẩy ngay sang agent; agent tắt lúc đó thì sync() đối chiếu hash và đẩy bù
# (be chạy sync khi khởi động và mỗi POLICY_SYNC_S giây).
import logging

import httpx
import psycopg
from psycopg.rows import dict_row

from phone_common.config import DATABASE_URL, ROOT
from phone_common.policies import load_seed, policy_hash

log = logging.getLogger('uvicorn.error')  # cùng logger với api.py để log đồng bộ hiện trong docker logs
SCHEMA = ROOT / 'data_schema' / 'policies.sql'


def _connect():
    return psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row)


def ensure_table() -> int:
    """Tạo bảng; bảng rỗng thì nạp dữ liệu mẫu (data_schema/policies). Trả về số tài liệu vừa nạp."""
    with _connect() as conn:
        conn.execute(SCHEMA.read_text(encoding='utf-8'))
        if conn.execute('SELECT 1 FROM policies LIMIT 1').fetchone():
            return 0
        seed = load_seed()
        with conn.cursor() as cur:
            cur.executemany('INSERT INTO policies (slug, content_md, prompt) VALUES (%(slug)s, %(content_md)s, %(prompt)s) '
                            'ON CONFLICT (slug) DO NOTHING', seed)
        return len(seed)


def list_all() -> list[dict]:
    with _connect() as conn:
        return conn.execute('SELECT slug, content_md, prompt, updated_at FROM policies ORDER BY slug').fetchall()


def get(slug: str) -> dict | None:
    with _connect() as conn:
        return conn.execute('SELECT slug, content_md, prompt, updated_at FROM policies WHERE slug = %s',
                            (slug,)).fetchone()


def save(slug: str, content_md: str, prompt: str | None) -> dict:
    with _connect() as conn:
        return conn.execute(
            """INSERT INTO policies (slug, content_md, prompt) VALUES (%s, %s, %s)
               ON CONFLICT (slug) DO UPDATE SET content_md = EXCLUDED.content_md, prompt = EXCLUDED.prompt,
                                                updated_at = now()
               RETURNING slug, content_md, prompt, updated_at""", (slug, content_md, prompt)).fetchone()


def delete(slug: str) -> bool:
    with _connect() as conn:
        return conn.execute('DELETE FROM policies WHERE slug = %s', (slug,)).rowcount > 0


def push(http: httpx.Client, doc: dict) -> None:
    """Đẩy 1 tài liệu sang agent (chunk + embed). Lỗi thì raise: httpx.HTTPError."""
    http.put(f"/policies/{doc['slug']}", json={'content_md': doc['content_md'], 'prompt': doc['prompt']},
             timeout=300).raise_for_status()


def sync(http: httpx.Client) -> dict:
    """Đưa Qdrant của agent về đúng bảng policies: đẩy tài liệu agent chưa có hoặc khác hash, xóa tài liệu đã bỏ.
    Agent không kết nối được thì raise httpx.HTTPError (lần sync sau thử lại)."""
    docs = {d['slug']: d for d in list_all()}
    r = http.get('/policies')
    r.raise_for_status()
    remote = r.json()
    pushed = [s for s, d in docs.items() if remote.get(s) != policy_hash(d['content_md'], d['prompt'])]
    for slug in pushed:
        push(http, docs[slug])
    removed = [s for s in remote if s not in docs]
    for slug in removed:
        http.delete(f'/policies/{slug}').raise_for_status()
    if pushed or removed:
        log.info('Đồng bộ chính sách sang agent: đẩy %s, xóa %s', pushed, removed)
    return {'pushed': pushed, 'removed': removed}
