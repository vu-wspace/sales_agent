# Chính sách cửa hàng: be giữ bản gốc trong Postgres, agent chỉ chunk + embed + lưu Qdrant.
# Quy trình (api.py: /admin/policies): sửa bản nháp -> kiểm tra bằng câu hỏi mẫu (agent thử bản nháp, không ghi
# Qdrant) -> xuất bản (thêm 1 version vào lịch sử) -> đẩy sang agent. Khách chỉ thấy bản đã xuất bản.
# Agent tắt lúc xuất bản thì sync() đối chiếu hash và đẩy bù (be chạy sync khi khởi động và mỗi POLICY_SYNC_S giây).
import logging

import httpx
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from phone_common.config import DATABASE_URL, ROOT
from phone_common.policies import load_seed, policy_hash

log = logging.getLogger('uvicorn.error')  # cùng logger với api.py để log đồng bộ hiện trong docker logs
SCHEMA = ROOT / 'data_schema' / 'policies.sql'
PUBLISHED = 'slug, content_md, prompt, version, updated_at'


class PolicyError(Exception):
    """Thao tác không hợp lệ với trạng thái hiện tại (chưa có nháp, chưa kiểm tra, kiểm tra chưa đạt...): api trả 409."""


def _connect():
    return psycopg.connect(DATABASE_URL, connect_timeout=5, row_factory=dict_row)


def ensure_table() -> int:
    """Tạo bảng; bảng rỗng thì nạp dữ liệu mẫu (data_schema/policies). Trả về số tài liệu vừa nạp."""
    seed = load_seed()
    with _connect() as conn:
        conn.execute(SCHEMA.read_text(encoding='utf-8'))
        if not conn.execute('SELECT 1 FROM policy_checks LIMIT 1').fetchone():
            _insert_checks(conn, [(d['slug'], c) for d in seed for c in d['checks']])
        if conn.execute('SELECT 1 FROM policies LIMIT 1').fetchone():
            return 0
        for d in seed:
            _publish(conn, d['slug'], d['content_md'], d['prompt'], None, 'dữ liệu mẫu')
        return len(seed)


def _insert_checks(conn, rows: list[tuple[str, dict]]) -> None:
    with conn.cursor() as cur:
        cur.executemany('INSERT INTO policy_checks (slug, question, must_contain) VALUES (%s, %s, %s)',
                        [(slug, c['question'], Jsonb(c.get('must_contain') or [])) for slug, c in rows])


def _publish(conn, slug: str, content_md: str, prompt: str | None, by: str | None, note: str | None) -> dict:
    # Khóa theo slug: 2 lần xuất bản cùng lúc không lấy trùng số version
    conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (slug,))
    version = conn.execute('SELECT coalesce(max(version), 0) + 1 AS v FROM policy_versions WHERE slug = %s',
                           (slug,)).fetchone()['v']
    conn.execute('INSERT INTO policy_versions (slug, version, content_md, prompt, published_by, note) '
                 'VALUES (%s, %s, %s, %s, %s, %s)', (slug, version, content_md, prompt, by, note))
    return conn.execute(
        f"""INSERT INTO policies (slug, content_md, prompt, version) VALUES (%s, %s, %s, %s)
            ON CONFLICT (slug) DO UPDATE SET content_md = EXCLUDED.content_md, prompt = EXCLUDED.prompt,
                                             version = EXCLUDED.version, updated_at = now()
            RETURNING {PUBLISHED}""", (slug, content_md, prompt, version)).fetchone()


# ---------------------------------------------------------------- bản đã xuất bản

def list_all() -> list[dict]:
    with _connect() as conn:
        return conn.execute(f'SELECT {PUBLISHED} FROM policies ORDER BY slug').fetchall()


def overview() -> list[dict]:
    """Mọi tài liệu (đã xuất bản hoặc mới có nháp), kèm trạng thái nháp, không kèm nội dung."""
    with _connect() as conn:
        return conn.execute(
            """SELECT coalesce(p.slug, d.slug) AS slug, p.version, p.updated_at, d.slug IS NOT NULL AS has_draft,
                      d.updated_at AS draft_updated_at, (d.check_result ->> 'passed')::boolean AS draft_passed
               FROM policies p FULL JOIN policy_drafts d ON d.slug = p.slug ORDER BY 1""").fetchall()


def get(slug: str) -> dict | None:
    with _connect() as conn:
        return conn.execute(f'SELECT {PUBLISHED} FROM policies WHERE slug = %s', (slug,)).fetchone()


def delete(slug: str) -> bool:
    """Gỡ tài liệu khỏi bản đang dùng (và bỏ nháp). Lịch sử, câu hỏi kiểm tra giữ lại."""
    with _connect() as conn:
        conn.execute('DELETE FROM policy_drafts WHERE slug = %s', (slug,))
        return conn.execute('DELETE FROM policies WHERE slug = %s', (slug,)).rowcount > 0


# ---------------------------------------------------------------- nháp, kiểm tra, xuất bản

def get_draft(slug: str) -> dict | None:
    with _connect() as conn:
        return conn.execute('SELECT slug, content_md, prompt, updated_at, check_result, checked_hash '
                            'FROM policy_drafts WHERE slug = %s', (slug,)).fetchone()


def save_draft(slug: str, content_md: str, prompt: str | None) -> dict:
    # Sửa nháp thì kết quả kiểm tra cũ không còn đúng -> xóa
    with _connect() as conn:
        return conn.execute(
            """INSERT INTO policy_drafts (slug, content_md, prompt) VALUES (%s, %s, %s)
               ON CONFLICT (slug) DO UPDATE SET content_md = EXCLUDED.content_md, prompt = EXCLUDED.prompt,
                                                updated_at = now(), check_result = NULL, checked_hash = NULL
               RETURNING slug, content_md, prompt, updated_at, check_result, checked_hash""",
            (slug, content_md, prompt)).fetchone()


def delete_draft(slug: str) -> bool:
    with _connect() as conn:
        return conn.execute('DELETE FROM policy_drafts WHERE slug = %s', (slug,)).rowcount > 0


def get_checks(slug: str) -> list[dict]:
    with _connect() as conn:
        return conn.execute('SELECT question, must_contain FROM policy_checks WHERE slug = %s ORDER BY id',
                            (slug,)).fetchall()


def set_checks(slug: str, checks: list[dict]) -> list[dict]:
    """Thay bộ câu hỏi kiểm tra của tài liệu. Kết quả kiểm tra nháp cũ bị xóa (đã kiểm bằng bộ câu khác)."""
    with _connect() as conn:
        conn.execute('DELETE FROM policy_checks WHERE slug = %s', (slug,))
        _insert_checks(conn, [(slug, c) for c in checks])
        conn.execute('UPDATE policy_drafts SET check_result = NULL, checked_hash = NULL WHERE slug = %s', (slug,))
    return get_checks(slug)


def check_draft(http: httpx.Client, slug: str) -> dict:
    """Agent thử bản nháp với câu hỏi mẫu (không ghi Qdrant, không gọi LLM); lưu kết quả kèm hash bản nháp.
    Agent lỗi thì raise httpx.HTTPError."""
    draft = get_draft(slug)
    if not draft:
        raise PolicyError(f'Chưa có bản nháp cho {slug}')
    checks = get_checks(slug)
    if not checks:
        raise PolicyError(f'{slug} chưa có câu hỏi kiểm tra: thêm bằng PUT /admin/policies/{slug}/checks')
    r = http.post(f'/policies/{slug}/check', json={'content_md': draft['content_md'], 'questions': checks},
                  timeout=300)
    r.raise_for_status()
    result = r.json()
    with _connect() as conn:
        conn.execute('UPDATE policy_drafts SET check_result = %s, checked_hash = %s WHERE slug = %s',
                     (Jsonb(result), policy_hash(draft['content_md'], draft['prompt']), slug))
    return result


def publish(slug: str, by: str | None = None, note: str | None = None) -> dict:
    """Xuất bản bản nháp: chỉ khi lần kiểm tra gần nhất đạt và chạy trên đúng nội dung nháp hiện tại."""
    with _connect() as conn:
        draft = conn.execute('SELECT content_md, prompt, check_result, checked_hash FROM policy_drafts '
                             'WHERE slug = %s FOR UPDATE', (slug,)).fetchone()
        if not draft:
            raise PolicyError(f'Chưa có bản nháp cho {slug}')
        if not draft['check_result'] or draft['checked_hash'] != policy_hash(draft['content_md'], draft['prompt']):
            raise PolicyError('Bản nháp chưa được kiểm tra (hoặc đã sửa sau lần kiểm tra): chạy POST .../draft/check')
        if not draft['check_result'].get('passed'):
            raise PolicyError('Bản nháp chưa đạt kiểm tra: xem check_result, sửa nháp hoặc câu hỏi kiểm tra rồi kiểm lại')
        doc = _publish(conn, slug, draft['content_md'], draft['prompt'], by, note)
        conn.execute('DELETE FROM policy_drafts WHERE slug = %s', (slug,))
    log.info('Xuất bản chính sách %s version %d (%s)', slug, doc['version'], by or 'không ghi người')
    return doc


# ---------------------------------------------------------------- lịch sử

def versions(slug: str) -> list[dict]:
    with _connect() as conn:
        return conn.execute('SELECT version, published_by, note, published_at FROM policy_versions '
                            'WHERE slug = %s ORDER BY version DESC', (slug,)).fetchall()


def get_version(slug: str, version: int) -> dict | None:
    with _connect() as conn:
        return conn.execute('SELECT slug, version, content_md, prompt, published_by, note, published_at '
                            'FROM policy_versions WHERE slug = %s AND version = %s', (slug, version)).fetchone()


def rollback(slug: str, version: int, by: str | None = None, note: str | None = None) -> dict:
    """Quay lại bản cũ: xuất bản lại nội dung của version đó thành version mới (lịch sử không bị xóa).
    Không chạy kiểm tra: bản cũ đã từng được xuất bản; dùng khi cần gỡ nhanh bản lỗi."""
    old = get_version(slug, version)
    if not old:
        raise PolicyError(f'{slug} không có version {version}')
    with _connect() as conn:
        doc = _publish(conn, slug, old['content_md'], old['prompt'], by, note or f'quay lại version {version}')
    log.info('Quay lại chính sách %s version %d -> version mới %d', slug, version, doc['version'])
    return doc


# ---------------------------------------------------------------- đồng bộ sang agent

def push(http: httpx.Client, doc: dict) -> None:
    """Đẩy 1 tài liệu sang agent (chunk + embed). Lỗi thì raise httpx.HTTPError."""
    http.put(f"/policies/{doc['slug']}", json={'content_md': doc['content_md'], 'prompt': doc['prompt']},
             timeout=300).raise_for_status()


def sync(http: httpx.Client) -> dict:
    """Đưa Qdrant của agent về đúng bản đã xuất bản: đẩy tài liệu agent chưa có hoặc khác hash, xóa tài liệu đã bỏ.
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
