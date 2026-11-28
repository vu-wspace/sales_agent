# Chính sách: be giữ bản gốc (nháp -> kiểm tra -> xuất bản -> lịch sử), đẩy bản đã xuất bản sang agent;
# agent chunk + embed + lưu Qdrant theo từng tài liệu, và thử bản nháp mà không ghi gì.
import hashlib
import math
import re
import uuid

import httpx
import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from phone_agent import policy_sync, vector_db
from phone_agent import service as svc
from phone_api import api as be
from phone_api.config import DATABASE_URL
from phone_common.config import VECTOR_SIZE
from phone_common.policies import load_seed, policy_hash
from qdrant_client import QdrantClient

DOC = '# Đổi trả\n\n## Lỗi nhà sản xuất\nĐổi mới trong 30 ngày.\n\n## Đổi ý\nCòn seal trong 7 ngày, phí 10%.\n'
SHORT = '# Đổi trả\n\n## Lỗi nhà sản xuất\nĐổi mới trong 15 ngày.\n'
SHIP = '# Giao hàng\n\n## Phí giao\nMiễn phí giao hàng toàn quốc.\n'


class FakeEmbeddings(Embeddings):
    # Không tải model 540MB: túi từ băm vào vector đã chuẩn hóa -> cosine theo số từ chung, đủ để xếp hạng
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        v = [0.0] * VECTOR_SIZE
        for w in re.findall(r'\w+', text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % VECTOR_SIZE] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]


@pytest.fixture
def qdrant(monkeypatch):
    client = QdrantClient(':memory:')
    monkeypatch.setattr(vector_db, 'get_client', lambda: client)
    monkeypatch.setattr(vector_db, 'get_embeddings', FakeEmbeddings)
    return client


# ---------------------------------------------------------------- agent: Qdrant theo từng tài liệu

def test_upsert_replaces_only_that_document(qdrant):
    assert vector_db.upsert_policy('doi-tra.md', DOC, 'Hỏi rõ trường hợp') == 2
    assert vector_db.upsert_policy('giao-hang.md', SHIP, None) == 1
    # Sửa tài liệu ít mục hơn: chunk cũ của nó bị thay hết, tài liệu khác giữ nguyên
    assert vector_db.upsert_policy('doi-tra.md', SHORT, 'Prompt mới') == 1
    assert vector_db.count_documents() == 2
    assert vector_db.policy_hashes() == {'doi-tra.md': policy_hash(SHORT, 'Prompt mới'),
                                         'giao-hang.md': policy_hash(SHIP, None)}
    # search_policy đọc được prompt riêng từ payload
    hits = vector_db.get_vector_store().similarity_search('đổi máy lỗi', k=5)
    assert {h.metadata['source']: h.metadata['prompt'] for h in hits} == {'doi-tra.md': 'Prompt mới',
                                                                          'giao-hang.md': None}
    vector_db.delete_policy('doi-tra.md')
    assert set(vector_db.policy_hashes()) == {'giao-hang.md'}


def test_upsert_empty_document_keeps_old_chunks(qdrant):
    vector_db.upsert_policy('doi-tra.md', DOC)
    with pytest.raises(ValueError):
        vector_db.upsert_policy('doi-tra.md', '   ')
    assert vector_db.count_documents() == 2


def test_reconcile_loads_new_documents_and_removes_deleted(qdrant, monkeypatch):
    docs = {d['slug']: d for d in load_seed()}
    manifest = {slug: policy_hash(doc['content_md'], doc['prompt']) for slug, doc in docs.items()}

    def get(path):
        if path == '/policies':
            return manifest
        slug = path.removeprefix('/policies/')
        if slug in docs:
            return docs[slug]
        response = httpx.Response(404, request=httpx.Request('GET', f'http://be{path}'))
        response.raise_for_status()

    monkeypatch.setattr(policy_sync.backend, 'get', get)
    vector_db.upsert_policy('cu.md', DOC)
    assert policy_sync.reconcile() == {'pushed': list(docs), 'removed': ['cu.md'], 'failed': []}
    assert vector_db.policy_hashes() == manifest
    assert policy_sync.reconcile() == {'pushed': [], 'removed': [], 'failed': []}


def test_reconcile_one_bad_document_does_not_block_others(qdrant, monkeypatch):
    docs = {d['slug']: d for d in load_seed()}
    bad = next(iter(docs))

    def get(path):
        if path == '/policies':
            return {slug: policy_hash(doc['content_md'], doc['prompt']) for slug, doc in docs.items()}
        slug = path.removeprefix('/policies/')
        if slug == bad:
            raise httpx.ConnectError('be lỗi đúng lúc này')
        if slug in docs:
            return docs[slug]
        httpx.Response(404, request=httpx.Request('GET', f'http://be{path}')).raise_for_status()

    monkeypatch.setattr(policy_sync.backend, 'get', get)
    vector_db.upsert_policy('cu.md', DOC)   # tạo collection; be không còn tài liệu này -> phải bị gỡ
    result = policy_sync.reconcile()
    assert result['failed'] == [bad] and result['removed'] == ['cu.md']
    assert sorted(result['pushed']) == sorted(set(docs) - {bad})


def test_check_policy_tries_draft_without_writing(qdrant):
    vector_db.upsert_policy('doi-tra.md', DOC)
    vector_db.upsert_policy('giao-hang.md', SHIP)
    before = vector_db.policy_hashes()
    result = vector_db.check_policy('doi-tra.md', SHORT, [
        {'question': 'lỗi nhà sản xuất đổi mới trong bao nhiêu ngày', 'must_contain': [['15 ngày']]},
        {'question': 'lỗi nhà sản xuất đổi mới trong bao nhiêu ngày', 'must_contain': [['30 ngày']]},
        {'question': 'phí giao hàng toàn quốc', 'must_contain': []},
    ])
    ok, missing_fact, other_doc = result['results']
    assert ok['passed'] and ok['retrieved'][0] == 'doi-tra.md'
    # Bản nháp bỏ mất ý "30 ngày" -> câu đó không đạt, chỉ rõ thiếu gì
    assert not missing_fact['passed'] and missing_fact['missing'] == [['30 ngày']]
    # Câu hỏi về tài liệu khác lấy ra tài liệu khác -> không đạt cho tài liệu này
    assert not other_doc['passed'] and other_doc['retrieved'][0] == 'giao-hang.md'
    assert result['passed'] is False
    assert vector_db.policy_hashes() == before   # không ghi gì vào Qdrant


def test_every_seed_policy_has_checks():
    assert all(d['checks'] for d in load_seed())


# ---------------------------------------------------------------- be: Postgres thật (database tạm), agent giả

@pytest.fixture
def pg_url():
    # Database tạm trên Postgres của docker compose; không có Postgres thì bỏ qua (CI không có database)
    name = f'policy_test_{uuid.uuid4().hex[:8]}'
    admin_url = re.sub(r'/[^/]*$', '/postgres', DATABASE_URL)
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=2) as conn:
            conn.execute(f'CREATE DATABASE {name}')
    except psycopg.OperationalError:
        pytest.skip('Không có Postgres (docker compose up -d postgres)')
    yield re.sub(r'/[^/]*$', f'/{name}', DATABASE_URL)
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as conn:
        conn.execute(f'DROP DATABASE {name} WITH (FORCE)')


class FakeStore:
    """Thay Qdrant của agent: {slug: (content, prompt)}. Kiểm tra nháp: đạt khi nội dung chứa đủ must_contain."""

    def __init__(self):
        self.docs = {}

    def upsert(self, slug, content_md, prompt=None):
        if not content_md.strip():
            raise ValueError('trống')
        self.docs[slug] = (content_md, prompt)
        return 1

    def hashes(self):
        return {s: policy_hash(c, p) for s, (c, p) in self.docs.items()}

    def check(self, slug, content_md, questions):
        results = [{'question': q['question'], 'retrieved': [slug],
                    'missing': (m := [g for g in q['must_contain'] if not any(k in content_md for k in g)]),
                    'passed': not m} for q in questions]
        return {'passed': all(r['passed'] for r in results), 'results': results}


ADMIN = {'Authorization': 'Bearer admin-key'}


@pytest.fixture
def stack(monkeypatch, pg_url):
    monkeypatch.setattr(be.policies, 'DATABASE_URL', pg_url)
    assert be.policies.ensure_table() == 5          # bảng rỗng: nạp 5 tài liệu mẫu
    store = FakeStore()
    monkeypatch.setattr(vector_db, 'upsert_policy', store.upsert)
    monkeypatch.setattr(vector_db, 'delete_policy', lambda slug: store.docs.pop(slug, None))
    monkeypatch.setattr(vector_db, 'policy_hashes', store.hashes)
    monkeypatch.setattr(svc, 'check_policy', store.check)
    monkeypatch.setattr(be, 'ADMIN_API_KEY', 'admin-key')
    agent = TestClient(svc.app)
    client = TestClient(be.app)
    monkeypatch.setattr(be.catalog, 'CATALOG_API_KEY', '')

    def transport(target):
        def send(request):
            response = target.request(request.method, request.url.raw_path.decode(), content=request.content,
                                      headers=dict(request.headers))
            return httpx.Response(response.status_code, content=response.content, headers=dict(response.headers))
        return httpx.MockTransport(send)

    with (httpx.Client(transport=transport(agent), base_url='http://agent') as agent_http,
          httpx.Client(transport=transport(client), base_url='http://be') as backend_http):
        monkeypatch.setattr(be.app.state, 'agent_http', agent_http, raising=False)

        def get(path):
            response = backend_http.get(path)
            response.raise_for_status()
            return response.json()

        monkeypatch.setattr(policy_sync.backend, 'get', get)
        assert client.post('/admin/policies/sync', headers=ADMIN).json()['pushed'] == [d['slug'] for d in load_seed()]
        yield store, client


def test_seed_is_version_1_with_history_and_checks(stack):
    store, client = stack
    docs = client.get('/admin/policies', headers=ADMIN).json()
    assert len(docs) == 5 and all(d['version'] == 1 and not d['has_draft'] for d in docs)
    assert [v['version'] for v in client.get('/admin/policies/doi-tra.md/versions', headers=ADMIN).json()] == [1]
    assert len(client.get('/admin/policies/doi-tra.md/checks', headers=ADMIN).json()) == 4
    assert be.policies.ensure_table() == 0          # chạy lại khi khởi động: không nạp trùng
    assert len(client.get('/admin/policies/doi-tra.md/checks', headers=ADMIN).json()) == 4


def test_draft_check_publish_flow(stack):
    store, client = stack
    live = client.get('/admin/policies/doi-tra.md', headers=ADMIN).json()
    assert client.post('/admin/policies/doi-tra.md/publish', headers=ADMIN).status_code == 409   # chưa có nháp

    # Đổi 30 ngày -> 15 ngày: nháp chưa tới tay khách
    changed = live['content_md'].replace('30 ngày', '15 ngày')
    new_prompt = live['prompt'].replace('30 ngày', '15 ngày')   # prompt cũng nêu mốc ngày: sửa cả hai
    assert client.put('/policies/doi-tra.md', json={'content_md': changed, 'prompt': new_prompt},
                      headers=ADMIN).status_code == 200
    assert client.get('/policies/doi-tra.md').json()['content_md'] == live['content_md']
    assert store.docs['doi-tra.md'][0] == live['content_md']
    r = client.post('/admin/policies/doi-tra.md/publish', headers=ADMIN)
    assert r.status_code == 409 and 'chưa được kiểm tra' in r.json()['detail']

    # Câu hỏi mẫu còn đòi "30 ngày" -> không đạt, không xuất bản được
    result = client.post('/admin/policies/doi-tra.md/draft/check', headers=ADMIN).json()
    assert result['passed'] is False and [['30 ngày']] in [x['missing'] for x in result['results']]
    assert 'chưa đạt' in client.post('/admin/policies/doi-tra.md/publish', headers=ADMIN).json()['detail']

    # Sửa câu hỏi mẫu cho đúng chính sách mới -> kết quả cũ bị xóa, kiểm lại thì đạt
    checks = client.get('/admin/policies/doi-tra.md/checks', headers=ADMIN).json()
    for c in checks:
        c['must_contain'] = [['15 ngày'] if g == ['30 ngày'] else g for g in c['must_contain']]
    client.put('/admin/policies/doi-tra.md/checks', json=checks, headers=ADMIN)
    assert client.get('/admin/policies/doi-tra.md/draft', headers=ADMIN).json()['check_result'] is None
    assert client.post('/admin/policies/doi-tra.md/draft/check', headers=ADMIN).json()['passed'] is True

    # Sửa nháp sau khi kiểm -> phải kiểm lại
    client.put('/admin/policies/doi-tra.md/draft', json={'content_md': changed + '\n', 'prompt': new_prompt},
               headers=ADMIN)
    assert client.post('/admin/policies/doi-tra.md/publish', headers=ADMIN).status_code == 409
    client.post('/admin/policies/doi-tra.md/draft/check', headers=ADMIN)

    r = client.post('/admin/policies/doi-tra.md/publish', json={'by': 'Lan', 'note': 'đổi 30 -> 15 ngày'}, headers=ADMIN)
    assert r.status_code == 200 and r.json()['version'] == 2 and r.json()['synced'] is True
    assert store.docs['doi-tra.md'][0] == changed + '\n'                       # agent có bản mới
    assert client.get('/admin/policies/doi-tra.md/draft', headers=ADMIN).status_code == 404   # nháp đã dùng
    history = client.get('/admin/policies/doi-tra.md/versions', headers=ADMIN).json()
    assert [(v['version'], v['published_by'], v['note']) for v in history] == [
        (2, 'Lan', 'đổi 30 -> 15 ngày'), (1, None, 'dữ liệu mẫu')]
    assert client.get('/admin/policies/doi-tra.md/versions/1', headers=ADMIN).json()['content_md'] == live['content_md']


def test_rollback_publishes_old_version_as_new(stack):
    store, client = stack
    live = client.get('/admin/policies/giao-hang.md', headers=ADMIN).json()
    client.put('/admin/policies/giao-hang.md/checks', json=[{'question': 'phí giao', 'must_contain': []}], headers=ADMIN)
    client.put('/admin/policies/giao-hang.md/draft', json={'content_md': SHIP}, headers=ADMIN)
    client.post('/admin/policies/giao-hang.md/draft/check', headers=ADMIN)
    assert client.post('/admin/policies/giao-hang.md/publish', headers=ADMIN).json()['version'] == 2
    r = client.post('/admin/policies/giao-hang.md/rollback', json={'version': 1, 'by': 'Lan'}, headers=ADMIN)
    assert r.json()['version'] == 3 and r.json()['content_md'] == live['content_md'] and r.json()['synced']
    assert store.docs['giao-hang.md'] == (live['content_md'], live['prompt'])
    assert [v['note'] for v in client.get('/admin/policies/giao-hang.md/versions', headers=ADMIN).json()][0] == \
        'quay lại version 1'
    assert client.post('/admin/policies/giao-hang.md/rollback', json={'version': 9}, headers=ADMIN).status_code == 409


def test_new_policy_needs_checks_before_publish(stack):
    store, client = stack
    client.put('/admin/policies/doi-diem.md/draft', json={'content_md': '# Đổi điểm\n\n## Quy đổi\n1 điểm = 1.000đ\n'},
               headers=ADMIN)
    r = client.post('/admin/policies/doi-diem.md/draft/check', headers=ADMIN)
    assert r.status_code == 409 and 'chưa có câu hỏi kiểm tra' in r.json()['detail']
    client.put('/admin/policies/doi-diem.md/checks', json=[{'question': '1 điểm bao nhiêu tiền',
                                                            'must_contain': [['1.000đ']]}], headers=ADMIN)
    client.post('/admin/policies/doi-diem.md/draft/check', headers=ADMIN)
    assert client.post('/admin/policies/doi-diem.md/publish', headers=ADMIN).json()['version'] == 1
    assert 'doi-diem.md' in store.docs
    assert any(d['slug'] == 'doi-diem.md' for d in client.get('/admin/policies', headers=ADMIN).json())


def test_publish_when_agent_down_then_sync(stack):
    store, client = stack
    client.put('/admin/policies/giao-hang.md/checks', json=[{'question': 'phí giao', 'must_contain': []}], headers=ADMIN)
    client.put('/admin/policies/giao-hang.md/draft', json={'content_md': SHIP}, headers=ADMIN)
    client.post('/admin/policies/giao-hang.md/draft/check', headers=ADMIN)
    real_http = be.app.state.agent_http
    def refuse(request):
        raise httpx.ConnectError('agent tắt', request=request)
    be.app.state.agent_http = httpx.Client(transport=httpx.MockTransport(refuse), base_url='http://agent')
    r = client.post('/admin/policies/giao-hang.md/publish', headers=ADMIN)
    assert r.status_code == 200 and r.json()['synced'] is False and r.json()['version'] == 2   # vẫn xuất bản
    assert client.post('/admin/policies/sync', headers=ADMIN).status_code == 503
    # Agent bật lại: sync đẩy tài liệu lệch hash, xóa tài liệu agent có mà be đã bỏ
    be.app.state.agent_http = real_http
    store.docs['cu.md'] = ('# Cũ', None)
    assert client.post('/admin/policies/sync', headers=ADMIN).json() == {'pushed': ['giao-hang.md'], 'removed': ['cu.md'],
                                                                          'failed': []}
    assert store.docs['giao-hang.md'] == (SHIP, None)
    assert client.post('/admin/policies/sync', headers=ADMIN).json() == {'pushed': [], 'removed': [], 'failed': []}


def test_delete_policy_keeps_history(stack):
    store, client = stack
    assert client.delete('/admin/policies/tra-gop.md', headers=ADMIN).json()['synced'] is True
    assert 'tra-gop.md' not in store.docs
    assert client.get('/admin/policies/tra-gop.md', headers=ADMIN).status_code == 404
    assert [v['version'] for v in client.get('/admin/policies/tra-gop.md/versions', headers=ADMIN).json()] == [1]


# ---------------------------------------------------------------- xác thực, kiểm tra đầu vào (không cần database)

def test_admin_needs_key(monkeypatch):
    client = TestClient(be.app)
    monkeypatch.setattr(be, 'ADMIN_API_KEY', 'admin-key')
    assert client.get('/admin/policies').status_code == 401
    assert client.get('/admin/policies', headers={'Authorization': 'Bearer sai'}).status_code == 401
    monkeypatch.setattr(be, 'ADMIN_API_KEY', '')
    assert client.get('/admin/policies', headers=ADMIN).status_code == 503   # chưa đặt key: tắt hẳn


@pytest.mark.parametrize('slug, body', [('Doi Tra', {'content_md': DOC}), ('../x.md', {'content_md': DOC}),
                                        ('doi-tra', {'content_md': DOC}), ('doi-tra.md', {'content_md': '  '})])
def test_draft_validation(monkeypatch, slug, body):
    monkeypatch.setattr(be, 'ADMIN_API_KEY', 'admin-key')
    saved = []
    monkeypatch.setattr(be.policies, 'save_draft', lambda *a: saved.append(a))
    r = TestClient(be.app).put(f'/admin/policies/{slug}/draft', json=body, headers=ADMIN)
    assert r.status_code in (404, 422) and saved == []


def test_agent_policy_endpoints_need_agent_key(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(vector_db, 'upsert_policy', store.upsert)
    monkeypatch.setattr(svc, 'check_policy', store.check)
    monkeypatch.setattr(svc, 'AGENT_API_KEY', 'agent-key')
    agent = TestClient(svc.app)
    key = {'Authorization': 'Bearer agent-key'}
    def get(path):
        assert path == '/policies/doi-tra.md'
        return {'content_md': DOC, 'prompt': None}

    monkeypatch.setattr(policy_sync.backend, 'get', get)
    monkeypatch.setattr(vector_db, 'policy_hashes', store.hashes)
    event = {'slug': 'doi-tra.md'}
    assert agent.post('/webhooks/policy-updated', json=event).status_code == 401
    assert agent.post('/webhooks/policy-updated', json=event, headers=key).status_code == 204
    assert store.docs['doi-tra.md'] == (DOC, None)
    assert agent.post('/webhooks/policy-updated', json={'slug': '../bad'}, headers=key).status_code == 422
    r = agent.post('/policies/doi-tra.md/check', headers=key,
                   json={'content_md': DOC, 'questions': [{'question': 'bao lâu', 'must_contain': [['30 ngày']]}]})
    assert r.json()['passed'] is True


def test_policy_reconciliation_retries_fast_on_error_then_hourly(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    calls = []
    results = [httpx.ConnectError('be đang khởi động'), {'failed': ['doi-tra.md']},
               {'failed': []}, httpx.ConnectError('be restart')]

    async def to_thread(fn):
        assert fn is policy_sync.reconcile
        calls.append('reconcile')
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def sleep(seconds):
        calls.append(seconds)
        if not results:
            raise asyncio.CancelledError

    monkeypatch.setattr(svc, 'asyncio', SimpleNamespace(to_thread=to_thread, sleep=sleep))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(svc._reconcile_loop())
    # lỗi: 30s rồi gấp đôi; thành công: 1 giờ và đặt lại mức thử lại về 30s
    assert calls == ['reconcile', 30, 'reconcile', 60, 'reconcile', 3600, 'reconcile', 30]
