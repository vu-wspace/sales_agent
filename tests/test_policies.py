# Chính sách: be giữ bản gốc (bảng policies), đẩy sang agent; agent chunk + embed + lưu Qdrant theo từng tài liệu
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from phone_agent import service as svc
from phone_agent import vector_db
from phone_api import api as be
from phone_common.config import VECTOR_SIZE
from phone_common.policies import load_seed, policy_hash

DOC = '# Đổi trả\n\n## Lỗi nhà sản xuất\nĐổi mới trong 30 ngày.\n\n## Đổi ý\nCòn seal trong 7 ngày, phí 10%.\n'
SHORT = '# Đổi trả\n\n## Lỗi nhà sản xuất\nĐổi mới trong 15 ngày.\n'


class FakeEmbeddings(Embeddings):
    # Không tải model 540MB: vector theo độ dài chữ, đủ để Qdrant lưu và tìm
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [1.0] + [len(text) % 7 / 7] * (VECTOR_SIZE - 1)


@pytest.fixture
def qdrant(monkeypatch):
    client = QdrantClient(':memory:')
    monkeypatch.setattr(vector_db, 'get_client', lambda: client)
    monkeypatch.setattr(vector_db, 'get_embeddings', FakeEmbeddings)
    return client


def test_upsert_replaces_only_that_document(qdrant):
    assert vector_db.upsert_policy('doi-tra.md', DOC, 'Hỏi rõ trường hợp') == 2
    assert vector_db.upsert_policy('giao-hang.md', '# Giao hàng\n\n## Phí\nMiễn phí.\n', None) == 1
    # Sửa tài liệu ít mục hơn: chunk cũ của nó bị thay hết, tài liệu khác giữ nguyên
    assert vector_db.upsert_policy('doi-tra.md', SHORT, 'Prompt mới') == 1
    assert vector_db.count_documents() == 2
    assert vector_db.policy_hashes() == {'doi-tra.md': policy_hash(SHORT, 'Prompt mới'),
                                         'giao-hang.md': policy_hash('# Giao hàng\n\n## Phí\nMiễn phí.\n', None)}
    # search_policy đọc được prompt riêng từ payload
    hits = vector_db.get_vector_store().similarity_search('đổi máy lỗi', k=5)
    assert {h.metadata['source']: h.metadata['prompt'] for h in hits} == {'doi-tra.md': 'Prompt mới',
                                                                          'giao-hang.md': None}
    assert '15 ngày' in next(h.page_content for h in hits if h.metadata['source'] == 'doi-tra.md')
    vector_db.delete_policy('doi-tra.md')
    assert set(vector_db.policy_hashes()) == {'giao-hang.md'}


def test_upsert_empty_document_keeps_old_chunks(qdrant):
    vector_db.upsert_policy('doi-tra.md', DOC)
    with pytest.raises(ValueError):
        vector_db.upsert_policy('doi-tra.md', '   ')
    assert vector_db.count_documents() == 2


def test_ingest_seed_loads_all_files_and_drops_unknown(qdrant):
    vector_db.upsert_policy('cu.md', DOC)
    vector_db.ingest_seed()
    assert set(vector_db.policy_hashes()) == {d['slug'] for d in load_seed()}


# ---------------------------------------------------------------- be <-> agent qua HTTP

class FakeStore:
    """Thay Qdrant của agent: {slug: (content, prompt)}."""

    def __init__(self):
        self.docs = {}

    def upsert(self, slug, content_md, prompt=None):
        if not content_md.strip():
            raise ValueError('trống')
        self.docs[slug] = (content_md, prompt)
        return 1

    def hashes(self):
        return {s: policy_hash(c, p) for s, (c, p) in self.docs.items()}


class FakeTable(dict):
    """Thay bảng policies của be."""

    def save(self, slug, content_md, prompt):
        self[slug] = {'slug': slug, 'content_md': content_md, 'prompt': prompt, 'updated_at': None}
        return self[slug]


ADMIN = {'Authorization': 'Bearer admin-key'}


@pytest.fixture
def stack(monkeypatch):
    store, table = FakeStore(), FakeTable()
    monkeypatch.setattr(svc, 'upsert_policy', store.upsert)
    monkeypatch.setattr(svc, 'delete_policy', lambda slug: store.docs.pop(slug, None))
    monkeypatch.setattr(svc, 'policy_hashes', store.hashes)
    monkeypatch.setattr(be.policies, 'list_all', lambda: sorted(table.values(), key=lambda d: d['slug']))
    monkeypatch.setattr(be.policies, 'get', table.get)
    monkeypatch.setattr(be.policies, 'save', table.save)
    monkeypatch.setattr(be.policies, 'delete', lambda slug: table.pop(slug, None) is not None)
    monkeypatch.setattr(be, 'ADMIN_API_KEY', 'admin-key')
    be.app.state.agent_http = TestClient(svc.app)
    return store, table, TestClient(be.app)


def test_admin_needs_key(stack, monkeypatch):
    client = stack[2]
    assert client.get('/admin/policies').status_code == 401
    assert client.get('/admin/policies', headers={'Authorization': 'Bearer sai'}).status_code == 401
    monkeypatch.setattr(be, 'ADMIN_API_KEY', '')
    assert client.get('/admin/policies', headers=ADMIN).status_code == 503   # chưa đặt key: tắt hẳn


def test_put_policy_saves_and_pushes_to_agent(stack):
    store, table, client = stack
    r = client.put('/admin/policies/doi-tra.md', json={'content_md': DOC, 'prompt': '  Hỏi rõ  '}, headers=ADMIN)
    assert r.status_code == 200 and r.json()['synced'] is True and r.json()['prompt'] == 'Hỏi rõ'
    assert table['doi-tra.md']['content_md'] == DOC and store.docs['doi-tra.md'] == (DOC, 'Hỏi rõ')
    assert client.get('/admin/policies/doi-tra.md', headers=ADMIN).json()['content_md'] == DOC


@pytest.mark.parametrize('slug, body', [('Doi Tra', {'content_md': DOC}), ('../x.md', {'content_md': DOC}),
                                        ('doi-tra', {'content_md': DOC}), ('doi-tra.md', {'content_md': '  '})])
def test_put_policy_validation(stack, slug, body):
    store, table, client = stack
    assert client.put(f'/admin/policies/{slug}', json=body, headers=ADMIN).status_code in (404, 422)
    assert table == {} and store.docs == {}


def test_agent_down_then_sync_pushes_missing(stack):
    store, table, client = stack
    real_http = be.app.state.agent_http
    def refuse(request):
        raise httpx.ConnectError('agent tắt', request=request)
    be.app.state.agent_http = httpx.Client(transport=httpx.MockTransport(refuse), base_url='http://agent')
    r = client.put('/admin/policies/doi-tra.md', json={'content_md': DOC}, headers=ADMIN)
    assert r.status_code == 200 and r.json()['synced'] is False and 'doi-tra.md' in table   # vẫn lưu
    assert client.post('/admin/policies/sync', headers=ADMIN).status_code == 503
    # Agent bật lại: sync đẩy tài liệu còn thiếu/khác hash, xóa tài liệu agent có mà be đã bỏ
    be.app.state.agent_http = real_http
    store.docs['cu.md'] = ('# Cũ', None)
    assert client.post('/admin/policies/sync', headers=ADMIN).json() == {'pushed': ['doi-tra.md'], 'removed': ['cu.md']}
    assert store.hashes() == {'doi-tra.md': policy_hash(DOC, None)}
    # Đã khớp thì không đẩy lại (không embed lại vô ích)
    assert client.post('/admin/policies/sync', headers=ADMIN).json() == {'pushed': [], 'removed': []}


def test_delete_policy(stack):
    store, table, client = stack
    client.put('/admin/policies/doi-tra.md', json={'content_md': DOC}, headers=ADMIN)
    assert client.delete('/admin/policies/doi-tra.md', headers=ADMIN).json()['synced'] is True
    assert table == {} and store.docs == {}
    assert client.delete('/admin/policies/doi-tra.md', headers=ADMIN).status_code == 404


def test_agent_policy_endpoints_need_agent_key(stack, monkeypatch):
    monkeypatch.setattr(svc, 'AGENT_API_KEY', 'agent-key')
    agent = TestClient(svc.app)
    assert agent.put('/policies/doi-tra.md', json={'content_md': DOC}).status_code == 401
    r = agent.put('/policies/doi-tra.md', json={'content_md': DOC}, headers={'Authorization': 'Bearer agent-key'})
    assert r.json() == {'slug': 'doi-tra.md', 'chunks': 1}
    assert agent.put('/policies/doi-tra.md', json={'content_md': ' '},
                     headers={'Authorization': 'Bearer agent-key'}).status_code == 422
