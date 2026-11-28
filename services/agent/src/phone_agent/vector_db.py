import uuid
from functools import lru_cache

import phone_common  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from phone_common.config import (
    COLLECTION,
    EMBEDDING_MODEL,
    POLICY_MIN_SCORE,
    POLICY_SCORE_MARGIN,
    POLICY_TOP_K,
    QDRANT_URL,
    VECTOR_SIZE,
)
from phone_common.policies import policy_hash
from pyvi import ViTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HasIdCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from phone_agent.data_prep import split_markdown


class BkaiEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def _enc(self, texts):
        texts = [ViTokenizer.tokenize(t) for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_documents(self, texts):
        return self._enc(texts)

    def embed_query(self, text):
        return self._enc([text])[0]

@lru_cache
def get_embeddings() -> BkaiEmbeddings:
    # Model ~540MB: chỉ nạp 1 lần dù có nhiều vector store
    return BkaiEmbeddings()

@lru_cache
def get_client() -> QdrantClient:
    # Dùng chung 1 client: tạo mới mỗi lần (vd mỗi lần healthcheck gọi /health) sẽ tích tụ kết nối không đóng.
    # Tạo client không kết nối ngay, nên Qdrant khởi động lại thì client vẫn dùng được.
    return QdrantClient(url=QDRANT_URL)

def get_vector_store() -> QdrantVectorStore:
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=get_embeddings())

def count_documents() -> int:
    return get_client().count(COLLECTION).count

def chunk_ids(chunks) -> list[str]:
    # id tính từ cả nội dung chunk: cùng nội dung thì cùng id
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f'{c.source}#{i}#{c.page_content}')) for i, c in enumerate(chunks)]

def _of(slug: str) -> FilterSelector:
    return FilterSelector(filter=Filter(must=[FieldCondition(key='metadata.source', match=MatchValue(value=slug))]))

def upsert_policy(slug: str, content_md: str, prompt: str | None = None) -> int:
    """Chunk + embed 1 tài liệu chính sách, thay toàn bộ chunk cũ của nó. prompt (hướng dẫn trả lời riêng) và hash
    lưu trong payload từng chunk để search_policy trả prompt và agent đối soát với hash của be."""
    chunks = split_markdown(content_md, slug)
    if not chunks:
        raise ValueError(f'Tài liệu {slug} không có nội dung')
    meta = {'source': slug, 'prompt': prompt, 'doc_hash': policy_hash(content_md, prompt)}
    texts = [c.page_content for c in chunks]
    get_vector_store()  # tạo collection nếu chưa có
    # Ghi chunk mới trước. Nếu upsert lỗi thì vẫn giữ được các chunk cũ.
    # Payload theo dạng QdrantVectorStore đọc (page_content + metadata) để search_policy dùng được ngay
    vectors = get_embeddings().embed_documents(texts)
    points = [PointStruct(id=i, vector=v, payload={'page_content': t, 'metadata': {
                  **meta, 'title': c.title, 'section': c.section}})
              for i, v, t, c in zip(chunk_ids(chunks), vectors, texts, chunks)]
    client = get_client()
    client.upsert(COLLECTION, points=points, wait=True)
    obsolete = Filter(must=_of(slug).filter.must, must_not=[HasIdCondition(has_id=[p.id for p in points])])
    client.delete(COLLECTION, points_selector=FilterSelector(filter=obsolete), wait=True)
    return len(points)

def delete_policy(slug: str) -> None:
    get_client().delete(COLLECTION, points_selector=_of(slug))

def policy_hashes() -> dict[str, str | None]:
    """{slug: hash} của các tài liệu đang có trong Qdrant (mỗi chunk mang hash của tài liệu)."""
    hashes, offset = {}, None
    while True:
        points, offset = get_client().scroll(COLLECTION, limit=256, offset=offset, with_vectors=False,
                                             with_payload=['metadata.source', 'metadata.doc_hash'])
        for p in points:
            m = (p.payload or {}).get('metadata') or {}
            slug, digest = m.get('source'), m.get('doc_hash')
            if slug:
                # A failed cleanup can leave both generations. Reconcile the whole document next time.
                hashes[slug] = digest if hashes.get(slug, digest) == digest else None
        if offset is None:
            return hashes

def relevant_hits(scored: list[tuple]) -> list[tuple]:
    """scored: [(đoạn, điểm cosine)] giảm dần. Bỏ đoạn dưới ngưỡng, và đoạn kém đoạn tốt nhất quá MARGIN.
    Dùng chung cho search_policy và check_policy để bản nháp được kiểm tra đúng như lúc khách hỏi."""
    if not scored or scored[0][1] < POLICY_MIN_SCORE:
        return []
    best = scored[0][1]
    return [(d, s) for d, s in scored if s >= POLICY_MIN_SCORE and best - s <= POLICY_SCORE_MARGIN]

def _norm(text: str) -> str:
    # Như evals/run_eval.py: không phân biệt hoa thường, "7,4" và "7.4"
    return text.lower().replace(',', '.')

def check_policy(slug: str, content_md: str, questions: list[dict]) -> dict:
    """Thử bản nháp trước khi xuất bản, không ghi gì vào Qdrant, không gọi LLM: embed bản nháp trong bộ nhớ, trộn với
    các tài liệu khác đang có, rồi với mỗi câu hỏi xem search_policy có lấy ra tài liệu này không (passed: có đoạn của
    nó sau bộ lọc điểm) và các đoạn đó có đủ ý bắt buộc không (must_contain: [[cách viết 1, cách viết 2], ...]).
    Không tính prompt riêng: sửa nội dung (30 -> 15 ngày) mà prompt còn ghi "30 ngày" thì vẫn phải báo thiếu.
    questions: [{question, must_contain}]."""
    chunks = split_markdown(content_md, slug)
    if not chunks:
        raise ValueError(f'Tài liệu {slug} không có nội dung')
    emb = get_embeddings()
    draft = list(zip(emb.embed_documents([c.page_content for c in chunks]), (c.page_content for c in chunks)))
    get_vector_store()  # tạo collection nếu chưa có
    others = Filter(must_not=[FieldCondition(key='metadata.source', match=MatchValue(value=slug))])
    results = []
    for q in questions:
        qv = emb.embed_query(q['question'])
        # Vector đã chuẩn hóa (normalize_embeddings=True): tích vô hướng = cosine, cùng thang điểm với Qdrant
        scored = [((slug, text), sum(a * b for a, b in zip(qv, v))) for v, text in draft]
        for p in get_client().query_points(COLLECTION, query=qv, limit=POLICY_TOP_K, query_filter=others,
                                           with_payload=True).points:
            scored.append((((p.payload or {}).get('metadata', {}).get('source'), ''), p.score))
        hits = relevant_hits(sorted(scored, key=lambda x: x[1], reverse=True)[:POLICY_TOP_K])
        found = '\n'.join(t for (source, t), _ in hits if source == slug)
        seen = _norm(found)
        missing = [g for g in q.get('must_contain') or [] if not any(_norm(kw) in seen for kw in g)]
        results.append({'question': q['question'], 'retrieved': [source for (source, _), _ in hits],
                        'missing': missing, 'passed': bool(found) and not missing})
    return {'passed': all(r['passed'] for r in results), 'results': results}

def get_retriever(k: int = 3):
    return get_vector_store().as_retriever(search_kwargs={'k': k})

def main():
    from phone_agent.policy_sync import reconcile

    get_vector_store()
    print(reconcile(), flush=True)

    for h in get_retriever().invoke('máy mới mua bị lỗi có được đổi không'):
        print(h.metadata['source'], '|', h.metadata['section'], '|', h.page_content[:60].replace('\n', ' '))

if __name__ == "__main__":
    main()
