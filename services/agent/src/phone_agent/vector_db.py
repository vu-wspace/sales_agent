import phone_common  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
import uuid
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pyvi import ViTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from phone_common.config import COLLECTION, EMBEDDING_MODEL, QDRANT_URL, VECTOR_SIZE
from phone_agent.data_prep import split_markdown
from phone_common.policies import load_seed, policy_hash

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
    lưu trong payload từng chunk: search_policy đọc prompt từ đây, be đọc hash (policy_hashes) để biết đã đồng bộ chưa."""
    chunks = split_markdown(content_md, slug)
    if not chunks:
        raise ValueError(f'Tài liệu {slug} không có nội dung')
    meta = {'source': slug, 'prompt': prompt, 'doc_hash': policy_hash(content_md, prompt)}
    texts = [c.page_content for c in chunks]
    get_vector_store()  # tạo collection nếu chưa có
    # Embed trước rồi mới xóa: embed lỗi thì chunk cũ vẫn còn. Xóa chunk cũ vì sửa tài liệu làm số chunk/nội dung đổi.
    # Payload theo dạng QdrantVectorStore đọc (page_content + metadata) để search_policy dùng được ngay
    vectors = get_embeddings().embed_documents(texts)
    points = [PointStruct(id=i, vector=v, payload={'page_content': t, 'metadata': {
                  **meta, 'title': c.title, 'section': c.section}})
              for i, v, t, c in zip(chunk_ids(chunks), vectors, texts, chunks)]
    get_client().delete(COLLECTION, points_selector=_of(slug))
    get_client().upsert(COLLECTION, points=points)
    return len(points)

def delete_policy(slug: str) -> None:
    get_client().delete(COLLECTION, points_selector=_of(slug))

def policy_hashes() -> dict[str, str]:
    """{slug: hash} của các tài liệu đang có trong Qdrant (mỗi chunk mang hash của tài liệu)."""
    hashes, offset = {}, None
    while True:
        points, offset = get_client().scroll(COLLECTION, limit=256, offset=offset, with_vectors=False,
                                             with_payload=['metadata.source', 'metadata.doc_hash'])
        for p in points:
            m = (p.payload or {}).get('metadata') or {}
            hashes[m.get('source')] = m.get('doc_hash')
        if offset is None:
            return hashes

def ingest_seed() -> int:
    """Nạp dữ liệu mẫu (data_schema/policies) thẳng vào Qdrant, khi chạy agent không có be. Có be thì be đẩy sang."""
    seed = load_seed()
    for slug in set(policy_hashes()) - {d['slug'] for d in seed}:
        delete_policy(slug)
    return sum(upsert_policy(d['slug'], d['content_md'], d['prompt']) for d in seed)

def get_retriever(k: int = 3):
    return get_vector_store().as_retriever(search_kwargs={'k': k})

def main():
    print('1. Đang kết nối Qdrant + tải model...', flush=True)
    get_vector_store()

    print('2. Đang nạp dữ liệu mẫu (data_schema/policies)...', flush=True)
    n = ingest_seed()
    print(f"3. Đã nạp {n} chunk chính sách vào '{COLLECTION}'", flush=True)

    for h in get_retriever().invoke('máy mới mua bị lỗi có được đổi không'):
        print(h.metadata['source'], '|', h.metadata['section'], '|', h.page_content[:60].replace('\n', ' '))

if __name__ == "__main__":
    main()
