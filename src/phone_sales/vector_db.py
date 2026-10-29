import phone_sales  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
import uuid
from functools import lru_cache
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pyvi import ViTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FilterSelector, VectorParams
from sentence_transformers import SentenceTransformer
from phone_sales.config import COLLECTION, EMBEDDING_MODEL, QDRANT_URL, VECTOR_SIZE
from phone_sales.data_prep import load_policy_chunks

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
    # id tính từ cả nội dung chunk: sửa file .md -> id đổi -> so với id trong Qdrant là biết dữ liệu đã cũ
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f'{c.source}#{i}#{c.page_content}')) for i, c in enumerate(chunks)]

def stored_ids() -> set[str]:
    ids, offset = set(), None
    while True:
        points, offset = get_client().scroll(COLLECTION, limit=256, offset=offset,
                                             with_payload=False, with_vectors=False)
        ids |= {str(p.id) for p in points}
        if offset is None:
            return ids

def is_up_to_date() -> bool:
    return stored_ids() == set(chunk_ids(load_policy_chunks()))

def ingest(db: QdrantVectorStore) -> int:
    chunks = load_policy_chunks()
    docs = [
        Document(page_content=c.page_content,
                 metadata={'source': c.source, 'title': c.title, 'section': c.section})
        for c in chunks
    ]
    # Xóa chunk cũ trước khi nạp: sửa file .md làm số chunk/nội dung đổi, giữ lại sẽ trả về chính sách cũ
    get_client().delete(COLLECTION, points_selector=FilterSelector(filter=Filter()))
    db.add_documents(docs, ids=chunk_ids(chunks))
    return len(docs)

def get_retriever(k: int = 3):
    return get_vector_store().as_retriever(search_kwargs={'k': k})

def main():
    print('1. Đang kết nối Qdrant + tải model...', flush=True)
    db = get_vector_store()

    print('2. Đang nạp dữ liệu...', flush=True)
    n = ingest(db)
    print(f"3. Đã nạp {n} chunk chính sách vào '{COLLECTION}'", flush=True)

    for h in get_retriever().invoke('máy mới mua bị lỗi có được đổi không'):
        print(h.metadata['source'], '|', h.metadata['section'], '|', h.page_content[:60].replace('\n', ' '))

if __name__ == "__main__":
    main()
