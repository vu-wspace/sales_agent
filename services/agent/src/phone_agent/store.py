# Trạng thái của agent trong Postgres riêng (AGENT_DATABASE_URL): checkpointer LangGraph (lịch sử, giỏ hàng) và bảng
# agent_threads (lượt đang chạy, lần tương tác cuối) để nhiều process/replica dùng chung.
# pool là None khi không đặt AGENT_DATABASE_URL: service.py giữ các thông tin này trong RAM như trước.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

pool: ConnectionPool | None = None

THREADS_TABLE = """CREATE TABLE IF NOT EXISTS agent_threads (
    thread_id  text PRIMARY KEY,
    last_seen  timestamptz NOT NULL DEFAULT now(),
    busy_since timestamptz)"""
RATE_TABLE = """CREATE TABLE IF NOT EXISTS agent_rate (
    client text NOT NULL,
    minute timestamptz NOT NULL,
    hits   integer NOT NULL,
    PRIMARY KEY (client, minute))"""


def open_checkpointer(url: str):
    global pool
    if not url:
        return InMemorySaver()
    # PostgresSaver cần autocommit + dict_row; prepare_threshold=0 để chạy được sau pgbouncer.
    # connect_timeout: 'localhost' trên Windows thử ::1 trước, không có timeout thì treo quá thời gian chờ của pool
    pool = ConnectionPool(url, min_size=1, max_size=20, open=True,
                          kwargs={'autocommit': True, 'prepare_threshold': 0, 'row_factory': dict_row,
                                  'connect_timeout': 5})
    saver = PostgresSaver(pool)
    saver.setup()
    with pool.connection() as conn:
        conn.execute(THREADS_TABLE)
        conn.execute(RATE_TABLE)
    return saver


def close() -> None:
    global pool
    if pool is not None:
        pool.close()
        pool = None


def touch(thread_id: str) -> None:
    with pool.connection() as conn:
        conn.execute('INSERT INTO agent_threads (thread_id) VALUES (%s) '
                     'ON CONFLICT (thread_id) DO UPDATE SET last_seen = now()', (thread_id,))


def claim(thread_id: str, timeout_s: int) -> bool:
    """Giữ thread cho 1 lượt (atomic giữa các replica). False: lượt khác đang chạy và chưa quá timeout_s."""
    with pool.connection() as conn:
        return conn.execute(
            """INSERT INTO agent_threads (thread_id, busy_since) VALUES (%s, now())
               ON CONFLICT (thread_id) DO UPDATE SET busy_since = now(), last_seen = now()
               WHERE agent_threads.busy_since IS NULL OR agent_threads.busy_since < now() - make_interval(secs => %s)
               RETURNING thread_id""", (thread_id, timeout_s)).fetchone() is not None


def release(thread_id: str) -> None:
    # Upsert: thread bị xóa giữa lượt (lượt vẫn ghi checkpoint) thì có lại dòng, lần hết hạn sau xóa hết
    with pool.connection() as conn:
        conn.execute('INSERT INTO agent_threads (thread_id) VALUES (%s) '
                     'ON CONFLICT (thread_id) DO UPDATE SET busy_since = NULL', (thread_id,))


def forget(thread_id: str) -> None:
    with pool.connection() as conn:
        conn.execute('DELETE FROM agent_threads WHERE thread_id = %s', (thread_id,))


def expired(ttl_s: int, turn_timeout_s: int) -> list[str]:
    """Xóa và trả các thread quá ttl_s không tương tác (bỏ qua thread đang chạy lượt). Mỗi thread chỉ 1 replica nhận."""
    with pool.connection() as conn:
        rows = conn.execute(
            """DELETE FROM agent_threads
               WHERE last_seen < now() - make_interval(secs => %s)
                 AND (busy_since IS NULL OR busy_since < now() - make_interval(secs => %s))
               RETURNING thread_id""", (ttl_s, turn_timeout_s)).fetchall()
        # Checkpoint không còn dòng agent_threads (lỗi giữa 2 bước xóa): trả về để xóa nốt, không giữ dữ liệu mãi.
        # Thread mới luôn có dòng trước checkpoint đầu tiên (touch/claim chạy trước khi ghi state).
        orphans = conn.execute(
            """SELECT DISTINCT c.thread_id FROM checkpoints c
               WHERE NOT EXISTS (SELECT 1 FROM agent_threads t WHERE t.thread_id = c.thread_id)""").fetchall()
    return list(dict.fromkeys([r['thread_id'] for r in rows] + [r['thread_id'] for r in orphans]))


def threads_with_prefix(prefix: str) -> list[str]:
    with pool.connection() as conn:
        # Cả bảng checkpoints: dữ liệu hội thoại nằm ở đó, không để sót thread mất dòng agent_threads
        rows = conn.execute(
            """SELECT thread_id FROM agent_threads WHERE left(thread_id, length(%s)) = %s
               UNION SELECT thread_id FROM checkpoints WHERE left(thread_id, length(%s)) = %s""",
            (prefix, prefix, prefix, prefix)).fetchall()
    return [r['thread_id'] for r in rows]


def hit(client: str) -> int:
    """Đếm 1 request của client trong phút hiện tại (dùng chung giữa các replica), trả số request trong phút đó."""
    with pool.connection() as conn:
        return conn.execute(
            """INSERT INTO agent_rate VALUES (%s, date_trunc('minute', now()), 1)
               ON CONFLICT (client, minute) DO UPDATE SET hits = agent_rate.hits + 1 RETURNING hits""",
            (client,)).fetchone()['hits']


def prune_rate() -> None:
    with pool.connection() as conn:
        conn.execute("DELETE FROM agent_rate WHERE minute < now() - interval '5 minutes'")
