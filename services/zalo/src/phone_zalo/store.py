# SQLite riêng của adapter Zalo. Một kết nối dùng chung, khóa bằng Lock (adapter chạy 1 process).
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,    -- id người dùng Zalo (theo OA)
    thread_id      TEXT,                -- cuộc trò chuyện trên be (uuid be cấp ở /session, không đoán được)
    customer_token TEXT,                -- mã khách quen be trả sau khi đặt hàng
    checkout       TEXT,                -- JSON: đang chờ khách gửi thông tin nhận hàng
    handoff_until  REAL,                -- đang chuyển nhân viên: không tự trả lời tới lúc này
    seen_at        REAL NOT NULL);
CREATE TABLE IF NOT EXISTS seen_messages (msg_id TEXT PRIMARY KEY, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def user(self, user_id: str) -> dict:
        with self.lock:
            row = self.conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
        if row is None:
            return {'user_id': user_id, 'thread_id': None, 'customer_token': None, 'checkout': None,
                    'handoff_until': None, 'seen_at': 0}
        user = dict(row)
        user['checkout'] = json.loads(user['checkout']) if user['checkout'] else None
        return user

    def save_user(self, user: dict) -> None:
        with self.lock:
            self.conn.execute(
                """INSERT INTO users VALUES (:user_id, :thread_id, :customer_token, :checkout, :handoff_until, :seen_at)
                   ON CONFLICT (user_id) DO UPDATE SET thread_id = excluded.thread_id,
                       customer_token = excluded.customer_token, checkout = excluded.checkout,
                       handoff_until = excluded.handoff_until, seen_at = excluded.seen_at""",
                # {} (đang chờ thông tin nhận hàng) khác None (không đặt hàng): không dùng "if user['checkout']"
                {**user, 'checkout': json.dumps(user['checkout']) if user['checkout'] is not None else None})

    def first_time(self, msg_id: str) -> bool:
        """Zalo có thể gửi lại cùng 1 sự kiện: chỉ xử lý lần đầu. Giữ id 1 ngày."""
        now = time.time()
        with self.lock:
            self.conn.execute('DELETE FROM seen_messages WHERE at < ?', (now - 86400,))
            return self.conn.execute('INSERT OR IGNORE INTO seen_messages VALUES (?, ?)',
                                     (msg_id, now)).rowcount == 1

    def get(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute('SELECT v FROM kv WHERE k = ?', (key,)).fetchone()
        return row['v'] if row else None

    def set(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute('INSERT INTO kv VALUES (?, ?) ON CONFLICT (k) DO UPDATE SET v = excluded.v', (key, value))
