import logging
from contextlib import contextmanager
from threading import Lock

import httpx

from phone_agent import backend, store, vector_db

log = logging.getLogger(__name__)
RECONCILE_SECONDS = 3600
LOCK_KEY = 0x706F6C6963790001  # pg_advisory_lock key riêng cho đồng bộ chính sách
_lock = Lock()


@contextmanager
def _locked():
    # Lock trong process; có database thì thêm advisory lock để các replica cũng không đồng bộ chen nhau
    with _lock:
        if store.pool is None:
            yield
            return
        with store.pool.connection() as conn:
            conn.execute('SELECT pg_advisory_lock(%s)', (LOCK_KEY,))
            try:
                yield
            finally:
                conn.execute('SELECT pg_advisory_unlock(%s)', (LOCK_KEY,))


def _refresh(slug: str, present: bool, current_hash: str | None) -> str | None:
    try:
        doc = backend.get(f'/policies/{slug}')
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        if present:
            vector_db.delete_policy(slug)
            return 'removed'
        return None

    # hash do be tính (phiên bản tài liệu): agent chỉ so sánh, không tự tính, nên be đổi cách tính không ảnh hưởng agent
    if doc['hash'] != current_hash:
        vector_db.upsert_policy(slug, doc['content_md'], doc['prompt'], doc['hash'])
        return 'pushed'
    return None


def refresh(slug: str) -> None:
    # Fetch under the same lock as reconciliation so concurrent notifications cannot apply in reverse order.
    with _locked():
        local = vector_db.policy_hashes(slug)
        _refresh(slug, slug in local, local.get(slug))


def reconcile() -> dict[str, list[str]]:
    with _locked():
        published = backend.get('/policies')
        local = vector_db.policy_hashes()
        changes: dict[str, list[str]] = {'pushed': [], 'removed': []}
        for slug in dict.fromkeys([*published, *local]):
            if slug not in published or local.get(slug) != published[slug]:
                # Confirm removals with a fresh GET too, in case BE published after listing.
                result = _refresh(slug, slug in local, local.get(slug))
                if result:
                    changes[result].append(slug)
        return changes
