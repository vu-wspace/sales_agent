import logging
from threading import Lock

import httpx
from phone_common.policies import policy_hash

from phone_agent import backend, vector_db

log = logging.getLogger(__name__)
RECONCILE_SECONDS = 3600
_lock = Lock()


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

    if policy_hash(doc['content_md'], doc['prompt']) != current_hash:
        vector_db.upsert_policy(slug, doc['content_md'], doc['prompt'])
        return 'pushed'
    return None


def refresh(slug: str) -> None:
    # Fetch under the same lock as reconciliation so concurrent notifications cannot apply in reverse order.
    with _lock:
        local = vector_db.policy_hashes()
        _refresh(slug, slug in local, local.get(slug))


def reconcile() -> dict[str, list[str]]:
    with _lock:
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
