from atexit import register

import httpx
from phone_agent.config import BE_URL, CATALOG_API_KEY

# HTTPX Client is safe to share between the chat and reconciliation workers.
# Drop idle connections before BE's 5s uvicorn keep-alive closes them, avoiding RemoteProtocolError on reuse.
_client = httpx.Client(base_url=BE_URL.rstrip('/'), timeout=5, trust_env=False,
                       limits=httpx.Limits(keepalive_expiry=2),
                       headers={'Authorization': f'Bearer {CATALOG_API_KEY}'} if CATALOG_API_KEY else {})
register(_client.close)


def get(path: str, **params):
    response = _client.get(path, params={key: value for key, value in params.items() if value is not None})
    response.raise_for_status()
    return response.json()
