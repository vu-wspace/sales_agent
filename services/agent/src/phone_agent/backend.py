import httpx
from phone_common.config import BE_URL, CATALOG_API_KEY


def get(path: str, **params):
    headers = {'Authorization': f'Bearer {CATALOG_API_KEY}'} if CATALOG_API_KEY else {}
    response = httpx.get(
        f"{BE_URL.rstrip('/')}{path}",
        params={key: value for key, value in params.items() if value is not None},
        headers=headers, timeout=5, trust_env=False,
    )
    response.raise_for_status()
    return response.json()
