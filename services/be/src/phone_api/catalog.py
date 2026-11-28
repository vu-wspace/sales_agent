import asyncio
import secrets
from typing import Annotated, Literal

import psycopg
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from phone_common import promotions
from phone_common.config import CATALOG_API_KEY

from phone_api import orders


async def require_catalog_key(authorization: Annotated[str | None, Header()] = None) -> None:
    if CATALOG_API_KEY and not secrets.compare_digest(
        (authorization or '').encode(), f'Bearer {CATALOG_API_KEY}'.encode()
    ):
        raise HTTPException(status_code=401, detail='Invalid catalog API key')


router = APIRouter(dependencies=[Depends(require_catalog_key)])


def _read_rows(sql: str, params: tuple = ()) -> list[dict]:
    try:
        with orders._connect() as conn:
            return conn.execute(sql, params).fetchall()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Catalog database unavailable') from exc


async def _query(sql: str, params: tuple = ()) -> list[dict]:
    # Chat requests can occupy Starlette's worker pool while Agent calls back into BE.
    return await asyncio.to_thread(_read_rows, sql, params)


async def summary(category: str = 'phone', brand: str = ''):
    return await _query(
        """SELECT p.product_id, p.name, p.brand, min(v.price) AS min_price, max(v.price) AS max_price,
                  sum(v.stock) AS stock, array_agg(DISTINCT v.storage) AS storages
           FROM products p JOIN variants v USING (product_id)
           WHERE p.category = %s AND (%s = '' OR p.brand ILIKE %s)
           GROUP BY p.product_id, p.name, p.brand ORDER BY min(v.price)""",
        (category, brand, brand),
    )


@router.get('/products')
async def products(category: str | None = None, brand: str = '',
                   view: Literal['summary', 'phone-variants'] | None = None):
    if view == 'summary':
        return await summary(category or 'phone', brand)
    if view == 'phone-variants':
        return await phone_variants()
    return await _query(
        """SELECT product_id, name, brand, category, description, attributes FROM products
           WHERE (%s::text IS NULL OR category = %s)""",
        (category, category),
    )


@router.get('/products/{product_id}/variants')
async def variants(product_id: str):
    return await _query(
        'SELECT variant_id, color, storage, price, stock FROM variants WHERE product_id = %s', (product_id,)
    )


@router.get('/internal/catalog/accessories')
async def accessories(phone_id: str, kinds: Annotated[list[str] | None, Query()] = None):
    return await _query(
        """SELECT a.name, a.brand, a.attributes->>'accessory_type' AS kind, v.price, v.stock
           FROM compatibility c JOIN products a ON a.product_id = c.accessory_id
           JOIN variants v ON v.product_id = a.product_id
           WHERE c.phone_id = %s AND (%s::text[] IS NULL OR a.attributes->>'accessory_type' = ANY(%s))
           ORDER BY v.stock = 0, v.price""",
        (phone_id, kinds, kinds),
    )


async def phone_variants():
    return await _query(
        """SELECT p.product_id, p.name, p.brand, v.storage, v.color, v.price, v.stock
           FROM products p JOIN variants v USING (product_id) WHERE p.category = 'phone'"""
    )


@router.get('/internal/catalog/compatible')
async def compatible(phone_ids: Annotated[list[str], Query()], accessory_ids: Annotated[list[str], Query()]):
    return await _query(
        'SELECT accessory_id FROM compatibility WHERE phone_id = ANY(%s) AND accessory_id = ANY(%s)',
        (phone_ids, accessory_ids),
    )


def _read_promotions():
    try:
        with orders._connect() as conn:
            return [
                {'scope': row['scope'], 'scope_value': row['scope_value'], 'summary': promotions.describe(row)}
                for row in promotions.active(conn)
            ]
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail='Catalog database unavailable') from exc


@router.get('/promotions')
async def active_promotions():
    return await asyncio.to_thread(_read_promotions)
