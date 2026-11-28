# Trạng thái của agent trong Postgres riêng (store.py): hội thoại còn sau restart, giữ lượt / hết hạn đúng giữa các process.
import operator
import re
import uuid
from typing import Annotated, TypedDict

import psycopg
import pytest
from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph
from phone_agent import policy_sync, store
from phone_agent import service as svc
from phone_api.config import DATABASE_URL


@pytest.fixture
def agent_db():
    # Database tạm trên Postgres của docker compose; không có Postgres thì bỏ qua
    name = f'agent_test_{uuid.uuid4().hex[:8]}'
    admin_url = re.sub(r'/[^/]*$', '/postgres', DATABASE_URL)
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=2) as conn:
            conn.execute(f'CREATE DATABASE {name}')
    except psycopg.OperationalError:
        pytest.skip('Không có Postgres (docker compose up -d postgres)')
    # 127.0.0.1: 'localhost' trên Windows thử ::1 trước, mỗi kết nối mất thêm connect_timeout
    url = re.sub(r'/[^/]*$', f'/{name}', DATABASE_URL).replace('@localhost:', '@127.0.0.1:')
    yield url
    store.close()
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE {name} WITH (FORCE)')


class State(TypedDict):
    log: Annotated[list, operator.add]


def _graph(checkpointer):
    builder = StateGraph(State)
    builder.add_node('echo', lambda s: {'log': ['bot']})
    builder.add_edge(START, 'echo')
    builder.add_edge('echo', END)
    return builder.compile(checkpointer=checkpointer)


def test_conversation_survives_restart(agent_db):
    config = {'configurable': {'thread_id': 't1'}}
    _graph(store.open_checkpointer(agent_db)).invoke({'log': ['khách']}, config)
    store.close()                                   # "restart" process
    g = _graph(store.open_checkpointer(agent_db))
    assert g.get_state(config).values['log'] == ['khách', 'bot']
    g.checkpointer.delete_thread('t1')
    assert g.get_state(config).values == {}


def test_claim_release_and_expire_are_shared(agent_db):
    store.open_checkpointer(agent_db)
    svc._claim('t1')
    with pytest.raises(HTTPException) as e:        # replica khác cũng thấy lượt đang chạy
        svc._claim('t1')
    assert e.value.status_code == 409
    svc._release('t1')
    svc._claim('t1')
    svc._release('t1')

    svc._touch('idle')
    svc._touch('active')
    with store.pool.connection() as conn:
        conn.execute("UPDATE agent_threads SET last_seen = now() - interval '1 hour' WHERE thread_id IN ('idle', 't1')")
        # lượt đang chạy thì không xóa dù lâu không tương tác
        conn.execute("UPDATE agent_threads SET busy_since = now() WHERE thread_id = 't1'")

    class Agent:
        class checkpointer:
            deleted = []
            delete_thread = deleted.append

    assert svc.expire_threads(Agent) == ['idle']
    assert Agent.checkpointer.deleted == ['idle']
    assert svc.expire_threads(Agent) == []          # replica khác không nhận lại thread đã xóa


def test_policy_sync_lock_is_reentrant_per_call(agent_db):
    store.open_checkpointer(agent_db)
    for _ in range(2):                              # khóa được nhả sau mỗi lần
        with policy_sync._locked():
            with store.pool.connection() as conn:
                assert conn.execute('SELECT count(*) AS n FROM pg_locks WHERE locktype = %s',
                                    ('advisory',)).fetchone()['n'] == 1
