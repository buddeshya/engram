from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from api.dependencies import (
    get_memory_service,
    get_message_repo,
    get_session_repo,
    get_session_service,
    get_user_id,
)
from api.main import app
from memory.types import MemoryStatus, MemoryType


def _fake_session(user_id):
    return {
        "id": uuid4(),
        "user_id": user_id,
        "title": "Demo session",
        "summary": None,
        "summary_turn_count": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def _fake_message(session_id, role, content):
    return {
        "id": uuid4(),
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }


def _fake_memory(user_id):
    return SimpleNamespace(
        id=uuid4(),
        user_id=user_id,
        type=MemoryType.FACT,
        content="Uses Python",
        confidence=0.9,
        status=MemoryStatus.ACTIVE,
        source_session_id=None,
        supersedes_id=None,
        metadata={},
        access_count=0,
        last_accessed_at=None,
        created_at=datetime.now(timezone.utc),
    )


def test_sessions_list_and_messages_route():
    user_id = uuid4()
    session = _fake_session(user_id)
    message = _fake_message(session["id"], "user", "hello")
    session_repo = SimpleNamespace(
        list_by_user=AsyncMock(return_value=[session]),
        get_by_id=AsyncMock(return_value=session),
        create=AsyncMock(return_value=session),
    )
    message_repo = SimpleNamespace(get_by_session=AsyncMock(return_value=[message]))

    app.dependency_overrides[get_user_id] = lambda: user_id
    app.dependency_overrides[get_session_repo] = lambda: session_repo
    app.dependency_overrides[get_message_repo] = lambda: message_repo

    try:
        client = TestClient(app)
        list_resp = client.get("/sessions")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        msg_resp = client.get(f"/sessions/{session['id']}/messages")
        assert msg_resp.status_code == 200
        assert msg_resp.json()[0]["content"] == "hello"
    finally:
        app.dependency_overrides.clear()


def test_memories_list_update_and_forget_routes():
    user_id = uuid4()
    memory = _fake_memory(user_id)
    memory_service = SimpleNamespace(
        list_active=AsyncMock(return_value={"memories": [memory], "total": 1, "by_type": {"fact": 1}}),
        update_content=AsyncMock(return_value=memory),
        forget=AsyncMock(return_value=None),
    )

    app.dependency_overrides[get_user_id] = lambda: user_id
    app.dependency_overrides[get_memory_service] = lambda: memory_service

    try:
        client = TestClient(app)
        list_resp = client.get("/memories")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] == 1

        patch_resp = client.patch(f"/memories/{memory.id}", json={"content": "Uses Python and FastAPI"})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["content"] == "Uses Python"

        del_resp = client.delete(f"/memories/{memory.id}")
        assert del_resp.status_code == 204
    finally:
        app.dependency_overrides.clear()


def test_end_session_route_triggers_service():
    user_id = uuid4()
    session = _fake_session(user_id)
    session_repo = SimpleNamespace(get_by_id=AsyncMock(return_value=session))
    session_service = SimpleNamespace(generate_episodic_memory=AsyncMock(return_value=None))

    app.dependency_overrides[get_user_id] = lambda: user_id
    app.dependency_overrides[get_session_repo] = lambda: session_repo
    app.dependency_overrides[get_session_service] = lambda: session_service

    try:
        client = TestClient(app)
        resp = client.post(f"/sessions/{session['id']}/end", json={"generate_episodic": True})
        assert resp.status_code == 202
        assert session_service.generate_episodic_memory.await_count == 1
    finally:
        app.dependency_overrides.clear()
