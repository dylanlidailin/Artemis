from fastapi.testclient import TestClient

from api import SESSION_STORE, app


client = TestClient(app)


def setup_function():
    SESSION_STORE.clear()


def test_healthz():
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_missing_session_message():
    response = client.post(
        "/ask",
        json={"question": "What is this PDF about?", "session_id": "missing"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Session not found or expired."}


def test_extract_keywords_returns_missing_session_message():
    response = client.post(
        "/extract_keywords",
        json={"question": "extract keywords", "session_id": "missing"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Session not found or expired."}
