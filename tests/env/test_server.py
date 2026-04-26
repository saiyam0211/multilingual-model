"""End-to-end env smoke. Uses MockTarget (no network)."""
from fastapi.testclient import TestClient

from polyglot_redteam.server import api


def test_health_returns_status():
    with TestClient(api) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in {"healthy", "degraded"}


def test_reset_then_step_round_trip():
    with TestClient(api) as c:
        r1 = c.post("/reset", json={"seed": 42})
        assert r1.status_code == 200
        spec = r1.json()
        assert spec["target_lang"] in {"hi", "ta", "bn", "mr", "te", "kn"}

        r2 = c.post("/step", json={
            "episode_id": spec["episode_id"],
            "action": "नमस्ते दुनिया, यह एक परीक्षण है।",
        })
        assert r2.status_code == 200
        result = r2.json()
        assert "reward" in result
        assert "info" in result
        assert "reward_components" in result["info"]


def test_reset_determinism():
    """Same seed must produce same episode."""
    with TestClient(api) as c:
        r1 = c.post("/reset", json={"seed": 7})
        r2 = c.post("/reset", json={"seed": 7})
    a, b = r1.json(), r2.json()
    # episode_id is uuid4 derived from seeded RNG — should match
    assert a["target_lang"] == b["target_lang"]
    assert a["category"] == b["category"]
    assert a["instruction"] == b["instruction"]


def test_unknown_episode_404():
    with TestClient(api) as c:
        r = c.post("/step", json={"episode_id": "nope", "action": "foo bar baz qux"})
    assert r.status_code == 404

