"""Tests for sqlrl.serving.

No GPU, no model, no checkpoint on disk. The FastAPI tests inject a fake
``SqlService`` via ``app.dependency_overrides`` -- see ``client`` below --
which only works because importing ``sqlrl.serving.api`` never builds a real
one. That property gets its own explicit test, since every other test in this
file depends on it being true.
"""

from __future__ import annotations

import dataclasses
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from sqlrl.serving.bench import Mode, parse_mode
from sqlrl.serving.service import CALIBRATION, Answer, Confidence, confidence

# --------------------------------------------------------------------------
# CALIBRATION / confidence()
# --------------------------------------------------------------------------


def test_calibration_table_shape():
    # (threshold, level, expected_accuracy), highest threshold first.
    assert CALIBRATION[0][1:] == ("high", 0.845)
    assert CALIBRATION[1][1:] == ("medium", 0.561)
    assert CALIBRATION[2][1:] == ("low", 0.440)


def test_unanimous_agreement_is_high():
    c = confidence(8, 8)
    assert c.level == "high"
    assert c.expected_accuracy == 0.845
    assert c.agreement == 8
    assert c.samples == 8


def test_seven_of_eight_is_medium():
    c = confidence(7, 8)
    assert c.level == "medium"
    assert c.expected_accuracy == 0.561


def test_four_of_eight_is_low():
    c = confidence(4, 8)
    assert c.level == "low"
    assert c.expected_accuracy == 0.440


def test_nothing_executed_is_none():
    c = confidence(0, 8)
    assert c.level == "none"
    assert c.expected_accuracy == 0.0


def test_single_greedy_sample_is_unmeasured_not_invented():
    c = confidence(1, 1)
    assert c.level == "unmeasured"
    assert c.expected_accuracy is None


def test_bucket_function_is_monotonic_in_agreement_fraction():
    for samples in (4, 8, 16):
        accuracies = [confidence(agreement, samples).expected_accuracy for agreement in range(samples + 1)]
        assert all(a is not None for a in accuracies)
        assert accuracies == sorted(accuracies), (
            f"higher agreement mapped to lower expected accuracy at samples={samples}"
        )


# --------------------------------------------------------------------------
# JSON serialisation
# --------------------------------------------------------------------------


def test_answer_and_confidence_serialise_to_json_cleanly():
    answer = Answer(
        sql="SELECT name FROM singer",
        rows=[["adele"], ["beyonce"]],
        status="ok",
        error=None,
        confidence=Confidence(agreement=4, samples=8, level="low", expected_accuracy=0.440),
        attempts=1,
        timings_ms={"generate": 812.3, "execute": 1.4, "total": 813.7},
    )

    payload = json.dumps(dataclasses.asdict(answer))
    restored = json.loads(payload)

    assert restored["sql"] == "SELECT name FROM singer"
    assert restored["rows"] == [["adele"], ["beyonce"]]
    assert restored["confidence"] == {
        "agreement": 4, "samples": 8, "level": "low", "expected_accuracy": 0.440,
    }
    assert restored["timings_ms"]["total"] == 813.7


def test_unmeasured_confidence_serialises_null_expected_accuracy():
    answer = Answer(
        sql="SELECT 1", rows=[[1]], status="ok", error=None,
        confidence=confidence(1, 1), attempts=1,
        timings_ms={"generate": 10.0, "execute": 0.1, "total": 10.1},
    )
    restored = json.loads(json.dumps(dataclasses.asdict(answer)))
    assert restored["confidence"]["expected_accuracy"] is None
    assert restored["confidence"]["level"] == "unmeasured"


# --------------------------------------------------------------------------
# bench.parse_mode
# --------------------------------------------------------------------------


def test_parse_mode_greedy():
    assert parse_mode("greedy") == Mode("greedy", samples=1, max_attempts=1)


def test_parse_mode_vote():
    assert parse_mode("vote8") == Mode("vote8", samples=8, max_attempts=1)


def test_parse_mode_retry():
    assert parse_mode("retry3") == Mode("retry3", samples=1, max_attempts=3)


def test_parse_mode_rejects_unknown():
    with pytest.raises(ValueError):
        parse_mode("bogus")


# --------------------------------------------------------------------------
# importing the API must not construct a model
# --------------------------------------------------------------------------


def test_importing_api_does_not_construct_a_model(monkeypatch):
    """The whole point of building the service in the lifespan handler
    instead of at import time -- see sqlrl.serving.api's module docstring.
    Every other test below depends on this being true.
    """
    import sqlrl.serving.api as api_module
    import sqlrl.serving.service as service_module

    def _boom(self, *args, **kwargs):
        raise AssertionError("SqlService was constructed while importing the module")

    monkeypatch.setattr(service_module.SqlService, "__init__", _boom)
    # Re-execute the module's top-level code under the patched constructor --
    # if anything at import time built a SqlService, this raises.
    importlib.reload(api_module)


# --------------------------------------------------------------------------
# FastAPI endpoints, against a fake service
# --------------------------------------------------------------------------


class _FakeBackend:
    device = "cpu"


class FakeSqlService:
    """Stands in for SqlService: same public surface api.py touches, no
    torch, no GPU, no checkpoint.
    """

    model_name = "fake-model"
    backend = _FakeBackend()

    def __init__(self, db_ids: list[str]) -> None:
        self._db_ids = db_ids

    @property
    def databases(self) -> list[str]:
        return list(self._db_ids)

    def answer(self, question: str, db_id: str, *, samples: int = 1, max_attempts: int = 1,
               temperature: float = 0.8) -> Answer:
        return Answer(
            sql="SELECT count(*) FROM singer",
            rows=[[42]],
            status="ok",
            error=None,
            confidence=confidence(samples, samples) if samples > 1 else confidence(1, 1),
            attempts=max_attempts,
            timings_ms={"generate": 5.0, "execute": 0.5, "total": 5.5},
        )


@pytest.fixture
def fake_service() -> FakeSqlService:
    return FakeSqlService(["concert_singer", "world_1"])


@pytest.fixture
def client(fake_service: FakeSqlService):
    # Imported inside the fixture, not module scope: test_importing_api_...
    # above reloads sqlrl.serving.api, and importing fresh here keeps this
    # fixture bound to whichever module object is current.
    from sqlrl.serving.api import app, get_service

    app.dependency_overrides[get_service] = lambda: fake_service
    try:
        # Deliberately not `with TestClient(app) as client` -- that would run
        # the lifespan handler and build a real, GPU-bound SqlService. Plain
        # construction never triggers startup/shutdown events.
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok", "model": "fake-model", "device": "cpu", "databases": 2,
    }


def test_databases(client: TestClient):
    response = client.get("/databases")
    assert response.status_code == 200
    assert response.json() == ["concert_singer", "world_1"]


def test_query_happy_path(client: TestClient):
    response = client.post(
        "/query", json={"question": "how many singers?", "db_id": "concert_singer"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sql"] == "SELECT count(*) FROM singer"
    assert body["status"] == "ok"
    assert body["confidence"]["level"] == "unmeasured"


def test_query_with_voting_reaches_the_fake_service(client: TestClient):
    response = client.post(
        "/query",
        json={"question": "how many singers?", "db_id": "concert_singer", "samples": 8},
    )
    assert response.status_code == 200
    assert response.json()["confidence"]["level"] == "high"


def test_query_unknown_db_id_returns_404_listing_valid_ones(client: TestClient):
    response = client.post("/query", json={"question": "q", "db_id": "nope"})
    assert response.status_code == 404
    assert "concert_singer" in response.json()["detail"]
    assert "world_1" in response.json()["detail"]


@pytest.mark.parametrize("samples", [0, 99])
def test_query_samples_out_of_range_is_422(client: TestClient, samples: int):
    response = client.post(
        "/query", json={"question": "q", "db_id": "concert_singer", "samples": samples}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("max_attempts", [0, 6])
def test_query_max_attempts_out_of_range_is_422(client: TestClient, max_attempts: int):
    response = client.post(
        "/query",
        json={"question": "q", "db_id": "concert_singer", "max_attempts": max_attempts},
    )
    assert response.status_code == 422
