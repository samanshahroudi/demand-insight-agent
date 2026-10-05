import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from demand_insight_agent.api import app
from demand_insight_agent.models import RunRequest


def test_empty_shortlist_is_rejected_by_shared_request_model():
    with pytest.raises(ValidationError, match="at least one product_id"):
        RunRequest(mode="shortlist")
    response = TestClient(app).post("/runs", json={"mode": "shortlist"})
    assert response.status_code == 422
    assert "at least one product_id" in response.json()["detail"][0]["msg"]


def test_discovery_and_valid_shortlist_still_run():
    client = TestClient(app)
    discovery = client.post("/runs", json={"mode": "discover", "limit": 1})
    assert discovery.status_code == 200
    assert len(discovery.json()["assessments"]) == 1
    shortlist = client.post("/runs", json={"mode": "shortlist", "product_ids": ["sample-001"]})
    assert shortlist.status_code == 200
    assert [item["product_id"] for item in shortlist.json()["assessments"]] == ["sample-001"]


@pytest.mark.parametrize("product_id", ["", " ", "\t\n"])
def test_blank_product_ids_are_rejected_before_graph_execution(monkeypatch, product_id):
    with pytest.raises(ValidationError):
        RunRequest(mode="shortlist", product_ids=["sample-001", product_id])
    monkeypatch.setattr("demand_insight_agent.api.run",
                        lambda request: pytest.fail("invalid IDs must not execute the graph"))
    response = TestClient(app).post("/runs", json={
        "mode": "shortlist", "product_ids": ["sample-001", product_id]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "product_ids", 1]
