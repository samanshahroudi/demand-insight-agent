import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from demand_insight_agent.api import app
from demand_insight_agent.models import RunRequest


@pytest.mark.parametrize("mode", ["discover", "shortlist"])
@pytest.mark.parametrize("catalog", [None, {}, "catalog", 7, False])
def test_nonlist_catalog_is_rejected_before_selection(monkeypatch, mode, catalog):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    monkeypatch.setattr(graph_module, "_catalog", lambda: catalog)
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("invalid catalogs must not request a summary"))
    response = TestClient(app).post("/runs", json={
        "mode": mode, "product_ids": ["sample-001"], "use_llm_summary": True,
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "Catalog must be a JSON list."}


def test_empty_catalog_still_returns_a_successful_empty_discovery(monkeypatch):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    monkeypatch.setattr(graph_module, "_catalog", list)
    response = TestClient(app).post("/runs", json={"mode": "discover"})
    assert response.status_code == 200
    assert response.json()["assessments"] == []
    assert response.json()["summary"] == "No products matched this run."


@pytest.mark.parametrize("mode", ["discover", "shortlist"])
@pytest.mark.parametrize("problem", ["missing_id", "unhashable_id", "non_object"])
def test_malformed_catalog_is_rejected_before_selection(monkeypatch, mode, problem):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    valid = graph_module._catalog()[0]
    invalid = dict(valid)
    if problem == "missing_id":
        del invalid["product_id"]
    elif problem == "unhashable_id":
        invalid["product_id"] = [valid["product_id"]]
    else:
        invalid = None
    monkeypatch.setattr(graph_module, "_catalog", lambda: [valid, invalid])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("invalid catalogs must not request a summary"))
    response = TestClient(app).post("/runs", json={
        "mode": mode, "product_ids": [valid["product_id"]], "use_llm_summary": True,
    })
    assert response.status_code == 422
    assert "detail" in response.json()
    assert "assessments" not in response.json()


@pytest.mark.parametrize("field", ["name", "category"])
@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_catalog_labels_prevent_results_and_model_calls(monkeypatch, field, value):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    item = {**graph_module._catalog()[0], field: value}
    monkeypatch.setattr(graph_module, "_catalog", lambda: [item])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("blank labels must not request a summary"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 422
    assert field in response.json()["detail"]
    assert "assessments" not in response.json()


@pytest.mark.parametrize("mode", ["discover", "shortlist"])
def test_duplicate_catalog_ids_cannot_produce_results_or_model_summaries(monkeypatch, mode):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    item = graph_module._catalog()[0]
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(graph_module, "_catalog", lambda: [item, {**item, "name": "Duplicate"}])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("duplicate IDs must not request a summary"))
    response = TestClient(app).post("/runs", json={
        "mode": mode, "product_ids": [item["product_id"]], "limit": 1,
        "use_llm_summary": True,
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "Catalog product IDs must be unique."}


def test_shortlist_cannot_hide_duplicate_ids_outside_selected_products(monkeypatch):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    selected, other = graph_module._catalog()[:2]
    monkeypatch.setattr(graph_module, "_catalog", lambda: [selected, other, dict(other)])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("invalid catalogs must not request a summary"))
    response = TestClient(app).post("/runs", json={
        "mode": "shortlist", "product_ids": [selected["product_id"]],
        "use_llm_summary": True,
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "Catalog product IDs must be unique."}


@pytest.mark.parametrize("limit", [True, False, "5", 5.0, 5.5])
def test_limit_requires_integer_before_graph_execution(monkeypatch, limit):
    with pytest.raises(ValidationError):
        RunRequest(limit=limit)
    monkeypatch.setattr("demand_insight_agent.api.run",
                        lambda request: pytest.fail("invalid limits must not execute the graph"))
    response = TestClient(app).post("/runs", json={"limit": limit})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "limit"]


@pytest.mark.parametrize("limit", [1, 20])
def test_integer_limit_boundaries_run_successfully(limit):
    assert RunRequest(limit=limit).limit == limit
    response = TestClient(app).post("/runs", json={"limit": limit})
    assert response.status_code == 200
    assert 0 < len(response.json()["assessments"]) <= limit


@pytest.mark.parametrize("flag", ["true", "false", "yes", "0", 0, 1, None])
def test_summary_opt_in_requires_boolean_before_graph_execution(monkeypatch, flag):
    with pytest.raises(ValidationError):
        RunRequest(use_llm_summary=flag)
    monkeypatch.setattr("demand_insight_agent.api.run",
                        lambda request: pytest.fail("invalid opt-in must not execute the graph"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": flag})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "use_llm_summary"]


@pytest.mark.parametrize("flag", [True, False])
def test_summary_opt_in_preserves_boolean_values(flag):
    assert RunRequest(use_llm_summary=flag).use_llm_summary is flag


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


@pytest.mark.parametrize("product_ids", [["unknown"], ["sample-001", "unknown"]])
def test_unknown_shortlist_ids_prevent_partial_results_and_model_calls(monkeypatch, product_ids):
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("unknown IDs must not request a summary"))
    response = TestClient(app).post("/runs", json={
        "mode": "shortlist", "product_ids": product_ids, "use_llm_summary": True,
    })
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown product IDs: unknown"}


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


@pytest.mark.parametrize("mode", ["discover", "shortlist"])
def test_tied_scores_use_product_ids_before_applying_limit(monkeypatch, mode):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    base = {"category": "demo", "demand_signal": 1, "advertiser_count": 8,
            "gross_margin": 0.6, "market_saturation": 0, "rating": 5, "evidence_count": 15}
    catalog = [{**base, "product_id": key, "name": key} for key in ["c", "a", "b"]]
    client = TestClient(app)
    payload = {"mode": mode, "limit": 2, "product_ids": ["c", "b", "a"]}
    for order in [catalog, list(reversed(catalog))]:
        monkeypatch.setattr(graph_module, "_catalog", lambda order=order: order)
        response = client.post("/runs", json=payload)
        assert response.status_code == 200
        assert [item["product_id"] for item in response.json()["assessments"]] == ["a", "b"]
        assert [item["score"] for item in response.json()["assessments"]] == [100, 100]


@pytest.mark.parametrize("field", ["advertiser_count", "evidence_count"])
def test_invalid_catalog_counts_do_not_produce_scores_or_request_summary(monkeypatch, field):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    item = {"product_id": "sample-001", "name": "Example", "category": "demo",
            "demand_signal": 1, "advertiser_count": 8, "gross_margin": 0.6,
            "market_saturation": 0, "rating": 5, "evidence_count": 15, field: True}
    monkeypatch.setattr(graph_module, "_catalog", lambda: [item])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("invalid counts must not request a summary"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 422
    assert field in response.json()["detail"]
    assert "assessments" not in response.json()


@pytest.mark.parametrize("field", ["demand_signal", "gross_margin", "market_saturation", "rating"])
@pytest.mark.parametrize("value", [True, False, "0.5"])
def test_invalid_catalog_signals_do_not_produce_scores_or_request_summary(monkeypatch, field, value):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    item = {"product_id": "sample-001", "name": "Example", "category": "demo",
            "demand_signal": 1, "advertiser_count": 8, "gross_margin": 0.6,
            "market_saturation": 0, "rating": 5, "evidence_count": 15, field: value}
    monkeypatch.setattr(graph_module, "_catalog", lambda: [item])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("invalid signals must not request a summary"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 422
    assert field in response.json()["detail"]
    assert "assessments" not in response.json()


@pytest.mark.parametrize("product_id", ["", " ", "\t\n"])
def test_blank_catalog_ids_cannot_produce_assessments_or_model_summaries(monkeypatch, product_id):
    import importlib

    graph_module = importlib.import_module("demand_insight_agent.graph")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    item = {**graph_module._catalog()[0], "product_id": product_id}
    monkeypatch.setattr(graph_module, "_catalog", lambda: [item])
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("blank catalog IDs must not request a summary"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 422
    assert "product_id" in response.json()["detail"]
    assert "assessments" not in response.json()


@pytest.mark.parametrize("extra", [{"product_id": "sample-001"}, {"limits": 1},
                                  {"use_llm_summmary": True}])
def test_unknown_request_fields_are_rejected_before_graph_execution(monkeypatch, extra):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunRequest.model_validate(extra)
    monkeypatch.setattr("demand_insight_agent.api.run",
                        lambda request: pytest.fail("unknown fields must not execute the graph"))
    response = TestClient(app).post("/runs", json=extra)
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"
    assert response.json()["detail"][0]["loc"] == ["body", next(iter(extra))]
