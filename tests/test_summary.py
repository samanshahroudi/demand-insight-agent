from demand_insight_agent.graph import deterministic_summary
from demand_insight_agent.models import Product
from demand_insight_agent.scoring import assess


def test_summary_labels_high_scoring_product_rejected_by_margin_gate():
    result = assess(Product(
        product_id="low-margin", name="Low margin product", category="demo",
        demand_signal=1, advertiser_count=8, gross_margin=0.1,
        market_saturation=0, rating=5, evidence_count=15,
    ))
    assert result.score >= 70
    assert result.verdict == "reject"
    summary = deterministic_summary({"assessments": [result.model_dump()]})["summary"]
    assert f"Low margin product ({result.score:.1f}, reject)" in summary
    assert summary.startswith("Top ranked products")


def test_model_summary_extracts_text_from_string_and_content_blocks(monkeypatch):
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    from demand_insight_agent.graph import model_summary

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    text = "The top product is promising. Evidence is limited."
    for content in [text, [{"type": "reasoning", "reasoning": "private reasoning"},
                           {"type": "text", "text": "The top product is promising. "},
                           {"type": "text", "text": "Evidence is limited."}]]:
        model = Mock()
        model.invoke.return_value = AIMessage(content=content)
        factory = Mock(return_value=model)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", factory)
        assert model_summary({"assessments": []}) == {"summary": text}
        factory.assert_called_once_with(model="test-model", temperature=0)
        model.invoke.assert_called_once()
