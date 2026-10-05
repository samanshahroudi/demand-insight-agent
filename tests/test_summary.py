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
