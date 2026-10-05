import pytest

from demand_insight_agent.models import Product
from demand_insight_agent.scoring import assess


def product(**changes):
    return Product.model_validate({
        "product_id": "boundary-example", "name": "Example", "category": "demo",
        "demand_signal": 0, "advertiser_count": 0, "gross_margin": 0.6,
        "market_saturation": 1, "rating": 0, "evidence_count": 0, **changes,
    })


@pytest.mark.parametrize("demand,advertisers,score,verdict", [
    (0.33, 0, 34.9, "reject"), (1 / 3, 0, 35, "weak"),
    (26.9 / 30, 0, 51.9, "weak"), (0.9, 0, 52, "promising"),
    (0.83, 8, 69.9, "promising"), (25 / 30, 8, 70, "winner"),
])
def test_verdict_score_boundaries(demand, advertisers, score, verdict):
    result = assess(product(demand_signal=demand, advertiser_count=advertisers))
    assert result.score == score
    assert result.verdict == verdict


@pytest.mark.parametrize("margin,verdict", [(0.1499, "reject"), (0.15, "winner")])
def test_margin_gate_overrides_high_score_only_below_fifteen_percent(margin, verdict):
    result = assess(product(demand_signal=1, advertiser_count=8, gross_margin=margin,
                            market_saturation=0, rating=5))
    assert result.score > 70
    assert result.verdict == verdict
    assert any("gross-margin gate" in reason for reason in result.reasons) == (margin < 0.15)


@pytest.mark.parametrize("advertisers,margin", [(8, 0.6), (80, 1)])
def test_component_weights_and_caps_keep_maximum_score_at_one_hundred(advertisers, margin):
    result = assess(product(demand_signal=1, advertiser_count=advertisers, gross_margin=margin,
                            market_saturation=0, rating=5))
    assert result.signals == {"demand": 30, "advertiser_validation": 20, "margin": 25,
                              "market_headroom": 15, "customer_rating": 10}
    assert result.score == 100
    assert result.verdict == "winner"


@pytest.mark.parametrize("evidence,confidence", [(0, "low"), (6, "low"), (7, "medium"),
                                             (14, "medium"), (15, "high")])
def test_evidence_confidence_boundaries_do_not_change_scores(evidence, confidence):
    result = assess(product(evidence_count=evidence))
    assert result.confidence == confidence
    assert result.score == 25
    assert result.verdict == "reject"
