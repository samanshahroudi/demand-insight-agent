from .models import Product, ProductAssessment

SCORING_VERSION = "0.1.0"


def assess(product: Product) -> ProductAssessment:
    """Score a product with explicit, inspectable rules; no model-generated scores."""
    signals = {
        "demand": product.demand_signal * 30,
        "advertiser_validation": min(product.advertiser_count / 8, 1) * 20,
        "margin": min(product.gross_margin / 0.60, 1) * 25,
        "market_headroom": (1 - product.market_saturation) * 15,
        "customer_rating": min(product.rating / 5, 1) * 10,
    }
    score = round(sum(signals.values()), 1)
    reasons = [
        f"Demand signal contributes {signals['demand']:.1f}/30 points.",
        f"Gross margin contributes {signals['margin']:.1f}/25 points.",
        f"Market headroom contributes {signals['market_headroom']:.1f}/15 points.",
    ]

    if product.gross_margin < 0.15:
        verdict = "reject"
        reasons.append("Rejected by the minimum 15% gross-margin gate.")
    elif score >= 70:
        verdict = "winner"
    elif score >= 52:
        verdict = "promising"
    elif score >= 35:
        verdict = "weak"
    else:
        verdict = "reject"

    confidence = (
        "high" if product.evidence_count >= 15 else "medium" if product.evidence_count >= 7 else "low"
    )
    return ProductAssessment(
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        score=score,
        verdict=verdict,
        confidence=confidence,
        gross_margin=product.gross_margin,
        signals={key: round(value, 1) for key, value in signals.items()},
        reasons=reasons,
    )
