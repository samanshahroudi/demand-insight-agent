from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    demand_signal: float = Field(ge=0, le=1)
    advertiser_count: int = Field(ge=0)
    gross_margin: float = Field(ge=0, le=1)
    market_saturation: float = Field(ge=0, le=1)
    rating: float = Field(ge=0, le=5)
    evidence_count: int = Field(ge=0)


class ProductAssessment(BaseModel):
    product_id: str
    name: str
    category: str
    score: float
    verdict: Literal["winner", "promising", "weak", "reject"]
    confidence: Literal["low", "medium", "high"]
    gross_margin: float
    signals: dict[str, float]
    reasons: list[str]


class RunRequest(BaseModel):
    mode: Literal["discover", "shortlist"] = "discover"
    product_ids: list[str] = Field(default_factory=list, max_length=30)
    limit: int = Field(default=5, ge=1, le=20)
    use_llm_summary: bool = False

    @model_validator(mode="after")
    def require_shortlist(self) -> "RunRequest":
        if self.mode == "shortlist" and not self.product_ids:
            raise ValueError("Provide at least one product_id in shortlist mode.")
        return self


class RunResponse(BaseModel):
    run_id: str
    mode: str
    assessments: list[ProductAssessment]
    summary: str
    scoring_version: str
