from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Product(BaseModel):
    product_id: str = Field(min_length=1, pattern=r"\S")
    name: str = Field(min_length=1, pattern=r"\S")
    category: str = Field(min_length=1, pattern=r"\S")
    demand_signal: float = Field(ge=0, le=1, strict=True)
    advertiser_count: int = Field(ge=0, strict=True)
    gross_margin: float = Field(ge=0, le=1, strict=True)
    market_saturation: float = Field(ge=0, le=1, strict=True)
    rating: float = Field(ge=0, le=5, strict=True)
    evidence_count: int = Field(ge=0, strict=True)


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
    product_ids: list[Annotated[str, Field(min_length=1, pattern=r"\S")]] = Field(
        default_factory=list, max_length=30,
    )
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    use_llm_summary: bool = Field(default=False, strict=True)

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
