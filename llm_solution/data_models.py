from enum import Enum

from pydantic import BaseModel, Field


class TextGenerationWebUIGenerateResult(BaseModel):
    text: str


class TextGenerationWebUIGenerateResponse(BaseModel):
    results: list[TextGenerationWebUIGenerateResult]


class Opinion(BaseModel):
    Source: list[list[str]] = Field(..., min_items=2, max_items=2)
    Target: list[list[str]] = Field(..., min_items=2, max_items=2)
    Polar_expression: list[list[str]] = Field(..., min_items=2, max_items=2)
    Polarity: str
    Intensity: str | None


class Sentence(BaseModel):
    sent_id: str
    text: str
    opinions: list[Opinion]


class Polarity(str, Enum):
    Neutral = "Neutral"
    Positive = "Positive"
    Negative = "Negative"


class Intensity(str, Enum):
    Average = "Average"
    Strong = "Strong"
    Standard = "Standard"
    Weak = "Weak"
    Slight = "Slight"


class SimpleOpinion(BaseModel):
    Source: str | None
    Target: str | None
    Polar_expression: str
    Polarity: Polarity
    Intensity: Intensity | None
