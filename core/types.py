"""Core data structures for the Shared Information Repository (SIR)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class RoutingPath(str, Enum):
    STRUCTURAL_RULE = "STRUCTURAL_RULE"
    SEMANTIC_FLOW = "SEMANTIC_FLOW"
    VISUAL_TOPOLOGY = "VISUAL_TOPOLOGY"


class RiskProfile(BaseModel):
    """Risk assessment profile for routing decisions."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    score: float = Field(ge=0.0, le=1.0)
    reason: str
    high_risk_pages: List[int] = Field(default_factory=list)


class Blueprint(BaseModel):
    """Inspector Agent output blueprint."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    selected_path: RoutingPath
    confidence_score: float = Field(ge=0.0, le=1.0)
    risk_assessment: RiskProfile
    execution_plan: Dict[str, Any] | str
