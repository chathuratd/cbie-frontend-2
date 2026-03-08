"""
api/models.py
=============
All Pydantic request/response schemas for the CBIE API.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / Reusable sub-models
# ---------------------------------------------------------------------------

class InterestEntry(BaseModel):
    cluster_id: Any = Field(..., description="DBSCAN cluster ID or 'absolute_fact'")
    representative_topics: List[str] = Field(..., description="LLM-generalised topic labels")
    frequency: int = Field(..., description="Number of behaviors in the cluster")
    consistency_score: float = Field(..., description="Gini-based consistency (0-1)")
    trend_score: float = Field(..., description="Mann-Kendall trend score (-1 to 1)")
    core_score: float = Field(..., description="AHP-weighted final confirmation score")
    status: str = Field(..., description="Stable | Emerging | Stable Fact | Noise | ARCHIVED_CORE")


# ---------------------------------------------------------------------------
# /context
# ---------------------------------------------------------------------------

class ContextResponse(BaseModel):
    user_id: str
    identity_anchor_prompt: str = Field(..., description="Ready-to-inject system prompt string")
    profile_exists: bool
    total_raw_behaviors: int
    last_updated: Optional[str] = None


# ---------------------------------------------------------------------------
# /pipeline
# ---------------------------------------------------------------------------

class PipelineRunResponse(BaseModel):
    job_id: str
    status: str = Field("QUEUED", description="QUEUED | RUNNING | COMPLETED | FAILED")
    user_id: str
    message: str


class PipelineStatusResponse(BaseModel):
    job_id: str
    user_id: str
    status: str = Field(..., description="QUEUED | RUNNING | COMPLETED | FAILED")
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = Field(None, description="Full profile JSON on completion")
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# /profiles
# ---------------------------------------------------------------------------

class ProfileSummary(BaseModel):
    user_id: str
    total_raw_behaviors: int
    interest_count: int
    fact_count: int
    stable_count: int
    emerging_count: int
    last_updated: Optional[str] = None


class ProfileListResponse(BaseModel):
    total: int
    profiles: List[ProfileSummary]


class ProfileResponse(BaseModel):
    user_id: str
    total_raw_behaviors: int
    confirmed_interests: List[InterestEntry]
    identity_anchor_prompt: Optional[str] = None
    last_updated: Optional[str] = None


class InterestsResponse(BaseModel):
    user_id: str
    interests: List[InterestEntry]
    total_count: int


class FactsResponse(BaseModel):
    user_id: str
    facts: List[InterestEntry] = Field(..., description="Only interests with status='Stable Fact'")
    total_count: int


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    engine: str = "CBIE"
    version: str = "1.0.0"
    pipeline_ready: bool


class RootResponse(BaseModel):
    service: str = "Core Behaviour Identification Engine (CBIE)"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    health_url: str = "/health"


# ---------------------------------------------------------------------------
# /admin
# ---------------------------------------------------------------------------

class UserDiscoveryItem(BaseModel):
    user_id: str
    total_behaviors: int
    last_behavior_at: Optional[str] = None
    has_profile: bool
    profile_interest_count: Optional[int] = None
    profile_stable_count: Optional[int] = None
    profile_emerging_count: Optional[int] = None
    profile_fact_count: Optional[int] = None
    profile_last_updated: Optional[str] = None

class UserDiscoveryResponse(BaseModel):
    total_users: int
    users: List[UserDiscoveryItem]

class ProfileSummaryStats(BaseModel):
    total_raw_behaviors: int
    interest_count: int
    stable_count: int
    emerging_count: int
    fact_count: int
    last_updated: Optional[str] = None
    last_job_id: Optional[str] = None

class UserSummaryResponse(BaseModel):
    user_id: str
    total_behaviors: int
    last_behavior_at: Optional[str] = None
    has_profile: bool
    profile_summary: Optional[ProfileSummaryStats] = None

class NoiseSummary(BaseModel):
    noise_count: int
    archived_count: int

class CoreProfileDetailResponse(BaseModel):
    user_id: str
    total_raw_behaviors: int
    critical_constraints: List[InterestEntry]
    stable_interests: List[InterestEntry]
    emerging_interests: List[InterestEntry]
    archived_core: List[InterestEntry]
    noise_summary: NoiseSummary
    identity_anchor_prompt: Optional[str] = None
    last_updated: Optional[str] = None

class AdminJobStatusResponse(BaseModel):
    job_id: str
    user_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

class BehaviorPreviewItem(BaseModel):
    behavior_id: Optional[str] = None
    created_at: Optional[str] = None
    behavior_text: str
    intent: Optional[str] = None
    target: Optional[str] = None
    context: Optional[str] = None
    polarity: Optional[str] = None
    behavior_state: Optional[str] = None
    credibility: Optional[float] = None
    clarity_score: Optional[float] = None
    extraction_confidence: Optional[float] = None

class BehaviorPreviewResponse(BaseModel):
    user_id: str
    total: int
    behaviors: List[BehaviorPreviewItem]
