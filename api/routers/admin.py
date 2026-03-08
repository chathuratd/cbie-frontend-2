"""
api/routers/admin.py
======================
Admin endpoints for managing users and their profiles, listing raw behaviors, and triggering pipeline runs.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from api.models import (
    UserDiscoveryItem,
    UserDiscoveryResponse,
    UserSummaryResponse,
    ProfileSummaryStats,
    CoreProfileDetailResponse,
    NoiseSummary,
    InterestEntry,
    AdminJobStatusResponse,
    BehaviorPreviewItem,
    BehaviorPreviewResponse,
    PipelineRunResponse
)
from api.dependencies import (
    get_pipeline,
    create_job,
    get_job,
    run_pipeline_background,
)
from data_adapter import DataAdapter

router = APIRouter(prefix="/admin", tags=["Admin"])
_data_adapter = DataAdapter()


def _parse_interests(raw) -> List[dict]:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return raw or []


# ---------------------------------------------------------------------------
# A. GET /admin/users
# ---------------------------------------------------------------------------
@router.get(
    "/users",
    response_model=UserDiscoveryResponse,
    summary="List All Users (Discovery)",
    description="Returns a list of all users who have ACTIVE behaviors recorded, along with their profile status.",
)
async def list_users():
    if not _data_adapter.supabase:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        # Fetch minimal info required for stats
        behaviors_resp = _data_adapter.supabase.table("behaviors").select("user_id, created_at").eq("behavior_state", "ACTIVE").execute()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database query failed: {e}")
    
    user_stats: Dict[str, Dict[str, Any]] = {}
    for row in behaviors_resp.data:
        uid = row["user_id"]
        if uid not in user_stats:
            user_stats[uid] = {"total": 0, "last_at": None}
        user_stats[uid]["total"] += 1
        
        created = row.get("created_at")
        if created:
            if user_stats[uid]["last_at"] is None or created > user_stats[uid]["last_at"]:
                user_stats[uid]["last_at"] = created
                
    try:
        profiles_resp = _data_adapter.supabase.table("core_behavior_profiles").select("user_id, confirmed_interests, updated_at").execute()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database query failed: {e}")
        
    profile_map = {row["user_id"]: row for row in profiles_resp.data}
    
    items: List[UserDiscoveryItem] = []
    for uid, stats in user_stats.items():
        has_profile = uid in profile_map
        prof = profile_map.get(uid)
        
        kwargs = {
            "user_id": uid,
            "total_behaviors": stats["total"],
            "last_behavior_at": stats["last_at"],
            "has_profile": has_profile
        }
        
        if prof:
            interests = _parse_interests(prof.get("confirmed_interests", []))
            kwargs.update({
                "profile_interest_count": len(interests),
                "profile_stable_count": len([i for i in interests if i.get("status") == "Stable"]),
                "profile_emerging_count": len([i for i in interests if i.get("status") == "Emerging"]),
                "profile_fact_count": len([i for i in interests if i.get("status") == "Stable Fact"]),
                "profile_last_updated": prof.get("updated_at")
            })
            
        items.append(UserDiscoveryItem(**kwargs))
        
    return UserDiscoveryResponse(total_users=len(items), users=items)


# ---------------------------------------------------------------------------
# B. GET /admin/users/{user_id}
# ---------------------------------------------------------------------------
@router.get(
    "/users/{user_id}",
    response_model=UserSummaryResponse,
    summary="Get Per-User Summary",
    description="Combines basic behavior stats from the behaviors table with the profile summary from core_behavior_profiles if it exists.",
)
async def get_user_summary(user_id: str):
    if not _data_adapter.supabase:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        behaviors_resp = _data_adapter.supabase.table("behaviors").select("created_at").eq("user_id", user_id).eq("behavior_state", "ACTIVE").execute()
        prof_resp = _data_adapter.supabase.table("core_behavior_profiles").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database query failed: {e}")
        
    if not behaviors_resp.data:
        raise HTTPException(status_code=404, detail=f"No behaviors found for user '{user_id}'.")
        
    total_behaviors = len(behaviors_resp.data)
    last_behavior_at = max((r.get("created_at") for r in behaviors_resp.data if r.get("created_at")), default=None)
    
    has_profile = len(prof_resp.data) > 0
    profile_summary = None
    
    if has_profile:
        prof = prof_resp.data[0]
        interests = _parse_interests(prof.get("confirmed_interests", []))
        profile_summary = ProfileSummaryStats(
            total_raw_behaviors=prof.get("total_raw_behaviors", 0),
            interest_count=len(interests),
            stable_count=len([i for i in interests if i.get("status") == "Stable"]),
            emerging_count=len([i for i in interests if i.get("status") == "Emerging"]),
            fact_count=len([i for i in interests if i.get("status") == "Stable Fact"]),
            last_updated=prof.get("updated_at"),
            last_job_id=None
        )
        
    return UserSummaryResponse(
        user_id=user_id,
        total_behaviors=total_behaviors,
        last_behavior_at=last_behavior_at,
        has_profile=has_profile,
        profile_summary=profile_summary
    )


# ---------------------------------------------------------------------------
# C. GET /admin/users/{user_id}/profile
# ---------------------------------------------------------------------------
@router.get(
    "/users/{user_id}/profile",
    response_model=CoreProfileDetailResponse,
    summary="Get Core Profile Detail (Admin-Friendly)",
    description="Returns full decoded core profile including categories of interests.",
)
async def get_core_profile_detail(user_id: str):
    if not _data_adapter.supabase:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    try:
        prof_resp = _data_adapter.supabase.table("core_behavior_profiles").select("*").eq("user_id", user_id).execute()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database query failed: {e}")
        
    if not prof_resp.data:
        raise HTTPException(status_code=404, detail=f"No profile found for user '{user_id}'.")
        
    prof = prof_resp.data[0]
    interests_raw = _parse_interests(prof.get("confirmed_interests", []))
    interests = [InterestEntry(**i) for i in interests_raw]
    
    critical_constraints = [i for i in interests if i.status == "Stable Fact"]
    stable_interests = [i for i in interests if i.status == "Stable"]
    emerging_interests = [i for i in interests if i.status == "Emerging"]
    archived_core = [i for i in interests if i.status == "ARCHIVED_CORE"]
    
    noise_count = len([i for i in interests if i.status == "Noise"])
    archived_count = len(archived_core)
    
    return CoreProfileDetailResponse(
        user_id=user_id,
        total_raw_behaviors=prof.get("total_raw_behaviors", 0),
        critical_constraints=critical_constraints,
        stable_interests=stable_interests,
        emerging_interests=emerging_interests,
        archived_core=archived_core,
        noise_summary=NoiseSummary(noise_count=noise_count, archived_count=archived_count),
        identity_anchor_prompt=prof.get("identity_anchor_prompt"),
        last_updated=prof.get("updated_at")
    )


# ---------------------------------------------------------------------------
# D. POST /admin/users/{user_id}/run_pipeline and GET /admin/jobs/{job_id}
# ---------------------------------------------------------------------------
@router.post(
    "/users/{user_id}/run_pipeline",
    response_model=PipelineRunResponse,
    status_code=202,
    summary="Trigger CBIE Pipeline Run (Admin Wrapper)",
    description="Thin wrapper around the pipeline/run to process a user manually from the admin panel.",
)
async def admin_trigger_pipeline(user_id: str, background_tasks: BackgroundTasks):
    get_pipeline()  # raises 500 if not initialised
    job_id = create_job(user_id)
    background_tasks.add_task(run_pipeline_background, job_id, user_id)
    
    return PipelineRunResponse(
        job_id=job_id,
        status="QUEUED",
        user_id=user_id,
        message=f"Pipeline run queued for user '{user_id}'. Poll GET /admin/jobs/{{job_id}} for progress."
    )


@router.get(
    "/jobs/{job_id}",
    response_model=AdminJobStatusResponse,
    summary="Get Pipeline Job Status (Admin Wrapper)",
    description="Thin wrapper around pipeline/status to get simplified job status for the frontend.",
)
async def admin_get_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        
    return AdminJobStatusResponse(
        job_id=job["job_id"],
        user_id=job["user_id"],
        status=job["status"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


# ---------------------------------------------------------------------------
# E. GET /admin/users/{user_id}/behaviors
# ---------------------------------------------------------------------------
@router.get(
    "/users/{user_id}/behaviors",
    response_model=BehaviorPreviewResponse,
    summary="Behavior Preview",
    description="Returns sample rows from behaviors (without embeddings) to inspect raw data.",
)
async def get_behaviors_preview(
    user_id: str,
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    if not _data_adapter.supabase:
        raise HTTPException(status_code=503, detail="Database connection unavailable.")

    # Select specific columns to explicitly omit 'embedding'
    columns = "behavior_id, created_at, behavior_text, intent, target, context, polarity, behavior_state, credibility, clarity_score, extraction_confidence"
    
    try:
        resp = (
            _data_adapter.supabase
            .table("behaviors")
            .select(columns, count="exact")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database query failed: {e}")
        
    items = []
    for row in resp.data:
        def _f(val):
            return float(val) if val is not None else None
            
        items.append(BehaviorPreviewItem(
            behavior_id=row.get("behavior_id"),
            created_at=row.get("created_at"),
            behavior_text=row.get("behavior_text", ""),
            intent=row.get("intent"),
            target=row.get("target"),
            context=row.get("context"),
            polarity=row.get("polarity"),
            behavior_state=row.get("behavior_state"),
            credibility=_f(row.get("credibility")),
            clarity_score=_f(row.get("clarity_score")),
            extraction_confidence=_f(row.get("extraction_confidence"))
        ))
        
    total = resp.count if resp.count is not None else len(items)
    return BehaviorPreviewResponse(user_id=user_id, total=total, behaviors=items)
