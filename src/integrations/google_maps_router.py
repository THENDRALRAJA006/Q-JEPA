"""
Google Maps integration FastAPI router — Surakshit Bharat Q-JEPA Traffic.

Endpoints:
  GET  /integrations/google/status         - Config status (no credentials exposed)
  GET  /integrations/google/config-status  - Alias for status
  POST /integrations/google/routes         - Compute traffic-aware route via Google

Design notes:
  - Server API key is read from env, NEVER returned to caller.
  - Rate limiting is enforced per-process (see GoogleMapsClient).
  - Google route data returned to the browser is tagged display-only.
  - Google route data must NOT be stored in ML datasets or checkpoints.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.integrations.google_maps_client import (
    GoogleMapsClient,
    RoutingPreference,
    TravelMode,
    get_google_maps_config_status,
)
from src.common.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/integrations/google", tags=["Google Maps Integration"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GoogleRouteRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90.0, le=90.0, description="Origin latitude")
    origin_lon: float = Field(..., ge=-180.0, le=180.0, description="Origin longitude")
    destination_lat: float = Field(..., ge=-90.0, le=90.0, description="Destination latitude")
    destination_lon: float = Field(..., ge=-180.0, le=180.0, description="Destination longitude")
    routing_preference: str = Field(
        "TRAFFIC_AWARE",
        description="TRAFFIC_UNAWARE | TRAFFIC_AWARE | TRAFFIC_AWARE_OPTIMAL",
    )
    travel_mode: str = Field("DRIVE", description="DRIVE | WALK | BICYCLE | TWO_WHEELER")
    compute_alternatives: bool = Field(True, description="Request up to 3 alternative routes")
    departure_time: Optional[str] = Field(
        None,
        description="RFC3339 UTC departure time, e.g. 2026-10-14T09:00:00Z. Defaults to now.",
    )

    @field_validator("routing_preference")
    @classmethod
    def validate_pref(cls, v: str) -> str:
        valid = {p.value for p in RoutingPreference}
        if v not in valid:
            raise ValueError(f"routing_preference must be one of {valid}")
        return v

    @field_validator("travel_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid = {m.value for m in TravelMode}
        if v not in valid:
            raise ValueError(f"travel_mode must be one of {valid}")
        return v

    @field_validator("departure_time")
    @classmethod
    def validate_departure_time(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(
                "departure_time must be a valid RFC3339 string, e.g. 2026-10-14T09:00:00Z"
            )
        return v


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status")
@router.get("/config-status")
async def google_maps_status() -> Dict[str, Any]:
    """
    Return Google Maps configuration status.
    Never reveals API key values - only whether they are set.
    """
    return get_google_maps_config_status()


@router.post("/routes")
async def compute_google_routes(req: GoogleRouteRequest) -> Dict[str, Any]:
    """
    Compute a traffic-aware route using the Google Routes API.

    IMPORTANT:
      - The server API key is used internally and never returned to the caller.
      - The returned route data is for DISPLAY PURPOSES ONLY.
      - Per Google Maps Platform ToS, route content (polylines, durations,
        distances) must NOT be used to train, test, or fine-tune ML models.

    When credentials are not configured, returns offline_mode status instead
    of raising an error.
    """
    pref = RoutingPreference(req.routing_preference)
    mode = TravelMode(req.travel_mode)

    async with GoogleMapsClient(routing_preference=pref) as client:
        result = await client.compute_routes(
            origin_lat=req.origin_lat,
            origin_lon=req.origin_lon,
            destination_lat=req.destination_lat,
            destination_lon=req.destination_lon,
            departure_time=req.departure_time,
            travel_mode=mode,
            compute_alternatives=req.compute_alternatives,
        )

    response = result.to_dict()
    response["_attribution"] = "Powered by Google Maps Platform. Map data (C) Google."
    response["_tos_boundary"] = (
        "Route content is for display only and must not be used in ML training or evaluation."
    )
    logger.info(
        "[google/routes] request_id=%s success=%s error=%s",
        result.request_id, result.success, result.error_code,
    )
    return response
