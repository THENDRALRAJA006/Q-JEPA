"""
Google Maps Platform Integration Client - Surakshit Bharat Q-JEPA Traffic.

Provides a typed asynchronous client for the Google Routes API v2.
Designed for use by the FastAPI backend ONLY (server-side API key).

Key design principles:
  - Server key is NEVER returned to the browser or logged in full.
  - Google response data must NOT be written into ML datasets or checkpoints.
  - Offline / unconfigured mode returns clear status without crashing.
  - Rate-limit and error handling are enforced before every live request.

Google Routes API reference:
  https://developers.google.com/maps/documentation/routes

Usage restrictions per Google Maps Platform Terms of Service:
  - Route data (polylines, durations, distances) must NOT be used to train,
    fine-tune, validate, or test the JEPAGraph model or any ML component.
  - Only application-owned metadata (request IDs, status, latency) may be stored.
  - Attribution "(C) Google" must accompany any displayed map content.
"""
from __future__ import annotations

import os
import time
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROUTES_API_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Field mask - only request what we use; avoids unnecessary billing and data.
# Per ToS: do NOT request fields you store permanently in ML datasets.
FIELD_MASK = ",".join([
    "routes.distanceMeters",
    "routes.duration",
    "routes.staticDuration",
    "routes.polyline.encodedPolyline",
    "routes.description",
    "routes.warnings",
])

REQUEST_TIMEOUT_S = 15.0
MAX_ALTERNATIVES = 2


class RoutingPreference(str, Enum):
    """Traffic model preference for driving routes."""
    TRAFFIC_UNAWARE = "TRAFFIC_UNAWARE"
    TRAFFIC_AWARE = "TRAFFIC_AWARE"
    TRAFFIC_AWARE_OPTIMAL = "TRAFFIC_AWARE_OPTIMAL"


class TravelMode(str, Enum):
    DRIVE = "DRIVE"
    WALK = "WALK"
    BICYCLE = "BICYCLE"
    TWO_WHEELER = "TWO_WHEELER"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """One route option returned by the Google Routes API."""
    route_index: int
    distance_meters: int
    duration_seconds: int          # Traffic-aware duration (if TRAFFIC_AWARE*)
    static_duration_seconds: int   # Duration without traffic consideration
    encoded_polyline: str          # For map rendering only - NOT for ML input
    description: str
    warnings: List[str]


@dataclass
class GoogleRoutesResponse:
    """Structured response from the Routes API client."""
    request_id: str
    success: bool
    routes: List[RouteResult] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    api_latency_ms: Optional[float] = None
    timestamp_utc: str = ""
    origin_lat: Optional[float] = None
    origin_lon: Optional[float] = None
    destination_lat: Optional[float] = None
    destination_lon: Optional[float] = None
    routing_preference: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for r in d.get("routes", []):
            r["_display_only"] = True
            r["_tos_note"] = (
                "Polyline is for display only. Per Google Maps Platform ToS, "
                "route content must not be used to train or evaluate ML models."
            )
        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_coord(lat: float, lon: float, label: str = "coordinate") -> None:
    """Raise ValueError if coordinates are out of geographic bounds."""
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"{label} latitude {lat} is out of range [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"{label} longitude {lon} is out of range [-180, 180].")


def _redact_key(key: str) -> str:
    """Return a redacted version of an API key safe for logging."""
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Simple in-process rate limiter
# ---------------------------------------------------------------------------

class _SimpleRateLimiter:
    """
    Token-bucket rate limiter for Google Maps API calls.
    Default: max 10 requests per 60-second window.
    """

    def __init__(self, max_requests: int = 10, window_seconds: float = 60.0):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: List[float] = []

    def check(self) -> Tuple[bool, str]:
        """Return (allowed, reason). Prune expired entries first."""
        now = time.monotonic()
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max:
            oldest = self._timestamps[0]
            wait = self._window - (now - oldest)
            return False, (
                f"Rate limit: {self._max} requests per {self._window:.0f}s. "
                f"Retry in {wait:.0f}s."
            )
        self._timestamps.append(now)
        return True, "ok"


_rate_limiter = _SimpleRateLimiter(max_requests=10, window_seconds=60.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GoogleMapsClient:
    """
    Async client for the Google Routes API.

    Validates credentials on construction but does NOT make a live API call.

    Usage:
        async with GoogleMapsClient() as client:
            response = await client.compute_routes(...)
    """

    def __init__(
        self,
        server_api_key: Optional[str] = None,
        routing_preference: RoutingPreference = RoutingPreference.TRAFFIC_AWARE,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ):
        self._key = server_api_key or os.environ.get("GOOGLE_MAPS_SERVER_API_KEY", "")
        self._preference = routing_preference
        self._timeout = timeout_s
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """True if a non-empty API key is set."""
        return bool(self._key and self._key.strip())

    async def __aenter__(self) -> "GoogleMapsClient":
        self._client = httpx.AsyncClient(timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _make_waypoint(self, lat: float, lon: float) -> Dict[str, Any]:
        return {"location": {"latLng": {"latitude": lat, "longitude": lon}}}

    def _build_payload(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        departure_time: Optional[str],
        travel_mode: TravelMode,
        compute_alternatives: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "origin": self._make_waypoint(origin_lat, origin_lon),
            "destination": self._make_waypoint(destination_lat, destination_lon),
            "travelMode": travel_mode.value,
            "computeAlternativeRoutes": compute_alternatives,
            "languageCode": "en-US",
            "units": "METRIC",
        }
        if travel_mode == TravelMode.DRIVE:
            payload["routingPreference"] = self._preference.value
            if departure_time:
                payload["departureTime"] = departure_time
            else:
                payload["departureTime"] = datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
        return payload

    async def compute_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        departure_time: Optional[str] = None,
        travel_mode: TravelMode = TravelMode.DRIVE,
        compute_alternatives: bool = True,
    ) -> GoogleRoutesResponse:
        """
        Request traffic-aware route(s) from the Google Routes API.

        Returns a GoogleRoutesResponse. Never raises on API errors -
        errors are captured in the response object.

        IMPORTANT: The returned route data (polylines, durations, distances)
        must NOT be fed into ML training, testing, or evaluation pipelines.
        """
        request_id = str(uuid.uuid4())[:12]
        ts = datetime.now(timezone.utc).isoformat()

        base_response = GoogleRoutesResponse(
            request_id=request_id,
            success=False,
            timestamp_utc=ts,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
            routing_preference=self._preference.value,
        )

        # 1. Config check
        if not self.is_configured:
            base_response.error_code = "NOT_CONFIGURED"
            base_response.error_message = (
                "GOOGLE_MAPS_SERVER_API_KEY is not set. "
                "See docs/GOOGLE_MAPS_SETUP.md to configure credentials."
            )
            logger.warning("[GoogleMapsClient] API key not configured.")
            return base_response

        # 2. Coordinate validation
        try:
            _validate_coord(origin_lat, origin_lon, "origin")
            _validate_coord(destination_lat, destination_lon, "destination")
        except ValueError as exc:
            base_response.error_code = "INVALID_COORDINATES"
            base_response.error_message = str(exc)
            return base_response

        # 3. Rate limit
        allowed, reason = _rate_limiter.check()
        if not allowed:
            base_response.error_code = "RATE_LIMITED"
            base_response.error_message = reason
            logger.warning("[GoogleMapsClient] %s", reason)
            return base_response

        # 4. Build and send request
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        payload = self._build_payload(
            origin_lat, origin_lon, destination_lat, destination_lon,
            departure_time, travel_mode, compute_alternatives,
        )
        headers = {
            "X-Goog-Api-Key": self._key,
            "X-Goog-FieldMask": FIELD_MASK,
            "Content-Type": "application/json",
        }

        t0 = time.monotonic()
        try:
            resp = await self._client.post(ROUTES_API_URL, json=payload, headers=headers)
            latency_ms = (time.monotonic() - t0) * 1000.0
            base_response.api_latency_ms = round(latency_ms, 1)

            logger.info(
                "[GoogleMapsClient] request_id=%s status=%s latency=%.0fms key=%s",
                request_id, resp.status_code, latency_ms,
                _redact_key(self._key),
            )

            if resp.status_code == 200:
                return self._parse_success(resp.json(), base_response)
            elif resp.status_code == 403:
                base_response.error_code = "AUTH_FAILURE"
                base_response.error_message = (
                    "Authentication failed (HTTP 403). "
                    "Verify GOOGLE_MAPS_SERVER_API_KEY, billing status, and API enablement."
                )
            elif resp.status_code == 429:
                base_response.error_code = "QUOTA_EXCEEDED"
                base_response.error_message = (
                    "Quota exceeded (HTTP 429). "
                    "Check your Google Cloud Console quotas and billing alerts."
                )
            else:
                body = resp.text[:400]
                base_response.error_code = f"HTTP_{resp.status_code}"
                base_response.error_message = (
                    f"Google Routes API returned HTTP {resp.status_code}. Details: {body}"
                )
            return base_response

        except httpx.TimeoutException:
            latency_ms = (time.monotonic() - t0) * 1000.0
            base_response.api_latency_ms = round(latency_ms, 1)
            base_response.error_code = "TIMEOUT"
            base_response.error_message = (
                f"Request timed out after {self._timeout:.0f}s. "
                "Check network connectivity."
            )
            logger.warning("[GoogleMapsClient] request_id=%s TIMEOUT", request_id)
            return base_response

        except httpx.RequestError as exc:
            base_response.error_code = "NETWORK_ERROR"
            base_response.error_message = f"Network error: {type(exc).__name__}"
            logger.exception("[GoogleMapsClient] request_id=%s network error", request_id)
            return base_response

        except Exception as exc:
            base_response.error_code = "INTERNAL_ERROR"
            base_response.error_message = f"Unexpected error: {type(exc).__name__}"
            logger.exception("[GoogleMapsClient] request_id=%s unexpected error", request_id)
            return base_response

    def _parse_success(
        self,
        body: Dict[str, Any],
        base: GoogleRoutesResponse,
    ) -> GoogleRoutesResponse:
        """Parse a 200-OK Routes API response into structured RouteResult objects."""
        raw_routes = body.get("routes", [])
        if not raw_routes:
            base.error_code = "NO_ROUTE"
            base.error_message = (
                "Google Routes API returned no routes for the requested origin/destination. "
                "Verify that the coordinates are valid and drivable."
            )
            return base

        def _parse_dur(s: str) -> int:
            try:
                return int(str(s).rstrip("s"))
            except (ValueError, AttributeError):
                return 0

        parsed: List[RouteResult] = []
        for idx, r in enumerate(raw_routes[:MAX_ALTERNATIVES + 1]):
            parsed.append(RouteResult(
                route_index=idx,
                distance_meters=r.get("distanceMeters", 0),
                duration_seconds=_parse_dur(r.get("duration", "0s")),
                static_duration_seconds=_parse_dur(r.get("staticDuration", "0s")),
                encoded_polyline=r.get("polyline", {}).get("encodedPolyline", ""),
                description=r.get("description", ""),
                warnings=r.get("warnings", []),
            ))

        base.success = True
        base.routes = parsed
        return base


# ---------------------------------------------------------------------------
# Configuration status helpers
# ---------------------------------------------------------------------------

def get_google_maps_config_status() -> Dict[str, Any]:
    """
    Return public-safe configuration status. Never exposes key values.
    Safe to call on every dashboard load.
    """
    server_key = os.environ.get("GOOGLE_MAPS_SERVER_API_KEY", "")
    browser_key = os.environ.get("GOOGLE_MAPS_BROWSER_API_KEY", "")

    server_configured = bool(server_key.strip())
    browser_configured = bool(browser_key.strip())

    return {
        "server_key_configured": server_configured,
        "browser_key_configured": browser_configured,
        "fully_configured": server_configured and browser_configured,
        "offline_mode": not (server_configured and browser_configured),
        "server_key_hint": _redact_key(server_key) if server_configured else None,
        "browser_key_hint": _redact_key(browser_key) if browser_configured else None,
        "setup_guide": "docs/GOOGLE_MAPS_SETUP.md",
        "pricing_reference": "https://developers.google.com/maps/billing-and-pricing/pricing-india",
        "tos_note": (
            "Google Maps route data (polylines, durations, distances) must NOT be used "
            "to train, test, or evaluate ML models per Google Maps Platform ToS."
        ),
    }


def get_browser_api_key_for_frontend() -> Optional[str]:
    """
    Returns the browser API key ONLY - safe to embed in Streamlit HTML.
    The browser key should be restricted to Maps JavaScript API +
    authorized HTTP referrers in your Google Cloud Console configuration.
    """
    key = os.environ.get("GOOGLE_MAPS_BROWSER_API_KEY", "").strip()
    return key if key else None
