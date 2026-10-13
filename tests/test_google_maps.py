"""
Integration and unit tests for Google Maps Platform client and router.
Surakshit Bharat Q-JEPA Traffic.

Covers:
  - Coordinate validation and boundary conditions
  - Secret redaction and privacy preservation
  - In-process rate limiting
  - Graceful offline / unconfigured mode
  - Mocked HTTP error handling (403, 429, 500, Timeout, Network Error)
  - Successful route response parsing and time conversion
  - ToS boundary preservation (display-only flag, ToS notes, no ML data leakage)
  - FastAPI endpoints (GET /status, GET /config-status, POST /routes)
  - Pydantic schema validation for coordinates, preferences, and timestamps

Run with:
  python -m pytest tests/test_google_maps.py -v --tb=short
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.integrations.google_maps_client import (
    GoogleMapsClient,
    GoogleRoutesResponse,
    RouteResult,
    RoutingPreference,
    TravelMode,
    _SimpleRateLimiter,
    _redact_key,
    _validate_coord,
    get_browser_api_key_for_frontend,
    get_google_maps_config_status,
)
from src.integrations.google_maps_router import GoogleRouteRequest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    """Ensure Google Maps env vars are unset for isolation."""
    monkeypatch.delenv("GOOGLE_MAPS_SERVER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_BROWSER_API_KEY", raising=False)


@pytest.fixture
def test_client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Validation Helpers & Redaction
# ---------------------------------------------------------------------------

def test_validate_coord_valid():
    """Valid coordinates must not raise exceptions."""
    _validate_coord(28.6139, 77.2090, "Delhi")
    _validate_coord(0.0, 0.0, "Origin")
    _validate_coord(-89.9, -179.9, "SouthWest")
    _validate_coord(90.0, 180.0, "NorthEast")


def test_validate_coord_invalid_latitude():
    """Latitude outside [-90, 90] must raise ValueError."""
    with pytest.raises(ValueError, match="latitude 91.0 is out of range"):
        _validate_coord(91.0, 77.0)

    with pytest.raises(ValueError, match="latitude -95.0 is out of range"):
        _validate_coord(-95.0, 77.0)


def test_validate_coord_invalid_longitude():
    """Longitude outside [-180, 180] must raise ValueError."""
    with pytest.raises(ValueError, match="longitude 181.0 is out of range"):
        _validate_coord(28.0, 181.0)

    with pytest.raises(ValueError, match="longitude -190.0 is out of range"):
        _validate_coord(28.0, -190.0)


def test_redact_key():
    """Keys must be redacted to preserve privacy in logs."""
    assert _redact_key("") == "***"
    assert _redact_key("short") == "***"
    assert _redact_key("1234567") == "***"

    key = "AIzaSyD1234567890abcdef"
    redacted = _redact_key(key)
    assert redacted.startswith("AIza...")
    assert redacted.endswith("cdef")
    assert key not in redacted  # Full key is never printed


# ---------------------------------------------------------------------------
# 2. Rate Limiter
# ---------------------------------------------------------------------------

def test_rate_limiter_allows_up_to_max():
    """Rate limiter allows requests up to max_requests and rejects subsequent."""
    limiter = _SimpleRateLimiter(max_requests=3, window_seconds=60.0)

    # First 3 should pass
    for i in range(3):
        allowed, msg = limiter.check()
        assert allowed is True
        assert msg == "ok"

    # 4th should be blocked
    allowed, msg = limiter.check()
    assert allowed is False
    assert "Rate limit: 3 requests per 60s" in msg


# ---------------------------------------------------------------------------
# 3. Configuration Helpers
# ---------------------------------------------------------------------------

def test_config_status_offline_mode(clean_env):
    """When env vars are absent, config status reports offline mode."""
    status = get_google_maps_config_status()
    assert status["server_key_configured"] is False
    assert status["browser_key_configured"] is False
    assert status["fully_configured"] is False
    assert status["offline_mode"] is True
    assert status["server_key_hint"] is None
    assert status["browser_key_hint"] is None
    assert "ToS" in status["tos_note"] or "Google Maps" in status["tos_note"]
    assert get_browser_api_key_for_frontend() is None


def test_config_status_configured(monkeypatch):
    """When env vars are set, config status reports active mode with redacted hints."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyServerKey123456789")
    monkeypatch.setenv("GOOGLE_MAPS_BROWSER_API_KEY", "AIzaSyBrowserKey123456789")

    status = get_google_maps_config_status()
    assert status["server_key_configured"] is True
    assert status["browser_key_configured"] is True
    assert status["fully_configured"] is True
    assert status["offline_mode"] is False
    assert status["server_key_hint"] == "AIza...6789"
    assert status["browser_key_hint"] == "AIza...6789"
    assert get_browser_api_key_for_frontend() == "AIzaSyBrowserKey123456789"


# ---------------------------------------------------------------------------
# 4. Client Behavior: Offline & Unconfigured
# ---------------------------------------------------------------------------

def test_client_unconfigured_returns_offline_response(clean_env):
    """Unconfigured client must return NOT_CONFIGURED without crashing."""
    client = GoogleMapsClient()
    assert not client.is_configured

    async def _run():
        async with client:
            return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "NOT_CONFIGURED"
    assert "GOOGLE_MAPS_SERVER_API_KEY is not set" in res.error_message
    assert len(res.routes) == 0


def test_client_invalid_coordinates(monkeypatch):
    """Client catches out-of-bounds coordinates before network call."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyDummyKey123456789")
    client = GoogleMapsClient()

    async def _run():
        async with client:
            return await client.compute_routes(95.0, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "INVALID_COORDINATES"
    assert "out of range" in res.error_message


# ---------------------------------------------------------------------------
# 5. Client Behavior: Mocked HTTP Error Handling
# ---------------------------------------------------------------------------

def test_client_auth_failure_403(monkeypatch):
    """HTTP 403 maps to AUTH_FAILURE."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyInvalidKey123456")
    client = GoogleMapsClient()

    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "Forbidden - invalid API key"

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "AUTH_FAILURE"
    assert "HTTP 403" in res.error_message


def test_client_quota_exceeded_429(monkeypatch):
    """HTTP 429 maps to QUOTA_EXCEEDED."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyValidKey123456789")
    client = GoogleMapsClient()

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "Quota exceeded"

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "QUOTA_EXCEEDED"
    assert "HTTP 429" in res.error_message


def test_client_timeout_handling(monkeypatch):
    """httpx.TimeoutException maps to TIMEOUT."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyValidKey123456789")
    client = GoogleMapsClient(timeout_s=1.0)

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ReadTimeout("Request timed out")
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "TIMEOUT"
    assert "timed out" in res.error_message


def test_client_network_error_handling(monkeypatch):
    """httpx.RequestError maps to NETWORK_ERROR."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyValidKey123456789")
    client = GoogleMapsClient()

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "NETWORK_ERROR"


# ---------------------------------------------------------------------------
# 6. Response Parsing & ToS Boundary Compliance
# ---------------------------------------------------------------------------

def test_client_successful_route_parsing(monkeypatch):
    """Successful Google Routes response is parsed into typed RouteResult objects."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyValidKey123456789")
    client = GoogleMapsClient()

    mock_json = {
        "routes": [
            {
                "distanceMeters": 18450,
                "duration": "1420s",
                "staticDuration": "1200s",
                "polyline": {"encodedPolyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"},
                "description": "Via NH48 / Delhi-Noida Direct",
                "warnings": ["Toll required on this route"],
            },
            {
                "distanceMeters": 21000,
                "duration": "1650s",
                "staticDuration": "1400s",
                "polyline": {"encodedPolyline": "g~iF~ps|U_ulLnnqC"},
                "description": "Via Ring Road",
                "warnings": [],
            },
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_json

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is True
    assert res.error_code is None
    assert len(res.routes) == 2

    r0 = res.routes[0]
    assert r0.route_index == 0
    assert r0.distance_meters == 18450
    assert r0.duration_seconds == 1420
    assert r0.static_duration_seconds == 1200
    assert r0.encoded_polyline == "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    assert r0.description == "Via NH48 / Delhi-Noida Direct"
    assert r0.warnings == ["Toll required on this route"]

    # Check ToS serialization behavior
    d = res.to_dict()
    assert d["success"] is True
    for route_dict in d["routes"]:
        assert route_dict["_display_only"] is True
        assert "Google Maps Platform ToS" in route_dict["_tos_note"]
        assert "must not be used to train or evaluate ML models" in route_dict["_tos_note"]


def test_client_empty_routes_returns_no_route(monkeypatch):
    """200 OK with empty routes array returns NO_ROUTE."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyValidKey123456789")
    client = GoogleMapsClient()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"routes": []}

    async def _run():
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_resp
            async with client:
                return await client.compute_routes(28.6139, 77.2090, 28.5355, 77.3910)

    res = asyncio.run(_run())
    assert res.success is False
    assert res.error_code == "NO_ROUTE"


# ---------------------------------------------------------------------------
# 7. FastAPI Router Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_status_endpoint(test_client, clean_env):
    """GET /integrations/google/status returns 200 and offline status."""
    res = test_client.get("/integrations/google/status")
    assert res.status_code == 200
    data = res.json()
    assert "offline_mode" in data
    assert data["offline_mode"] is True
    assert "server_key_hint" in data
    assert data["server_key_hint"] is None


def test_api_config_status_alias(test_client):
    """GET /integrations/google/config-status acts as alias."""
    res = test_client.get("/integrations/google/config-status")
    assert res.status_code == 200
    assert "offline_mode" in res.json()


def test_api_routes_unconfigured(test_client, clean_env):
    """POST /integrations/google/routes without key returns NOT_CONFIGURED safely."""
    payload = {
        "origin_lat": 28.6139,
        "origin_lon": 77.2090,
        "destination_lat": 28.5355,
        "destination_lon": 77.3910,
        "routing_preference": "TRAFFIC_AWARE",
        "travel_mode": "DRIVE",
        "compute_alternatives": True,
    }
    res = test_client.post("/integrations/google/routes", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert data["error_code"] == "NOT_CONFIGURED"
    assert "_attribution" in data
    assert "_tos_boundary" in data
    assert "display only" in data["_tos_boundary"]


def test_api_routes_pydantic_validation_invalid_lat(test_client):
    """Invalid latitude in request triggers 422 Unprocessable Entity."""
    payload = {
        "origin_lat": 120.0,  # Invalid
        "origin_lon": 77.2090,
        "destination_lat": 28.5355,
        "destination_lon": 77.3910,
    }
    res = test_client.post("/integrations/google/routes", json=payload)
    assert res.status_code == 422


def test_api_routes_pydantic_validation_invalid_pref(test_client):
    """Invalid routing preference triggers 422 Unprocessable Entity."""
    payload = {
        "origin_lat": 28.6139,
        "origin_lon": 77.2090,
        "destination_lat": 28.5355,
        "destination_lon": 77.3910,
        "routing_preference": "SUPER_FAST_INVALID",
    }
    res = test_client.post("/integrations/google/routes", json=payload)
    assert res.status_code == 422


def test_api_routes_pydantic_validation_invalid_departure_time(test_client):
    """Malformed RFC3339 timestamp triggers 422."""
    payload = {
        "origin_lat": 28.6139,
        "origin_lon": 77.2090,
        "destination_lat": 28.5355,
        "destination_lon": 77.3910,
        "departure_time": "yesterday-at-noon",
    }
    res = test_client.post("/integrations/google/routes", json=payload)
    assert res.status_code == 422


def test_api_routes_mocked_success(test_client, monkeypatch):
    """Mocked live route computation through FastAPI endpoint."""
    monkeypatch.setenv("GOOGLE_MAPS_SERVER_API_KEY", "AIzaSyMockedKey123456789")

    mock_routes = [
        RouteResult(
            route_index=0,
            distance_meters=15000,
            duration_seconds=1100,
            static_duration_seconds=950,
            encoded_polyline="_p~iF~ps|U_ulLnnqC",
            description="Route 1",
            warnings=[],
        )
    ]
    mock_resp = GoogleRoutesResponse(
        request_id="test-req-123",
        success=True,
        routes=mock_routes,
        api_latency_ms=120.5,
    )

    with patch.object(GoogleMapsClient, "compute_routes", new_callable=AsyncMock) as mock_comp:
        mock_comp.return_value = mock_resp

        payload = {
            "origin_lat": 28.6139,
            "origin_lon": 77.2090,
            "destination_lat": 28.5355,
            "destination_lon": 77.3910,
            "routing_preference": "TRAFFIC_AWARE_OPTIMAL",
        }
        res = test_client.post("/integrations/google/routes", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert len(data["routes"]) == 1
        assert data["routes"][0]["distance_meters"] == 15000
        assert data["routes"][0]["duration_seconds"] == 1100
        assert data["routes"][0]["_display_only"] is True
        assert "_attribution" in data
        assert "_tos_boundary" in data
