"""
Google Maps Integration Page — Surakshit Bharat Q-JEPA Traffic Dashboard.

Displays:
  - Google Maps interactive map (Maps JavaScript API, browser key)
  - Traffic-aware route from Google Routes API (via our FastAPI backend)
  - Comparison with our internal JEPAGraph routing strategies
  - Offline mode when credentials are absent
  - Clear labelling: real Google data vs simulated model data

ToS boundaries enforced in this component:
  - Google route data is ONLY used for display.
  - No Google route content is written to ML datasets, metrics, or checkpoints.
  - Attribution is always displayed alongside map content.
  - Polylines are decoded for map rendering only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components

from src.integrations.google_maps_client import (
    get_google_maps_config_status,
    get_browser_api_key_for_frontend,
    _validate_coord,
)
from src.simulation.road_network import RoadNetwork
from src.routing.shortest_path import compare_all_routes


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_gmaps_state():
    if "gmaps_last_result" not in st.session_state:
        st.session_state.gmaps_last_result = None
    if "gmaps_request_count" not in st.session_state:
        st.session_state.gmaps_request_count = 0
    if "gmaps_last_request_ts" not in st.session_state:
        st.session_state.gmaps_last_request_ts = 0.0


# ---------------------------------------------------------------------------
# Internal route comparison helper
# ---------------------------------------------------------------------------

def _get_internal_routes(network: RoadNetwork, origin: str, dest: str) -> List[Dict[str, Any]]:
    """Run our internal routing strategies on the simulation network."""
    try:
        routes = compare_all_routes(network, origin, dest, risk_penalty=0.3)
        return [
            {
                "name": r.router_type.value.replace("_", " ").title(),
                "path": r.path,
                "distance_m": r.total_distance_m,
                "eta_s": r.estimated_travel_time_s,
                "uncertainty_s": r.uncertainty_s,
                "feasible": r.feasible,
                "explanation": r.explanation,
            }
            for r in routes
        ]
    except Exception as exc:
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# Google Maps HTML component builder
# ---------------------------------------------------------------------------

def _build_maps_html(
    browser_key: str,
    center_lat: float = 12.9716,
    center_lon: float = 77.5946,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    dest_lat: Optional[float] = None,
    dest_lon: Optional[float] = None,
    encoded_polylines: Optional[List[str]] = None,
    zoom: int = 12,
) -> str:
    """Build a self-contained HTML page with embedded Google Maps JavaScript API."""

    # Build JS for route polylines (display only)
    polyline_js = ""
    if encoded_polylines:
        for i, poly in enumerate(encoded_polylines[:3]):  # Max 3 alternatives
            color = ["#c0392b", "#2980b9", "#27ae60"][i % 3]
            opacity = 1.0 if i == 0 else 0.5
            polyline_js += f"""
            var path{i} = google.maps.geometry.encoding.decodePath("{poly}");
            var route{i} = new google.maps.Polyline({{
                path: path{i},
                geodesic: true,
                strokeColor: "{color}",
                strokeOpacity: {opacity},
                strokeWeight: {5 - i},
            }});
            route{i}.setMap(map);
            """

    # Origin/destination markers
    markers_js = ""
    if origin_lat is not None and origin_lon is not None:
        markers_js += f"""
        var originMarker = new google.maps.Marker({{
            position: {{lat: {origin_lat}, lng: {origin_lon}}},
            map: map,
            title: "Origin",
            icon: {{
                path: google.maps.SymbolPath.CIRCLE,
                scale: 10,
                fillColor: "#27ae60",
                fillOpacity: 1,
                strokeColor: "#fff",
                strokeWeight: 2,
            }},
        }});
        """
    if dest_lat is not None and dest_lon is not None:
        markers_js += f"""
        var destMarker = new google.maps.Marker({{
            position: {{lat: {dest_lat}, lng: {dest_lon}}},
            map: map,
            title: "Destination",
            icon: {{
                path: google.maps.SymbolPath.CIRCLE,
                scale: 10,
                fillColor: "#c0392b",
                fillOpacity: 1,
                strokeColor: "#fff",
                strokeWeight: 2,
            }},
        }});
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Surakshit Bharat — Google Maps Route</title>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; background: #0a0f1e; }}
    #attribution {{
      position: absolute; bottom: 4px; left: 4px; z-index: 999;
      font-size: 10px; color: #aaa; background: rgba(0,0,0,0.5);
      padding: 2px 6px; border-radius: 3px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="attribution">Map data &copy; Google | Display only &mdash; not for ML training</div>
  <script>
    function initMap() {{
      var center = {{lat: {center_lat}, lng: {center_lon}}};
      var map = new google.maps.Map(document.getElementById("map"), {{
        zoom: {zoom},
        center: center,
        mapTypeId: "roadmap",
        styles: [
          {{elementType:"geometry",stylers:[{{color:"#1a2035"}}]}},
          {{elementType:"labels.text.fill",stylers:[{{color:"#8ec3b9"}}]}},
          {{elementType:"labels.text.stroke",stylers:[{{color:"#1a3646"}}]}},
          {{featureType:"road",elementType:"geometry",stylers:[{{color:"#304a7d"}}]}},
          {{featureType:"water",elementType:"geometry",stylers:[{{color:"#0e1626"}}]}},
        ],
      }});
      {markers_js}
      {polyline_js}
    }}
  </script>
  <script
    src="https://maps.googleapis.com/maps/api/js?key={browser_key}&libraries=geometry&callback=initMap"
    async defer>
  </script>
</body>
</html>
"""
    return html


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_google_maps_tab():
    """Render the Google Maps Integration page inside the Streamlit dashboard."""
    _init_gmaps_state()

    # Header
    st.markdown("""
    <div class="page-header">
      <h1 style="color:#4fc3f7;margin:0;">🗺️ Google Maps Integration</h1>
      <p style="color:#7a9bc7;margin:0.3rem 0 0 0;">
        Traffic-aware routing via Google Maps Platform + JEPAGraph-FM comparison
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Config status banner
    cfg = get_google_maps_config_status()
    browser_key = get_browser_api_key_for_frontend()

    if cfg["fully_configured"]:
        st.markdown(
            f'<div class="alert-green">✅ <b>Google Maps configured</b> — '
            f'server key: <code>{cfg["server_key_hint"]}</code> &nbsp;|&nbsp; '
            f'browser key: <code>{cfg["browser_key_hint"]}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        missing = []
        if not cfg["server_key_configured"]:
            missing.append("<code>GOOGLE_MAPS_SERVER_API_KEY</code>")
        if not cfg["browser_key_configured"]:
            missing.append("<code>GOOGLE_MAPS_BROWSER_API_KEY</code>")
        st.markdown(
            f'<div class="alert-yellow">⚠️ <b>Offline mode</b> — '
            f'Missing: {", ".join(missing)}. '
            f'See <a href="docs/GOOGLE_MAPS_SETUP.md" style="color:#ffe082;">docs/GOOGLE_MAPS_SETUP.md</a> '
            f'to configure. The simulation routing below remains fully functional.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Layout ──
    col_inputs, col_map = st.columns([1, 2], gap="medium")

    with col_inputs:
        st.markdown("### 📍 Route Request")

        st.markdown("**Origin coordinates**")
        c1, c2 = st.columns(2)
        with c1:
            origin_lat = st.number_input(
                "Lat", value=12.9716, min_value=-90.0, max_value=90.0,
                step=0.0001, format="%.4f", key="gmap_origin_lat",
            )
        with c2:
            origin_lon = st.number_input(
                "Lon", value=77.5946, min_value=-180.0, max_value=180.0,
                step=0.0001, format="%.4f", key="gmap_origin_lon",
            )

        st.markdown("**Destination coordinates**")
        c3, c4 = st.columns(2)
        with c3:
            dest_lat = st.number_input(
                "Lat", value=12.9352, min_value=-90.0, max_value=90.0,
                step=0.0001, format="%.4f", key="gmap_dest_lat",
            )
        with c4:
            dest_lon = st.number_input(
                "Lon", value=77.6245, min_value=-180.0, max_value=180.0,
                step=0.0001, format="%.4f", key="gmap_dest_lon",
            )

        routing_pref = st.selectbox(
            "Routing preference",
            ["TRAFFIC_AWARE", "TRAFFIC_AWARE_OPTIMAL", "TRAFFIC_UNAWARE"],
            index=0,
            key="gmap_routing_pref",
        )
        show_alternatives = st.checkbox("Request alternative routes", value=True, key="gmap_alts")

        # Cost/rate note
        st.caption(
            "ℹ️ Each route request uses the Google Routes API. "
            "Review pricing at [developers.google.com/maps/billing-and-pricing/pricing-india]"
            "(https://developers.google.com/maps/billing-and-pricing/pricing-india)."
        )

        request_btn = st.button(
            "🚗 Request Google Route",
            disabled=not cfg["server_key_configured"],
            use_container_width=True,
            key="gmap_request_btn",
        )

        if not cfg["server_key_configured"]:
            st.caption("⚠️ Button disabled — configure GOOGLE_MAPS_SERVER_API_KEY first.")

        # Trigger route request via our backend
        if request_btn:
            # Simple per-session rate guard
            now_ts = time.time()
            if now_ts - st.session_state.gmaps_last_request_ts < 5.0:
                st.warning("Please wait a few seconds before requesting another route.")
            else:
                with st.spinner("Calling Google Routes API via backend..."):
                    try:
                        import httpx as _httpx
                        payload = {
                            "origin_lat": origin_lat,
                            "origin_lon": origin_lon,
                            "destination_lat": dest_lat,
                            "destination_lon": dest_lon,
                            "routing_preference": routing_pref,
                            "compute_alternatives": show_alternatives,
                        }
                        r = _httpx.post(
                            "http://localhost:8000/integrations/google/routes",
                            json=payload,
                            timeout=20.0,
                        )
                        if r.status_code == 200:
                            st.session_state.gmaps_last_result = r.json()
                            st.session_state.gmaps_request_count += 1
                            st.session_state.gmaps_last_request_ts = now_ts
                        else:
                            st.error(f"Backend error {r.status_code}: {r.text[:300]}")
                    except Exception as exc:
                        # Show offline fallback hint
                        st.warning(
                            f"Could not reach backend (is `uvicorn src.api.main:app` running?). "
                            f"Error: {exc}"
                        )

    # ── Map panel ──
    with col_map:
        # Determine map center
        center_lat = (origin_lat + dest_lat) / 2
        center_lon = (origin_lon + dest_lon) / 2

        last = st.session_state.gmaps_last_result
        polylines: List[str] = []

        if last and last.get("success") and last.get("routes"):
            polylines = [
                r.get("encoded_polyline", "")
                for r in last["routes"]
                if r.get("encoded_polyline")
            ]

        if browser_key:
            html_content = _build_maps_html(
                browser_key=browser_key,
                center_lat=center_lat,
                center_lon=center_lon,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                dest_lat=dest_lat,
                dest_lon=dest_lon,
                encoded_polylines=polylines if polylines else None,
                zoom=12,
            )
            components.html(html_content, height=450, scrolling=False)
            st.caption("Map data © Google | Route display only — not for ML training or evaluation.")
        else:
            # Offline map fallback
            st.markdown(
                """
                <div style="background:#0d1a3a;border:1px solid #1e3a6e;border-radius:10px;
                  height:450px;display:flex;align-items:center;justify-content:center;
                  flex-direction:column;gap:1rem;">
                  <span style="font-size:3rem;">🗺️</span>
                  <p style="color:#7a9bc7;text-align:center;max-width:320px;">
                    <b>Map offline</b><br/>
                    Set <code>GOOGLE_MAPS_BROWSER_API_KEY</code> to enable the interactive map.<br/>
                    See <code>docs/GOOGLE_MAPS_SETUP.md</code>.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Route results ──
    st.divider()
    st.markdown("### 📊 Route Comparison")

    result_col, sim_col = st.columns(2, gap="medium")

    with result_col:
        st.markdown(
            "#### 🟢 Google Routes API Result\n"
            "<small style='color:#7a9bc7;'>Real traffic data. Display only — not for ML use.</small>",
            unsafe_allow_html=True,
        )
        last = st.session_state.gmaps_last_result
        if last is None:
            st.info("No route requested yet. Click **Request Google Route** above.")
        elif not last.get("success"):
            err_code = last.get("error_code", "UNKNOWN")
            err_msg = last.get("error_message", "Unknown error")
            if err_code == "NOT_CONFIGURED":
                st.warning(f"**Offline mode** — {err_msg}")
            elif err_code == "RATE_LIMITED":
                st.warning(f"**Rate limited** — {err_msg}")
            elif err_code in ("AUTH_FAILURE", "QUOTA_EXCEEDED"):
                st.error(f"**{err_code}** — {err_msg}")
            else:
                st.error(f"**{err_code}** — {err_msg}")
        else:
            st.success(f"✅ {len(last['routes'])} route(s) returned")
            for route in last.get("routes", []):
                dist_km = route["distance_meters"] / 1000
                dur_min = route["duration_seconds"] / 60
                static_min = route["static_duration_seconds"] / 60
                delay_min = dur_min - static_min
                idx = route["route_index"] + 1
                label = "🥇 Primary route" if idx == 1 else f"🔀 Alternative {idx - 1}"

                with st.expander(f"{label} — {dist_km:.1f} km", expanded=(idx == 1)):
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Distance", f"{dist_km:.2f} km")
                    c_b.metric("ETA (traffic)", f"{dur_min:.0f} min")
                    c_c.metric(
                        "Traffic delay",
                        f"+{delay_min:.0f} min" if delay_min > 0 else "None",
                    )
                    st.caption(f"Static (no traffic): {static_min:.0f} min")
                    if route.get("description"):
                        st.caption(f"Route: {route['description']}")
                    if route.get("warnings"):
                        for w in route["warnings"]:
                            st.warning(w)
                    if route.get("_tos_note"):
                        st.caption(f"⚠️ {route['_tos_note']}")

            if last.get("api_latency_ms"):
                st.caption(f"API latency: {last['api_latency_ms']:.0f} ms")
            st.caption(f"Request ID: {last.get('request_id', 'n/a')}")
            st.caption(last.get("_attribution", "© Google"))

    with sim_col:
        st.markdown(
            "#### 🔵 JEPAGraph Simulation Routing\n"
            "<small style='color:#7a9bc7;'>Separate synthetic network — not a direct comparison.</small>",
            unsafe_allow_html=True,
        )
        st.info(
            "**Important:** Our simulation network is an independent synthetic grid. "
            "It does not share road topology, speed sensors, or trip pairs with Google's network. "
            "These results cannot be directly compared with Google route quality."
        )

        network: RoadNetwork = st.session_state.get(
            "network",
            RoadNetwork.create_synthetic_grid(rows=5, cols=6, seed=42),
        )
        nodes = network.node_list()
        sim_origin = st.selectbox("Simulation origin node", nodes, index=0, key="gmap_sim_origin")
        sim_dest = st.selectbox(
            "Simulation destination node", nodes, index=len(nodes) - 1, key="gmap_sim_dest",
        )

        if st.button("🔵 Run Simulation Routes", key="gmap_run_sim", use_container_width=True):
            if sim_origin == sim_dest:
                st.warning("Origin and destination must be different.")
            else:
                sim_routes = _get_internal_routes(network, sim_origin, sim_dest)
                st.session_state.gmaps_sim_routes = sim_routes

        sim_routes_result = st.session_state.get("gmaps_sim_routes")
        if sim_routes_result:
            for r in sim_routes_result:
                if "error" in r:
                    st.error(r["error"])
                    continue
                feasible_icon = "✅" if r.get("feasible") else "❌"
                with st.expander(f"{feasible_icon} {r['name']}", expanded=False):
                    if r.get("feasible"):
                        cA, cB = st.columns(2)
                        cA.metric("Distance", f"{r['distance_m']:.0f} m")
                        cB.metric("ETA", f"{r['eta_s']:.0f} s")
                        if r.get("uncertainty_s", 0) > 0:
                            st.caption(f"Uncertainty: ±{r['uncertainty_s']:.1f}s")
                        if r.get("path"):
                            st.caption(f"Path: {' → '.join(r['path'])}")
                    else:
                        st.caption(r.get("explanation", "No feasible route found."))

        st.markdown(
            '<div class="alert-yellow" style="font-size:0.8rem;">'
            "⚠️ <b>Simulation data only.</b> NOT real traffic. "
            "NOT suitable for operational ambulance dispatch.</div>",
            unsafe_allow_html=True,
        )

    # ── ToS reminder footer ──
    st.divider()
    with st.expander("📋 Google Maps Platform — Usage Boundaries & ToS", expanded=False):
        st.markdown("""
**What this integration does:**
- Displays interactive maps using the **Maps JavaScript API** (browser key, display only).
- Computes traffic-aware routes using the **Routes API** (server key, via our backend).
- Compares results against our internal **JEPAGraph-FM simulation** (independent network).

**What this integration does NOT do (mandatory restrictions):**
- ❌ Does NOT store Google route polylines, durations, or distances in ML datasets.
- ❌ Does NOT use Google route data to train, fine-tune, or evaluate JEPAGraph.
- ❌ Does NOT scrape or bulk-download Google Maps content.
- ❌ Does NOT claim Google Maps provides ambulance pre-emption or emergency dispatch.
- ❌ Does NOT cache Google map content beyond what the ToS permits.

**Attribution:** All Google Maps content is labelled "© Google" in the map view.

**Pricing reference:** https://developers.google.com/maps/billing-and-pricing/pricing-india

**Setup guide:** `docs/GOOGLE_MAPS_SETUP.md`
        """)
