# Google Maps Platform Setup Guide — Surakshit Bharat Q-JEPA

This guide details the procedure for configuring the **Google Maps Platform** integration in Surakshit Bharat Q-JEPA Traffic.

---

## 1. Overview & Architecture Separation

The Google Maps integration connects real-world geographic routing with the Surakshit Bharat platform while strictly adhering to Google Maps Platform Terms of Service and data-hygiene boundaries.

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│  - Streamlit UI (dashboard/components/google_maps_tab.py)   │
│  - Embeds Google Maps JavaScript API iframe                │
│  - Uses GOOGLE_MAPS_BROWSER_API_KEY (HTTP referrer restricted)│
└──────────────────────────────▲──────────────────────────────┘
                               │
                               │ Route Requests
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│  - Endpoint: POST /integrations/google/routes               │
│  - Client: src/integrations/google_maps_client.py           │
│  - Calls Google Routes API via HTTPS                        │
│  - Uses GOOGLE_MAPS_SERVER_API_KEY (API restricted)         │
│  - Enforces local rate limiting (10 req / 60 sec)           │
└─────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Strict Two-Key Architecture**:
> Never use a single unrestricted API key for both frontend and backend.
> - **`GOOGLE_MAPS_SERVER_API_KEY`**: Kept confidential on the server, only accesses Google Routes API.
> - **`GOOGLE_MAPS_BROWSER_API_KEY`**: Sent only to the client browser, restricted by HTTP referrer to load Google Maps tiles.

---

## 2. Google Cloud Platform (GCP) Setup

### Step 2.1: Create or Select a GCP Project
1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (e.g., `surakshit-bharat-traffic`) or select an existing project.
3. Link an active billing account (Google provides a $200 recurring monthly credit for Maps Platform).

### Step 2.2: Enable Required APIs
Navigate to **APIs & Services > Library** and enable:
1. **Routes API** (`routes.googleapis.com`) — required for dynamic vehicle routing and traffic-aware ETA calculation.
2. **Maps JavaScript API** (`maps-backend.googleapis.com`) — required for interactive map rendering in the Streamlit dashboard.
3. *(Optional)* **Places API** / **Geocoding API** — for future address geocoding.

---

## 3. Creating & Restricting API Keys

### Key 1: Server Key (`GOOGLE_MAPS_SERVER_API_KEY`)
1. In Cloud Console, go to **APIs & Services > Credentials**.
2. Click **Create Credentials > API key**. Name it `q-jepa-server-routes-key`.
3. Under **API restrictions**:
   - Select **Restrict key**.
   - Check **Routes API** only.
4. Under **Application restrictions**:
   - For production: Select **IP addresses** and enter your server IP(s).
   - For local development: Set to **None** or leave IP restricted to your gateway.
5. Save the key.

### Key 2: Browser Key (`GOOGLE_MAPS_BROWSER_API_KEY`)
1. Click **Create Credentials > API key**. Name it `q-jepa-browser-maps-key`.
2. Under **API restrictions**:
   - Select **Restrict key**.
   - Check **Maps JavaScript API** only.
3. Under **Application restrictions**:
   - Select **Websites (HTTP referrers)**.
   - Add:
     - `http://localhost:8501/*` (Streamlit default port)
     - `http://127.0.0.1:8501/*`
     - Your production web dashboard domain/URL if deployed.
4. Save the key.

---

## 4. Local Configuration

### Step 4.1: Populate Environment Variables
In the project root, edit your `.env` file (copied from `.env.example`):

```bash
# Google Maps Platform Credentials
GOOGLE_MAPS_SERVER_API_KEY=AIzaSy...your_server_key...
GOOGLE_MAPS_BROWSER_API_KEY=AIzaSy...your_browser_key...
```

Alternatively, export them directly in your shell:
```powershell
$env:GOOGLE_MAPS_SERVER_API_KEY="AIzaSy..."
$env:GOOGLE_MAPS_BROWSER_API_KEY="AIzaSy..."
```

> [!NOTE]
> `.env` is included in `.gitignore` and must **never** be committed to version control.

---

## 5. Offline & Fallback Mode

If keys are **not set**, the application runs smoothly in **Offline Mode**:
- The backend returns `status: "offline_mode"` and HTTP 200 with clear guidance.
- The dashboard displays an amber informational banner and a fallback map container.
- All internal JEPAGraph models, simulator components, and benchmarks continue to operate without interruption.
- Tests pass in offline mode without requiring network calls or API credentials.

---

## 6. Rate Limiting & Cost Guardrails

1. **Server-Side Token Bucket**:
   - `src/integrations/google_maps_client.py` includes a built-in rate limiter limiting requests to 10 requests per 60 seconds per process.
2. **Client-Side Debouncing**:
   - `dashboard/components/google_maps_tab.py` enforces a minimum 5-second interval between user button triggers in the UI.
3. **Cloud Console Budgets**:
   - Set a budget alert at $10.00 and $50.00 in the GCP Billing Console to avoid unexpected overages beyond the free tier.

---

## 7. Verification & Health Check

### Backend Status Check
Start the FastAPI server:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

In another terminal, test the status endpoint:
```bash
curl http://localhost:8000/integrations/google/status
```

Expected response (with keys configured):
```json
{
  "configured": true,
  "offline_mode": false,
  "server_key_set": true,
  "browser_key_set": true,
  "rate_limit_per_minute": 10,
  "service": "Google Maps Platform (Routes API + Maps JS API)"
}
```

### Route Request Check
```bash
curl -X POST http://localhost:8000/integrations/google/routes \
  -H "Content-Type: application/json" \
  -d '{
    "origin_lat": 28.6139,
    "origin_lng": 77.2090,
    "dest_lat": 28.5355,
    "dest_lng": 77.3910,
    "routing_preference": "TRAFFIC_AWARE_OPTIMAL"
  }'
```

---

## 8. Terms of Service & Compliance Boundaries

> [!WARNING]
> **Machine Learning Ground-Truth Boundary**:
> Under Google Maps Platform Terms of Service:
> 1. **No ML Pretraining or Fine-Tuning**: Data obtained from Google APIs (polylines, duration in traffic, distance, step instructions) **must never** be used to train, evaluate, or benchmark machine learning models (including `JEPAGraph-FM`).
> 2. **No Data Scraping or Caching**: Route geometries may not be stored in training datasets (Parquet, CSV, HDF5, or PyTorch checkpoints).
> 3. **Attribution**: The `© Google` attribution notice must always remain visible alongside any Google Maps visual output.
> 4. **Display Only**: The integration is strictly a real-time geographic overlay and baseline reference for human operators.
