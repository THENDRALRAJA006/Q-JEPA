#!/usr/bin/env bash
# start_local.sh — Unix/macOS start script for Surakshit Bharat Q-JEPA
set -e

echo "=== Surakshit Bharat Q-JEPA Traffic ==="
echo "Starting local development environment..."

# Step 1: Generate data if missing
if [ ! -f "data/synthetic/synthetic_traffic_normal.parquet" ]; then
    echo "[1/4] Generating synthetic data..."
    python scripts/generate_demo_data.py
else
    echo "[1/4] Synthetic data already exists."
fi

# Step 2: Train models if missing
if [ ! -f "models/checkpoints/lstm.pt" ]; then
    echo "[2/4] Training forecasting models (this takes ~2 minutes)..."
    python scripts/train_baselines.py
else
    echo "[2/4] Model checkpoints already exist."
fi

# Step 3: Start API backend in background
echo "[3/4] Starting FastAPI backend on port 8000..."
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!
sleep 3

# Step 4: Start dashboard
echo "[4/4] Starting Streamlit dashboard on port 8501..."
echo ""
echo "  Dashboard: http://localhost:8501"
echo "  API docs:  http://localhost:8000/docs"
echo ""
streamlit run dashboard/app.py --server.port 8501

# Cleanup
kill $API_PID 2>/dev/null || true
