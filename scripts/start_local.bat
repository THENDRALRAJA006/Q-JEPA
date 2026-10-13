@echo off
REM start_local.bat — Windows batch script to start Surakshit Bharat Q-JEPA locally
echo === Surakshit Bharat Q-JEPA Traffic ===
echo Starting local development environment...

REM Step 1: Generate data if missing
IF NOT EXIST "data\synthetic\synthetic_traffic_normal.parquet" (
    echo [1/4] Generating synthetic data...
    python scripts\generate_demo_data.py
) ELSE (
    echo [1/4] Synthetic data already exists.
)

REM Step 2: Train models if missing
IF NOT EXIST "models\checkpoints\lstm.pt" (
    echo [2/4] Training forecasting models ^(this takes ~2 minutes^)...
    python scripts\train_baselines.py
) ELSE (
    echo [2/4] Model checkpoints already exist.
)

REM Step 3: Start API backend in background
echo [3/4] Starting FastAPI backend on port 8000...
start /B python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level warning

REM Wait for API to start
timeout /t 3 /nobreak >nul

REM Step 4: Start Streamlit dashboard
echo [4/4] Starting Streamlit dashboard on port 8501...
echo.
echo Dashboard: http://localhost:8501
echo API docs:  http://localhost:8000/docs
echo.
streamlit run dashboard\app.py --server.port 8501
