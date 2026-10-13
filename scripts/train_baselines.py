"""Train all forecasting baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.forecasting.train import train_all

if __name__ == "__main__":
    results = train_all()
    print("\n=== Forecasting Results ===")
    for model, metrics in results.items():
        mae_val = metrics.get("MAE", "N/A")
        rmse_val = metrics.get("RMSE", "N/A")
        if isinstance(mae_val, float):
            print(f"  {model:25s}: MAE={mae_val:.4f}  RMSE={rmse_val:.4f}")
        else:
            print(f"  {model}: {metrics}")
