"""
Main application entry point for Customer Churn Prediction System.
Run: python app.py
"""

import os
import sys
import threading
import time
import webbrowser

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend import create_app
from ml.model_utils import load_artifacts, get_model_paths

def check_and_prepare_artifacts():
    """Verifies that model artifacts exist, automatically trains if missing."""
    paths = get_model_paths()
    if not os.path.exists(paths['model']) or not os.path.exists(paths['preprocessor']):
        print("\n[INFO] ML model artifacts not detected. Running automated training pipeline...")
        from ml.train_model import train_models
        train_models()
        print("[INFO] Model pipeline completed successfully.\n")

def open_browser():
    """Opens default browser after a brief delay for server initialization."""
    time.sleep(1.5)
    try:
        webbrowser.open("http://127.0.0.1:5000/")
    except Exception:
        pass

app = create_app()

if __name__ == '__main__':
    # Verify model readiness
    check_and_prepare_artifacts()
    _, _, metadata = load_artifacts()
    model_name = metadata.get('model_name', 'Random Forest') if metadata else 'Random Forest'

    print("=" * 44)
    print("  CUSTOMER CHURN PREDICTION SYSTEM")
    print(f"  Model:     {model_name}")
    print("  Status:    Ready")
    print("  Dashboard: http://127.0.0.1:5000/")
    print("  Press CTRL+C to stop.")
    print("=" * 44)

    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
