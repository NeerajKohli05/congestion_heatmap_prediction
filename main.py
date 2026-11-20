"""
Main deployment entrypoint for FastAPI application.
Enables cloud platforms (Render, Railway, Hugging Face Spaces, Fly.io, Koyeb, AWS)
to discover and launch the EDA Congestion Heatmap Predictor dashboard.
"""
import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from web_app.server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
