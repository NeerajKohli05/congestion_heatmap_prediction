"""
Root FastAPI entrypoint for cloud deployment.
Platforms (Render, Railway, HuggingFace Spaces, Fly.io) auto-detect main:app.
"""
import os
import sys

# Ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.server import app  # noqa: F401 — re-export for uvicorn

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
