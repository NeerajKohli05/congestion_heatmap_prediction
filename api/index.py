"""
Vercel serverless entrypoint.
Vercel auto-detects api/index.py as the default Python handler.
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_app.server import app  # noqa: F401
