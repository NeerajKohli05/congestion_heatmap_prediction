"""
FastAPI Server for VLSI Routing Congestion Prediction.
Loads the trained Residual Attention U-Net and serves interactive predictions via web UI.
"""

import os
import io
import time
import base64
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, root_dir)

from src.models.unet import CongestionUNet
from src.metrics.eval_metrics import evaluate_all_metrics

app = FastAPI(title="EDA Routing Congestion Heatmap Predictor")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Load Model Checkpoint Candidates (supports both full and stripped weights)
CKPT_CANDIDATES = [
    os.path.join(root_dir, "outputs/checkpoints/model_weights.pt"),
    os.path.join(root_dir, "outputs/checkpoints/best_congestion_unet.pt"),
]
DATA_DIR = os.path.join(root_dir, "data/circuitnet_processed_v2")
DEMO_DIR = os.path.join(root_dir, "demo_samples")
BENCHMARK_REPORT = os.path.join(root_dir, "outputs/test_results_v2/results_report.txt")

device = torch.device("cpu")
models = {}


def get_available_checkpoints():
    return {
        os.path.basename(path): path
        for path in CKPT_CANDIDATES
        if os.path.isfile(path)
    }


def get_model(model_name: Optional[str] = None):
    available = get_available_checkpoints()
    if not available:
        raise FileNotFoundError("No trained model checkpoint is available")

    selected_name = model_name or next(iter(available))
    if selected_name not in available:
        raise ValueError(f"Unknown model checkpoint: {selected_name}")

    if selected_name not in models:
        checkpoint = torch.load(available[selected_name], map_location=device)
        model_config = checkpoint.get("config", {}).get("model", {})
        base_channels = model_config.get("base_channels", 32)
        model_instance = CongestionUNet(in_channels=6, out_channels=2, base_channels=base_channels)
        model_instance.load_state_dict(checkpoint["model_state_dict"])
        model_instance.eval()
        models[selected_name] = model_instance
        print(f"[*] Loaded checkpoint {selected_name} (base_channels={base_channels})")

    return models[selected_name]


def array_to_b64(arr: np.ndarray, cmap: str = "hot", vmin: Optional[float] = None, vmax: Optional[float] = None) -> str:
    """Converts 2D float array to base64 PNG image."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor('#020617')  # Match slate-950
    ax.set_facecolor('#020617')
    ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/presets")
async def list_presets():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(DEMO_DIR, "*.npz")))
    preset_names = [os.path.basename(f) for f in files]
    return {"presets": preset_names}


@app.get("/api/models")
async def list_models():
    models_available = []
    for model_name in get_available_checkpoints():
        models_available.append({
            "id": model_name,
            "name": "Best validation checkpoint" if model_name == "best_congestion_unet.pt" else "Deployed model weights",
            "architecture": "Residual Attention U-Net",
            "base_channels": 32,
            "trained": True,
        })
    return {"models": models_available}


@app.get("/api/benchmark")
async def benchmark_results():
    if not os.path.exists(BENCHMARK_REPORT):
        return JSONResponse({"error": "Saved benchmark report is unavailable"}, status_code=404)

    summary = {}
    samples = []
    in_sample_table = False

    with open(BENCHMARK_REPORT, "r", encoding="utf-8") as report_file:
        for line in report_file:
            line = line.strip()
            if line.startswith("Per-Sample Details:"):
                in_sample_table = True
                continue

            if in_sample_table:
                parts = line.split()
                if len(parts) == 5 and parts[0].endswith(".npz"):
                    try:
                        samples.append({
                            "file": parts[0],
                            "pcc": float(parts[1]),
                            "nmse": float(parts[2]),
                            "f1_top5": float(parts[3]),
                            "iou_top5": float(parts[4]),
                        })
                    except ValueError:
                        continue
                continue

            if line.startswith("Test samples"):
                summary["sample_count"] = int(line.split()[-1])
            elif line.startswith("Avg inference time"):
                summary["inference_ms"] = float(line.split()[-2])
            elif line.startswith("Mean PCC"):
                values = line.split()
                summary["mean_pcc"] = float(values[-3])
                summary["pcc_std"] = float(values[-1])
            elif line.startswith("Mean NMSE"):
                values = line.split()
                summary["mean_nmse"] = float(values[-3])
                summary["nmse_std"] = float(values[-1])
            elif line.startswith("Mean MAE"):
                summary["mean_mae"] = float(line.split()[-1])
            elif line.startswith("Mean RMSE"):
                summary["mean_rmse"] = float(line.split()[-1])
            elif line.startswith("Hotspot F1"):
                values = line.split()
                summary["hotspot_f1_top5"] = float(values[-3])
                summary["hotspot_f1_std"] = float(values[-1])
            elif line.startswith("Hotspot IoU"):
                values = line.split()
                summary["hotspot_iou_top5"] = float(values[-3])
                summary["hotspot_iou_std"] = float(values[-1])
            elif line.startswith("H-Congestion PCC"):
                summary["h_pcc"] = float(line.split()[-1])
            elif line.startswith("V-Congestion PCC"):
                summary["v_pcc"] = float(line.split()[-1])
            elif line.startswith("H-Congestion NMSE"):
                summary["h_nmse"] = float(line.split()[-1])
            elif line.startswith("V-Congestion NMSE"):
                summary["v_nmse"] = float(line.split()[-1])
            elif line.startswith("H-Congestion F1@5%"):
                summary["h_f1_top5"] = float(line.split()[-1])
            elif line.startswith("V-Congestion F1@5%"):
                summary["v_f1_top5"] = float(line.split()[-1])

    if not samples:
        return JSONResponse({"error": "No per-sample benchmark data found"}, status_code=500)

    return {"summary": summary, "samples": samples}


@app.post("/api/predict")
async def predict_congestion(
    preset: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    model_name: Optional[str] = Form(None),
):
    try:
        if file is not None and file.filename != "":
            contents = await file.read()
            data = np.load(io.BytesIO(contents))
            design_name = file.filename
        elif preset:
            fpath = os.path.join(DATA_DIR, preset)
            if not os.path.exists(fpath):
                fpath = os.path.join(DEMO_DIR, preset)
            if not os.path.exists(fpath):
                return JSONResponse({"error": f"Preset {preset} not found"}, status_code=404)
            data = np.load(fpath)
            design_name = preset
        else:
            return JSONResponse({"error": "No preset or file provided"}, status_code=400)

        features = data["features"].astype(np.float32)  # (6, 128, 128)
        target = data["target"].astype(np.float32)      # (2, 128, 128)

        net = get_model(model_name)

        # Inference
        t0 = time.time()
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0)
            pred = net(x).squeeze(0).numpy()
        latency = (time.time() - t0) * 1000

        # Calculate metrics
        metrics = evaluate_all_metrics(pred, target)
        metrics["mae"] = float(np.mean(np.abs(pred - target)))

        # Convert images to base64
        vmax_h = max(float(target[0].max()), float(pred[0].max()), 0.05)
        vmax_v = max(float(target[1].max()), float(pred[1].max()), 0.05)

        img_gt_h = array_to_b64(target[0], cmap="hot", vmin=0, vmax=vmax_h)
        img_pred_h = array_to_b64(pred[0], cmap="hot", vmin=0, vmax=vmax_h)
        img_gt_v = array_to_b64(target[1], cmap="hot", vmin=0, vmax=vmax_v)
        img_pred_v = array_to_b64(pred[1], cmap="hot", vmin=0, vmax=vmax_v)

        # Discrepancy Error Maps
        err_h = np.abs(pred[0] - target[0])
        err_v = np.abs(pred[1] - target[1])
        img_err_h = array_to_b64(err_h, cmap="inferno", vmin=0, vmax=max(float(err_h.max()), 0.05))
        img_err_v = array_to_b64(err_v, cmap="inferno", vmin=0, vmax=max(float(err_v.max()), 0.05))

        # Features
        img_feat_pin = array_to_b64(features[0], cmap="plasma")
        img_feat_cell = array_to_b64(features[1], cmap="magma")
        img_feat_rudy_h = array_to_b64(features[2], cmap="Blues")
        img_feat_rudy_v = array_to_b64(features[3], cmap="Greens")
        img_feat_rudy_tot = array_to_b64(features[4], cmap="viridis")
        img_feat_macro = array_to_b64(features[5], cmap="gray")

        return {
            "design_name": design_name,
            "model_name": model_name or next(iter(get_available_checkpoints())),
            "latency_ms": latency,
            "metrics": metrics,
            "images": {
                "gt_h": img_gt_h,
                "pred_h": img_pred_h,
                "gt_v": img_gt_v,
                "pred_v": img_pred_v,
                "err_h": img_err_h,
                "err_v": img_err_v,
                "feat_pin": img_feat_pin,
                "feat_cell": img_feat_cell,
                "feat_rudy_h": img_feat_rudy_h,
                "feat_rudy_v": img_feat_rudy_v,
                "feat_rudy_tot": img_feat_rudy_tot,
                "feat_macro": img_feat_macro
            }
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
