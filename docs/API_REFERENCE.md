# 📖 API & Developer Integration Reference

This guide covers programmatic access to the **Residual Attention U-Net Routing Congestion Predictor**, including Python SDK integration, CLI usage, and REST API endpoints.

---

## 1. Python SDK / In-Code Inference

Use the trained PyTorch checkpoint directly inside your Python physical design or optimization scripts:

```python
import torch
import numpy as np
from src.models.unet import CongestionUNet

# 1. Load trained checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("outputs/checkpoints/best_congestion_unet.pt", map_location=device)

model = CongestionUNet(in_channels=6, out_channels=2, base_channels=32).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Prepare 6-channel input feature tensor (1, 6, 128, 128)
# Features: [Pin Density, Cell Density, Short RUDY, Long RUDY, Total RUDY, Macro Mask]
data = np.load("data/circuitnet_processed_v2/circuit_0000_nvdla-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz")
features = torch.from_numpy(data["features"]).unsqueeze(0).to(device)

# 3. Predict routing congestion
with torch.no_grad():
    prediction = model(features).squeeze(0).cpu().numpy()

# Channel 0: Horizontal Congestion Utilization (0.0 to 1.0)
# Channel 1: Vertical Congestion Utilization (0.0 to 1.0)
h_congestion = prediction[0]
v_congestion = prediction[1]

print(f"Prediction successful! H-Max: {h_congestion.max():.3f}, V-Max: {v_congestion.max():.3f}")
```

---

## 2. Command-Line Interface (CLI)

### Test Evaluation & Visualizer
Evaluate checkpoints across held-out test designs and generate 12-panel inspection plots:

```bash
python3 src/pipeline/test_evaluate.py \
    --checkpoint outputs/checkpoints/best_congestion_unet.pt \
    --data_dir data/circuitnet_processed_v2 \
    --output_dir outputs/test_results_v2 \
    --max_vis 8
```

### Ingest New Hugging Face Designs
Download and unpack other CircuitNet architectures (e.g. `nvdla-large`, `RISCY`, `Vortex`):

```bash
python3 src/data/import_circuitnet.py --design nvdla-large
python3 src/data/build_v2_dataset.py
```

### Retrain / Fine-Tune Model
Train on custom datasets with configurable hyperparameters:

```bash
python3 src/pipeline/train_circuitnet.py \
    --config configs/circuitnet_v2_config.yaml \
    --epochs 20
```

---

## 3. REST API Reference (FastAPI Server)

The web server exposes REST endpoints for automated CI/CD and EDA integration.

### `GET /api/presets`
Returns a list of all preprocessed semiconductor placement designs available for inference.

**Response:**
```json
{
  "presets": [
    "circuit_0000_nvdla-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap.npz",
    "circuit_0001_nvdla-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ar.npz"
  ]
}
```

### `POST /api/predict`
Executes forward inference on a preset design name or uploaded `.npz` file.

**Parameters (Form-Data or Multipart):**
* `preset` *(string, optional)*: Filename of an existing preset design.
* `file` *(binary, optional)*: Uploaded custom `.npz` file containing `features` and optional `target`.

**Response:**
```json
{
  "design_name": "circuit_0000_nvdla-small.npz",
  "latency_ms": 32.4,
  "metrics": {
    "mean_pcc": 0.7898,
    "mean_nmse": 0.2422,
    "mae": 0.0582,
    "mean_hotspot_f1_top5pct": 0.5379
  },
  "images": {
    "gt_h": "<base64_png>",
    "pred_h": "<base64_png>",
    "gt_v": "<base64_png>",
    "pred_v": "<base64_png>",
    "err_h": "<base64_png>",
    "err_v": "<base64_png>",
    "feat_pin": "<base64_png>",
    "feat_cell": "<base64_png>",
    "feat_rudy_h": "<base64_png>",
    "feat_rudy_v": "<base64_png>",
    "feat_rudy_tot": "<base64_png>",
    "feat_macro": "<base64_png>"
  }
}
```
