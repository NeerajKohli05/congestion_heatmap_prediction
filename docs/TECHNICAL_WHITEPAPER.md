# VLSI + AI/ML Routing Congestion Heatmap Prediction
## Technical Whitepaper & Architectural Specification

---

## 1. Executive Summary & Problem Formulation

In modern VLSI physical design (sub-14nm FinFET and below), **physical routing** is one of the most computationally demanding stages in the ASIC design cycle. Generating routing trees that satisfy complex Design Rule Constraints (DRC) often takes hours or days on high-performance multi-threaded compute nodes.

When a placed netlist causes routing congestion that exceeds layer capacities, the entire placement must be rejected, modified, and rerouted. This trial-and-error feedback loop accounts for substantial delays in chip tape-outs.

```
Conventional Flow:
[Synthesis] ──> [Placement] ──> [Global Routing (HOURS)] ──> [DRC Violations Found!] ──> [Redo Placement]

AI-Accelerated Flow:
[Synthesis] ──> [Placement] ──> [AI Predictor (33 ms)] ──> [Instant Hotspot Feedback] ──> [Optimized Placement]
```

This system predicts the 2D spatial routing congestion heatmap directly from placed netlist features in **~33 milliseconds**—over **1,000× faster** than commercial global routers—enabling real-time congestion mitigation inside physical placement engines.

---

## 2. Multi-Modal Feature Engineering & Extraction

Physical design information is mapped into a normalized **6-channel spatial tensor** of dimension $(6, 128, 128)$ representing physical GCells (Global Routing Cells):

### Channel 0: Pin Density Map ($C_{pin}$)
Represents the count of standard cell and macro pins falling inside each grid cell:
$$C_{pin}(r, c) = \frac{\sum_{i \in \text{pins}} \mathbb{I}(\text{pos}(i) \in \text{bin}(r, c))}{\max(C_{pin})}$$
High pin density directly correlates with high local wire demand.

### Channel 1: Standard Cell Area Density ($C_{cell}$)
Fractional area occupied by standard logic cells within each GCell:
$$C_{cell}(r, c) = \frac{\sum_{j \in \text{cells}} \text{OverlapArea}(\text{cell}_j, \text{bin}(r, c))}{\text{Area}(\text{bin}(r, c))}$$

### Channel 2 & 3: Directional RUDY Wire Demand ($C_{rudy\_h}, C_{rudy\_v}$)
**Rectilinear Uniform Wire Density (RUDY)** approximates the probability of routing track occupancy based on net bounding boxes:
$$\text{RUDY}_H(r, c) = \sum_{net \in \mathcal{N}} \frac{W_{net} \cdot \mathbb{I}((r, c) \in \text{bbox}(net))}{\text{Height}(\text{bbox}(net))}$$
$$\text{RUDY}_V(r, c) = \sum_{net \in \mathcal{N}} \frac{W_{net} \cdot \mathbb{I}((r, c) \in \text{bbox}(net))}{\text{Width}(\text{bbox}(net))}$$
Short-net RUDY serves as a primary indicator for horizontal wire demand, while long-net RUDY indicates vertical routing demand across cluster boundaries.

### Channel 4: Total Rectilinear Demand ($C_{rudy\_tot}$)
$$\text{Total RUDY}(r, c) = \text{RUDY}_H(r, c) + \text{RUDY}_V(r, c)$$

### Channel 5: Macro & Blockage Obstacle Mask ($C_{macro}$)
Binary mask indicating regions occupied by hard macros (such as SRAM arrays or analog IP):
$$C_{macro}(r, c) = \begin{cases} 1.0 & \text{if tile contains macro blockage} \\ 0.0 & \text{otherwise} \end{cases}$$
Macros block intermediate routing layers, forcing wires to detour around boundaries and causing severe edge congestion.

---

## 3. Deep Learning Architecture: Residual Attention U-Net

The architecture contains **8,203,966 trainable parameters**, organized into a symmetric encoder-decoder topology with skip attention gates:

```
Input (6, 128, 128)
  │
  ├──> ConvBlock (32)  ───────────────────────────────> AttentionGate ──> UpBlock (32) ──> Out (2, 128, 128)
  │        │ MaxPool(2)                                            ▲
  │        ▼                                                       │
  ├──> ConvBlock (64)  ────────────────────────> AttentionGate ──> UpBlock (64)
  │        │ MaxPool(2)                                     ▲
  │        ▼                                                │
  ├──> ConvBlock (128) ────────────────> AttentionGate ────> UpBlock (128)
  │        │ MaxPool(2)                             ▲
  │        ▼                                        │
  ├──> ConvBlock (256) ────────> AttentionGate ────> UpBlock (256)
  │        │ MaxPool(2)                      ▲
  │        ▼                                 │
  └──> Bottleneck (512) ─────────────────────┘
```

### Key Architectural Components:
1. **Residual Convolutional Blocks**:
   Each block consists of two $3 \times 3$ convolutions, Batch Normalization, and GELU activations, augmented with a $1 \times 1$ identity shortcut:
   $$\mathbf{y} = \text{GELU}(\text{BatchNorm}(\text{Conv}_{3\times3}(\text{GELU}(\text{BatchNorm}(\text{Conv}_{3\times3}(\mathbf{x}))))) + \mathcal{W}_s \mathbf{x})$$
2. **Attention Gates (AG)**:
   Over 90% of a semiconductor die has low routing congestion. Standard skip connections pass flat background features that can blur localized hotspots. The Attention Gate uses gating signal $\mathbf{g}$ from the coarser decoder level to modulate encoder features $\mathbf{x}$:
   $$\alpha = \sigma\left(\psi^T \left[\text{ReLU}\left(W_g^T \mathbf{g} + W_x^T \mathbf{x} + b_g\right)\right] + b_\psi\right)$$
   $$\hat{\mathbf{x}} = \mathbf{x} \cdot \alpha$$
3. **Composite Multi-Objective Loss**:
   Balances pixel-wise outlier robustness with structural coherence:
   $$\mathcal{L}_{total} = \mathcal{L}_{\text{Huber, Weighted}} + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}$$
   $$\mathcal{L}_{\text{Huber, Weighted}} = \frac{1}{N} \sum_{i=1}^N w_i \cdot \text{Huber}(y_i, \hat{y}_i), \quad w_i = \begin{cases} 3.0 & \text{if } y_i \ge 0.5 \\ 1.0 & \text{otherwise} \end{cases}$$

---

## 4. Benchmark Dataset & Verification (CircuitNet N14)

The pipeline uses the official **CircuitNet N14** physical design dataset hosted on Hugging Face (`CircuitNet/CircuitNet`):
* **Technology Node**: 14nm FinFET.
* **Architecture**: NVIDIA Deep Learning Accelerator (NVDLA-small).
* **Sample Count**: 96 real physical placement variations spanning different clock frequencies (200MHz, 500MHz), macro placement strategies, and pin densities.
* **Dataset Split**: 82 designs for training, 14 strictly held-out designs for testing.

---

## 5. Quantitative Test Evaluation

Evaluated across the 14 held-out test placements:

| Metric | Measured Value | Standard Deviation |
| :--- | :---: | :---: |
| **Mean Pearson Correlation (PCC)** | **0.7762** | $\pm 0.0200$ |
| **Peak Sample PCC** | **0.8038** | — |
| **Normalized Mean Squared Error (NMSE)** | **0.2617** | $\pm 0.0219$ |
| **Mean Absolute Error (MAE)** | **0.0554** | $\pm 0.0071$ |
| **Root Mean Squared Error (RMSE)** | **0.0907** | $\pm 0.0084$ |
| **Hotspot F1-Score @ Top-5%** | **0.5246** | $\pm 0.0360$ |
| **Hotspot F1-Score @ Top-10%** | **0.5889** | $\pm 0.0315$ |
| **Pixels within ±10% Capacity** | **80.8%** | $\pm 2.4\%$ |
| **Mean Inference Time** | **32.9 ms** | $\pm 1.8 \text{ ms}$ |

---

## 6. Web Interface System Design

* **Server Stack**: FastAPI asynchronous REST API running on an ASGI Uvicorn engine.
* **Rendering Engine**: Headless Matplotlib buffer serialization to Base64 PNG.
* **Frontend UI**: Single-page application built with HTML5, Tailwind CSS, and JetBrains Mono typography.
* **Endpoints**:
  * `GET /`: Serves the primary web dashboard.
  * `GET /api/presets`: Returns available CircuitNet design configurations.
  * `POST /api/predict`: Executes forward inference given a preset name or uploaded `.npz` file and returns base64-encoded heatmaps, error maps, and metrics.
