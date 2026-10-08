# Pi3 Custom Launcher & Workbench

An optimized, non-intrusive Gradio workbench and Windows runner for **[Pi3](https://github.com/yyfz/Pi3)** (by yyfz). 

Designed as a modular extension, this repository supercharges upstream Pi3—a **Feed-forward Structure-from-Motion (SfM)**—with robust iterative gravity alignment, Blender-ready coordinate transforms, and dynamic memory recycling, all without modifying upstream source code.

---
<img width="1389" height="761" alt="image" src="https://github.com/user-attachments/assets/84c252d8-d519-4109-a8ed-25658d719953" />
<img width="1423" height="751" alt="image" src="https://github.com/user-attachments/assets/2bc316cb-1763-44bf-859c-051e094e6035" />
<img width="912" height="681" alt="image" src="https://github.com/user-attachments/assets/9c8b3f74-ae89-479f-9cb7-520423c53fc1" />
<img width="1210" height="740" alt="image" src="https://github.com/user-attachments/assets/171c053b-d47a-42d6-8fe5-a6d1cb703b23" />

> **Preview Overview**: Figures 1 & 3 show the Gradio interactive workbench; Figures 2 & 4 show point clouds and camera setups imported directly into Blender (rendered via *Point Cloud Visualizer* add-on). Note that while GeoCalib auto-aligns the scene to world vertical direction, residual alignment errors are minimized via iterative medoid filtering.

---

## 🌟 Key Features
* **Feed-Forward SfM Engine**: Direct batch reconstruction of dense 3D point clouds, local point maps, and camera poses from arbitrary image sequences via Pi3X, bypassing time-consuming traditional SfM solving.
* **Robust Medoid Gravity Alignment**: Leverages **[GeoCalib](https://github.com/cvg/GeoCalib)** strictly for vertical gravity alignment. Uses a 3-pass iterative center-refinement algorithm (`robust_g_mean`) to suppress pitch outliers and align point clouds to world vertical (`[0, -1, 0]`).
* **Blender Native Camera Export**: Generates one-click Python import scripts (`import_cameras_blender.py`) with full intrinsic/extrinsic alignment. Features regression-based FOV solving directly from local point maps, with automatic focal clustering and rounding (e.g., `18.3 mm`) to eliminate 2D-3D projection misalignments.
* **Performance & Memory Optimization**: Built-in LRU Session Manager and smart CPU/GPU offloading between Pi3X and GeoCalib prevent CUDA OOM on consumer GPUs while automatically purging stale disk/RAM caches.
---

## 📁 Repository Structure

Place all files directly into your local **Pi3 project root**:

```text
<Your_Pi3_Project_Root>/
├── venv/                        # Virtual environment (Default: "venv")
├── pi3/                         # Upstream Pi3 codebase
├── weights/                     # Weights directory (auto-created by run_pi3_local.bat)
│   ├── geocalib/                # Auto-downloaded GeoCalib weights
│   ├── Pi3/                     # Base Pi3 weights
│   └── Pi3X/                    # Base Pi3X weights (place model.safetensors here)
├── _gradio_work/                # Auto-managed session cache (Auto-cleaned)
├── outputs/                     # Final exported PLY point clouds & camera pose files
├── run_pi3_local.py             # [Mod] Custom Launcher & Workbench UI
└── run_pi3_local.bat            # [Mod] Windows one-click runner

```

> ⚠️ **Note**: `run_pi3_local.bat` defaults to `.\venv`. Adjust the path if your environment name differs.

---

## 🚀 Setup & Installation

### Step 1: Upstream Pi3 Base Environment

Follow the official [Pi3 Repository](https://github.com/yyfz/Pi3) to complete the base setup:

```bat
git clone [https://github.com/yyfz/Pi3.git](https://github.com/yyfz/Pi3.git)
cd Pi3
python -m venv venv
call venv\Scripts\activate

pip install torch==2.5.1 torchvision==0.20.1 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -r requirements.txt

```

### Step 2: Mod Additions

With your `venv` active, install GeoCalib:

```bat
call venv\Scripts\activate
pip install git+[https://github.com/cvg/GeoCalib.git](https://github.com/cvg/GeoCalib.git)

```

---

## 🛠️ Required Patch (Windows Only)

To fix PyTorch SDPA backend errors on Windows, edit `pi3/models/layers/attention.py`:

Replace all 4 occurrences of:

```python
with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):

```

with:

```python
with nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):

```

---

## 📐 Coordinate Systems & Software Compatibility

* **Primary Target**: **Blender** (Z-Up). Point clouds (`.ply`) and camera scripts (`import_cameras_blender.py`) are pre-transformed with a $+90^\circ$ X-axis rotation.
* **Other Engines**: When importing into Unreal Engine, Maya, or TouchDesigner, manual axis remapping may be required.

---

## 🏃 Usage

Run the launcher using the batch script or terminal:

```bat
# Double-click run_pi3_local.bat OR run manually:
call venv\Scripts\activate
python run_pi3_local.py

```

---

## 📄 License & Attribution

* **Launcher / Mod**: Distributed under the **[MIT License](https://opensource.org/licenses/MIT)**.
* **Upstream Code**:
* **[Pi3](https://github.com/yyfz/Pi3)** — BSD-3-Clause License.
* **[GeoCalib](https://github.com/cvg/GeoCalib)** — Apache-2.0 License.



```

```
