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

* **Feed-forward SfM Engine**: Direct batch estimation of dense 3D point clouds, intrinsics, and camera poses from arbitrary, unconstrained image sequences.(implemented via pi3; I just provide the custom-built workbench)
* **Robust Medoid Gravity Alignment**: Integrated with **[GeoCalib](https://github.com/cvg/GeoCalib)** using a 3-pass iterative center-refinement algorithm (`robust_g_mean`). Effectively suppresses extreme pitch outliers to align point clouds precisely with world vertical gravity (`[0, -1, 0]`).
* **Self-Cleaning Session Manager**: LRU-based memory management prioritizing RAM for fast interactive previews while purging expired prediction caches and disk files to prevent storage bloat.
* **Low-VRAM Offloading**: Smart CPU/GPU memory swapping between GeoCalib and Pi3 inference models prevents CUDA OOM on consumer GPUs.
* **Blender Native Compatibility**: Automatically bakes a $+90^\circ$ X-axis coordinate flip into `.ply` point clouds and generates one-click Python scripts (`import_cameras_blender.py`) with real focal length remapping for Blender camera imports.

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
