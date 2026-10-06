# Pi3 Custom Launcher & Workbench

An optimized, non-intrusive Gradio workbench and Windows runner for **[Pi3](https://github.com/yyfz/Pi3)** (by yyfz). 

Designed as a modular extension, this repository supercharges upstream Pi3 with automated gravity alignment, Blender-ready coordinate transforms, and dynamic memory recycling—without modifying the upstream code.

---
<img width="1389" height="761" alt="image" src="https://github.com/user-attachments/assets/84c252d8-d519-4109-a8ed-25658d719953" />
<img width="1423" height="751" alt="image" src="https://github.com/user-attachments/assets/2bc316cb-1763-44bf-859c-051e094e6035" />
<img width="912" height="681" alt="image" src="https://github.com/user-attachments/assets/9c8b3f74-ae89-479f-9cb7-520423c53fc1" />
<img width="1201" height="755" alt="image" src="https://github.com/user-attachments/assets/3e89f2c1-2915-435d-af50-658fbc8c9b5e" />

> **Preview Overview**: Figures 1 & 3 show the Gradio interactive workbench; Figures 2 & 4 show point clouds and camera setups imported directly into Blender (rendered via *Point Cloud Visualizer* add-on). Note that while GeoCalib auto-aligns the scene to world vertical direction, residual alignment errors are unavoidable depending on visual cues.

---

## 🌟 Key Features

* **GeoCalib Gravity Alignment**: Integrated with **[GeoCalib](https://github.com/cvg/GeoCalib)** to estimate scene gravity vectors across frame samples, automatically rotating point clouds and camera poses to standard vertical orientation (`[0, -1, 0]`).
* **RAM-First Cache & Auto Cleanup**: Implements a LRU `SessionManager` that prioritizes RAM for interactive previews and automatically purges old session files and disk caches, keeping temporary folder bloat strictly under control.
* **Low-VRAM Offloading**: Smart CPU/GPU memory swapping between GeoCalib and Pi3 prevents CUDA OOM on budget GPUs.
* **Blender Native Compatibility**: Automatically bakes a $+90^\circ$ X-axis coordinate flip into `.ply` point clouds and generates one-click Python scripts (`import_cameras_blender.py`) for camera pose imports.

---

## 📁 Repository Structure

Place all files directly into your local **Pi3 project root**:

```text
<Your_Pi3_Project_Root>/
├── venv/                       # Virtual environment (Default: "venv")
├── pi3/                        # Upstream Pi3 codebase
├── weights/
│   ├── geocalib/               # Auto-downloaded GeoCalib weights
│   ├── Pi3/                    # Base Pi3 weights
│   └── Pi3X/                   # Base Pi3X weights
├── _gradio_work/               # Auto-managed session cache (Auto-cleaned)
├── outputs/                    # Final exported PLY & camera pose files
├── run_pi3_local.py            # [Mod] Custom Launcher & Workbench UI
└── run_pi3_local.bat            # [Mod] Windows one-click runner

```

> ⚠️ **Note**: `run_pi3_local.bat` defaults to `.\venv`. Adjust the path if your environment name differs.

---

## 🚀 Setup & Installation

### Step 1: Upstream Pi3 Base Environment

Follow the official [Pi3 Repository](https://github.com/yyfz/Pi3) to complete the base installation:

```bat
git clone [https://github.com/yyfz/Pi3.git](https://github.com/yyfz/Pi3.git)
cd Pi3
python -m venv venv
call venv\Scripts\activate

pip install torch==2.5.1 torchvision==0.20.1 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -r requirements.txt

```

### Step 2: Mod Additions

With your `venv` active, install the GeoCalib package:

```bat
mkdir weights\geocalib
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

* **Primary Target**: **Blender** (Z-Up). Point clouds (`.ply`) and camera scripts (`import_cameras_blender.py`) are pre-transformed with $+90^\circ$ X-axis rotation.
* **Other Engines**: When importing into Unreal Engine, Maya, or TouchDesigner, manual axis remapping may be required.

---

## 🏃 Usage

Run the launcher using the batch script or terminal:

```bat
# Double click run_pi3_local.bat OR run manually:
call venv\Scripts\activate
python run_pi3_local.py

```

---

## 📄 License & Attribution

* **Launcher / Mod**: Distributed under the **[MIT License](https://opensource.org/licenses/MIT)**.
* **Upstream Code**:
* **[Pi3](https://github.com/yyfz/Pi3)** — BSD-3-Clause License.
* **[GeoCalib](https://github.com/cvg/GeoCalib)** — Apache-2.0 License.



