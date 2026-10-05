<img width="1389" height="761" alt="image" src="https://github.com/user-attachments/assets/84c252d8-d519-4109-a8ed-25658d719953" />
<img width="1423" height="751" alt="image" src="https://github.com/user-attachments/assets/2bc316cb-1763-44bf-859c-051e094e6035" />


# Pi3 Custom Launcher & Workbench

An extensible Gradio-based launcher and local workbench for **[Pi3](https://github.com/yyfz/Pi3)** (by yyfz). 

This repository provides an optimized runner (`run_pi3_local.py`) and Windows entry point (`run_pi3_local.bat`). It is designed as a modular wrapper to integrate experimental features and pipeline enhancements onto Pi3 without modifying the upstream codebase directly.

---

## 🌟 Features

* **GeoCalib Gravity Alignment (Feature #1)**: Integrates **[GeoCalib](https://github.com/cvg/GeoCalib)** (by CVG) to estimate multi-view gravity vectors, automatically aligning reconstructed point clouds and camera poses to standard world vertical direction (`[0, -1, 0]`).
* **Low-VRAM Optimization**: Dynamically manages CPU/GPU weight offloading between GeoCalib and Pi3, preventing CUDA OOM on budget GPUs.
* **Blender-Oriented Coordinates**: Automatically bakes $+90^\circ$ X-axis coordinate transformation for `.ply` point clouds and camera pose exports for native compatibility with Blender.
* **Windows SDPA Compatibility Fix**: Includes a clean source-patch solution for PyTorch SDPA kernel errors on Windows.

---

## 📁 Repository Structure

Copy the files from this repository directly into your local **Pi3 root folder**:

```text
<Your_Pi3_Project_Root>/
├── venv/                       # Target virtual environment (Default name: "venv")
├── pi3/                        # Upstream Pi3 codebase
├── weights/
│   ├── geocalib/               # Auto-downloaded GeoCalib weights
│   │   ├── hub/geocalib/pinhole.tar
│   │   └── pinhole.tar
│   ├── Pi3/
│   │   └── model.safetensors   # Base Pi3 weights
│   └── Pi3X/
│       └── model.safetensors   # Base Pi3X weights
├── run_pi3_local.py            # [Mod] Custom Launcher & UI
└── run_pi3_local.bat           # [Mod] Windows one-click runner

```

> ⚠️ **Note**: `run_pi3_local.bat` assumes your local virtual environment is located at `.\venv`. If you use a custom venv path, edit the batch script accordingly.

---

## 🚀 Quick Setup

### Step 1: Upstream Pi3 Base Environment (Reference)

*Follow official [Pi3 Installation Instructions](https://github.com/yyfz/Pi3) to complete basic installation.*

```bat
# Clone base Pi3 repository and set up virtual environment
git clone [https://github.com/yyfz/Pi3.git](https://github.com/yyfz/Pi3.git)
cd Pi3
python -m venv venv
call venv\Scripts\activate

# Install base requirements
pip install torch==2.5.1 torchvision==0.20.1 --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
pip install -r requirements.txt

```

---

### Step 2: Mod-Specific Setup (This Repository)

With your Pi3 `venv` activated, apply the mod-specific additions:

```bat
# 1. Install GeoCalib inference package
mkdir weights\geocalib
pip install git+[https://github.com/cvg/GeoCalib.git](https://github.com/cvg/GeoCalib.git)

# 2. Pin Gradio version (required for launch compatibility)
pip install --upgrade "gradio<6.0"

```

---

## 🛠️ Required Patch (Windows Only)

Due to PyTorch SDPA limitations on Windows, modify `pi3/models/layers/attention.py` in the upstream codebase:

Replace all 4 occurrences of:

```python
with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):

```

with:

```python
with nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):

```

---

## 📐 Coordinate System & Software Compatibility

* **Primary Target**: **Blender**. Exported `.ply` files and camera scripts (`import_cameras_blender.py`) include built-in $+90^\circ$ X-axis rotation alignment.
* **Other Engines**: If importing exports into Unreal Engine, Maya, or TouchDesigner, manual axis remap/rotation may be required.

---

## 🏃 Usage

Launch the UI using the provided batch file or terminal:

```bat
# Double click run_pi3_local.bat OR run manually:
call venv\Scripts\activate
python run_pi3_local.py

```

---

## 📄 License & Attribution

* **This Launcher / Mod**: Released under the permissve **[MIT License](https://www.google.com/search?q=LICENSE)**. Feel free to use, modify, and distribute.
* **Upstream Dependencies**:
* **[Pi3](https://github.com/yyfz/Pi3)** is released under the **BSD-3-Clause License**.
* **[GeoCalib](https://github.com/cvg/GeoCalib)** is released under the **Apache-2.0 License**.

