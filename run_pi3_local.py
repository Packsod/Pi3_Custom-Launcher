from __future__ import annotations

import gc, json, os, shutil, sys, time, traceback, uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from threading import Lock

# 配置 CUDA 内存分配器环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import matplotlib
import numpy as np
import torch
import trimesh
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation

from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor

try:
    from geocalib import GeoCalib
    GEOCALIB_AVAILABLE = True
except ImportError:
    GEOCALIB_AVAILABLE = False
    print("⚠️ GeoCalib module not found.")

_GEOCALIB_MODELS = {}
_GEOCALIB_LOCK = Lock()

# =============================================================================
# Global Configuration
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
WORLD_DOWN = np.array([0.0, -1.0, 0.0], dtype=np.float32)

@dataclass(frozen=True)
class AppConfig:
    pi3_checkpoint: Path = REPO_ROOT / "weights" / "Pi3X" / "model.safetensors"
    geocalib_models_dir: Path = REPO_ROOT / "weights" / "geocalib"
    output_root: Path = REPO_ROOT / "outputs"
    work_root: Path = REPO_ROOT / "_gradio_work"
    
    max_sessions: int = 3
    max_images_ram_bytes: int = 8_000_000_000
    max_predictions_ram_bytes: int = 4_000_000_000
    keep_image_disk_cache: bool = False
    keep_prediction_disk_cache: bool = False
    
    default_confidence_percent: float = 20.0
    save_per_view_geometry: bool = False
    save_camera_json: bool = True
    
    blender_placeholder_lens_mm: float = 50.0
    blender_sensor_width_mm: float = 36.0
    preview_camera_width_ratio: float = 0.025
    preview_camera_height_ratio: float = 0.050
    preview_srgb_to_linear: bool = True
    preview_color_gain: float = 1.0

CONFIG = AppConfig()
CONFIG.geocalib_models_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(CONFIG.geocalib_models_dir)

SERVER_NAME = os.environ.get("PI3_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("PI3_SERVER_PORT", "7860"))
SHARE = os.environ.get("PI3_SHARE", "0").strip().lower() in {"1", "true", "yes"}
QUEUE_MAX_SIZE = 2


# =============================================================================
# Session Manager
# =============================================================================
class SessionManager:
    def __init__(self, work_root: Path):
        self.work_root = work_root
        self.sessions: OrderedDict[str, dict] = OrderedDict()

    @staticmethod
    def _tensor_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.element_size() * value.nelement()
        if isinstance(value, np.ndarray):
            return value.itemsize * value.size
        return 0

    @classmethod
    def _prediction_bytes(cls, predictions: dict | None) -> int:
        if not isinstance(predictions, dict):
            return 0
        return sum(cls._tensor_bytes(v) for v in predictions.values() if v is not None)

    def create_session(self, session_dir: Path) -> str:
        self._evict_if_needed()
        session_id = uuid.uuid4().hex
        final_dir = self.work_root / session_id
        
        if session_dir.exists():
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            session_dir.rename(final_dir)
            
        images_dir = final_dir / "images"
        rel_image_paths = sorted([str(p) for p in images_dir.glob("*") if p.is_file()])

        self.sessions[session_id] = {
            "session_id": session_id,
            "imgs_cpu": None,          
            "imgs_dir": images_dir,
            "predictions": None,       
            "predictions_path": None,  
            "gallery_paths": rel_image_paths,
            "last_export_dir": None,
        }
        self.sessions.move_to_end(session_id)
        return session_id

    def get_session(self, session_id: str) -> dict:
        if not session_id or session_id not in self.sessions:
            raise KeyError("Session not found or expired.")
        self.sessions.move_to_end(session_id)
        return self.sessions[session_id]

    def load_images_to_ram(self, session_id: str) -> torch.Tensor:
        sess = self.get_session(session_id)
        if sess["imgs_cpu"] is not None:
            return sess["imgs_cpu"]

        imgs = load_images_as_tensor(str(sess["imgs_dir"]), interval=1)
        sess["imgs_cpu"] = imgs
        self._evict_if_needed(protected_id=session_id)
        return imgs

    def set_predictions(self, session_id: str, predictions: dict, predictions_path: Path | None = None):
        sess = self.get_session(session_id)
        sess["predictions"] = predictions
        sess["predictions_path"] = predictions_path
        self._evict_if_needed(protected_id=session_id)

    def get_predictions(self, session_id: str) -> dict | None:
        sess = self.get_session(session_id)
        if sess["predictions"] is not None:
            return sess["predictions"]
        
        path = sess["predictions_path"]
        if path is not None and path.exists():
            with np.load(path) as loaded:
                predictions = {
                    "images": np.asarray(loaded["images"]),
                    "points": np.asarray(loaded["points"]),
                    "conf": np.asarray(loaded["conf"]),
                    "camera_poses": np.asarray(loaded["camera_poses"]),
                    "focals": np.asarray(loaded["focals"]) if "focals" in loaded else None,
                    "principal_points": np.asarray(loaded["principal_points"]) if "principal_points" in loaded else None,
                }
                sess["predictions"] = predictions
                self._evict_if_needed(protected_id=session_id)
                return predictions
        return None

    def destroy_session(self, session_id: str):
        sess = self.sessions.pop(session_id, None)
        if sess is None:
            return

        prediction_path = sess.get("predictions_path")
        session_dir = sess["imgs_dir"].parent if sess.get("imgs_dir") else None

        sess["imgs_cpu"] = None
        sess["predictions"] = None

        if prediction_path and prediction_path.exists() and not CONFIG.keep_prediction_disk_cache:
            prediction_path.unlink(missing_ok=True)

        if session_dir and session_dir.exists() and not CONFIG.keep_image_disk_cache:
            shutil.rmtree(session_dir, ignore_errors=True)

    def _oldest_session_with(self, key: str, protected_id: str | None = None) -> str | None:
        for sid, sess in self.sessions.items():
            if sid != protected_id and sess.get(key) is not None:
                return sid
        return None

    def _evict_if_needed(self, protected_id: str | None = None):
        while len(self.sessions) > CONFIG.max_sessions:
            oldest_id = next(s for s in self.sessions if s != protected_id)
            self.destroy_session(oldest_id)

        while sum(self._tensor_bytes(s.get("imgs_cpu")) for s in self.sessions.values()) > CONFIG.max_images_ram_bytes:
            evict_id = self._oldest_session_with("imgs_cpu", protected_id)
            if not evict_id: break
            self.sessions[evict_id]["imgs_cpu"] = None

        while sum(self._prediction_bytes(s.get("predictions")) for s in self.sessions.values()) > CONFIG.max_predictions_ram_bytes:
            evict_id = self._oldest_session_with("predictions", protected_id)
            if not evict_id: break
            self.sessions[evict_id]["predictions"] = None


SESSION_MANAGER = SessionManager(CONFIG.work_root)


# =============================================================================
# Basic Utilities & Common Helpers
# =============================================================================
def cleanup_memory(force_cuda: bool = False) -> None:
    gc.collect()
    if force_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_directories() -> None:
    CONFIG.output_root.mkdir(parents=True, exist_ok=True)
    CONFIG.work_root.mkdir(parents=True, exist_ok=True)

def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Pi3 inference requires CUDA.")
    return torch.device("cuda")

def get_autocast_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

def safe_filename(value: str, fallback: str = "result") -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value).strip().replace(" ", "_"))
    return cleaned.strip("._") or fallback

def create_export_dir() -> Path:
    export_dir = CONFIG.output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_dir.mkdir(parents=True, exist_ok=False)
    return export_dir

def batch_chw_to_hwc(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images

def parse_frame_index(filter_str: str | None) -> int | None:
    if not filter_str or str(filter_str).lower() == "all":
        return None
    try:
        return int(str(filter_str).split(":")[0])
    except (ValueError, IndexError):
        return None

def resolve_input_path(file_data) -> Path:
    value = file_data.get("name", file_data) if isinstance(file_data, dict) else file_data
    return Path(value)

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def rgb_float_to_uint8(rgb: np.ndarray, *, linearize: bool = False, gain: float = 1.0) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    if linearize:
        rgb = srgb_to_linear(rgb)
    return np.clip(rgb * gain * 255.0, 0, 255).astype(np.uint8)

def valid_point_mask(points_flat: np.ndarray, conf_flat: np.ndarray, threshold_percent: float) -> np.ndarray:
    threshold = max(float(threshold_percent) / 100.0, 0.0)
    return (
        np.isfinite(points_flat).all(axis=1)
        & np.isfinite(conf_flat)
        & (conf_flat >= threshold)
        & (conf_flat > 1e-5)
    )

def extract_valid_points(predictions: dict, conf_thres: float, frame_filter: str = "All") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = parse_frame_index(frame_filter)
    
    pts = np.asarray(predictions["points"])
    conf = np.asarray(predictions["conf"])
    imgs = batch_chw_to_hwc(predictions["images"])
    poses = np.asarray(predictions["camera_poses"])

    if idx is not None and 0 <= idx < len(pts):
        pts, conf, imgs, poses = pts[idx:idx+1], conf[idx:idx+1], imgs[idx:idx+1], poses[idx:idx+1]

    pts_flat = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    imgs_flat = imgs.reshape(-1, 3)

    mask = valid_point_mask(pts_flat, conf_flat, conf_thres)
    return pts_flat[mask], imgs_flat[mask], poses


# =============================================================================
# Vector Geometry & Alignment
# =============================================================================
def transform_world_points(points: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ world_transform[:3, :3].T + world_transform[:3, 3]

def transform_camera_poses(camera_to_world: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    return world_transform[None, :, :] @ np.asarray(camera_to_world, dtype=np.float32)

def align_hemisphere(vecs: np.ndarray, ref_vec: np.ndarray) -> np.ndarray:
    ref = ref_vec / max(np.linalg.norm(ref_vec), 1e-12)
    aligned = np.asarray(vecs, dtype=np.float64).copy()
    aligned[(aligned @ ref) < 0.0] *= -1.0
    return aligned

def robust_g_mean(vectors: list[np.ndarray], max_angle_deg: float = 20.0, max_iters: int = 3):
    if not vectors:
        return None, None, -1

    g_vecs = np.asarray([v / max(np.linalg.norm(v), 1e-12) for v in vectors], dtype=np.float64)
    if len(g_vecs) == 1:
        return g_vecs[0].astype(np.float32), np.ones(1, dtype=bool), 0

    medoid_idx = int(np.argmax(np.sum(g_vecs @ g_vecs.T, axis=1)))
    g_medoid = g_vecs[medoid_idx]
    g_aligned = align_hemisphere(g_vecs, g_medoid)

    cos_thresh = np.cos(np.deg2rad(max_angle_deg))
    g_center = g_medoid.copy()

    for _ in range(max_iters):
        inliers = (g_aligned @ g_center) >= cos_thresh
        if not np.any(inliers): inliers[medoid_idx] = True

        refined = g_aligned[inliers].mean(axis=0)
        norm = np.linalg.norm(refined)
        if norm < 1e-12: break
        refined /= norm

        if np.dot(g_center, refined) > 1.0 - 1e-8:
            g_center = refined
            break
        g_center = refined

    inliers = (g_aligned @ g_center) >= cos_thresh
    if not np.any(inliers):
        inliers[np.argmax(g_aligned @ g_center)] = True

    g_mean = g_aligned[inliers].mean(axis=0)
    g_mean /= max(np.linalg.norm(g_mean), 1e-12)
    return g_mean.astype(np.float32), inliers, medoid_idx

def rotation_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    R_mat, _ = Rotation.align_vectors(target[None, :], source[None, :])
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat.as_matrix().astype(np.float32)
    return T

def apply_gravity_alignment(points: np.ndarray, camera_poses: np.ndarray, gravity_vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    g_mean, _, _ = robust_g_mean(gravity_vectors, max_angle_deg=20.0, max_iters=3)
    if g_mean is None:
        return points, camera_poses

    T_align = rotation_align_vectors(g_mean, WORLD_DOWN)
    print("📐 [GeoCalib Alignment] 旋转对齐完成！")
    return transform_world_points(points, T_align), transform_camera_poses(camera_poses, T_align)


# =============================================================================
# GeoCalib Calibration & FOV Processing
# =============================================================================
def get_geocalib_model(weights="pinhole", device="cuda"):
    if not GEOCALIB_AVAILABLE:
        return None
    with _GEOCALIB_LOCK:
        if weights not in _GEOCALIB_MODELS:
            print(f"[GeoCalib] Loading model '{weights}' from: {CONFIG.geocalib_models_dir}")
            _GEOCALIB_MODELS[weights] = GeoCalib(weights=weights).eval().to("cpu")
        return _GEOCALIB_MODELS[weights].to(device).eval()

def unload_geocalib_from_cuda(delete_model: bool = False) -> None:
    global _GEOCALIB_MODELS
    with _GEOCALIB_LOCK:
        for weights, geo_model in list(_GEOCALIB_MODELS.items()):
            try: geo_model.to("cpu")
            except Exception: pass
            if delete_model: del _GEOCALIB_MODELS[weights]
        if delete_model: _GEOCALIB_MODELS.clear()
    cleanup_memory(force_cuda=True)

def make_principal_points(num_frames: int, image_width: int, image_height: int) -> np.ndarray:
    return np.tile([image_width / 2.0, image_height / 2.0], (num_frames, 1)).astype(np.float32)

def greedy_group_focals(focals_dict: dict[int, list[float]], num_frames: int, rel_tol: float = 0.08) -> np.ndarray:
    sorted_indices = sorted(focals_dict.keys())
    fx_vals = np.array([focals_dict[i][0] for i in sorted_indices], dtype=np.float32)
    fy_vals = np.array([focals_dict[i][1] for i in sorted_indices], dtype=np.float32)

    clusters: list[list[int]] = []
    for idx, (fx, fy) in enumerate(zip(fx_vals, fy_vals)):
        f_mean = (fx + fy) / 2.0
        assigned = False
        for cluster in clusters:
            cluster_f_mean = np.mean([(fx_vals[c] + fy_vals[c]) / 2.0 for c in cluster])
            if cluster_f_mean > 1e-6 and abs(f_mean - cluster_f_mean) / cluster_f_mean <= rel_tol:
                cluster.append(idx)
                assigned = True
                break
        if not assigned: clusters.append([idx])

    sample_count = len(sorted_indices)
    valid_clusters = clusters if sample_count < 4 else [c for c in clusters if len(c) >= max(2, int(np.ceil(sample_count * 0.15)))] or clusters

    smooth_fx, smooth_fy = fx_vals.copy(), fy_vals.copy()
    for cluster in valid_clusters:
        median_fx, median_fy = np.median(fx_vals[cluster]), np.median(fy_vals[cluster])
        for c_idx in cluster:
            smooth_fx[c_idx], smooth_fy[c_idx] = median_fx, median_fy

    cluster_means = [np.mean([(fx_vals[c] + fy_vals[c]) / 2.0 for c in cluster]) for cluster in valid_clusters]
    main_indices = {idx for cluster in valid_clusters for idx in cluster}

    for idx in range(len(sorted_indices)):
        if idx not in main_indices:
            f_mean = (fx_vals[idx] + fy_vals[idx]) / 2.0
            closest_cluster = valid_clusters[int(np.argmin([abs(f_mean - m) for m in cluster_means]))]
            smooth_fx[idx], smooth_fy[idx] = np.median(fx_vals[closest_cluster]), np.median(fy_vals[closest_cluster])

    focals_arr = np.tile([np.median(smooth_fx), np.median(smooth_fy)], (num_frames, 1)).astype(np.float32)
    for i_seq, frame_idx in enumerate(sorted_indices):
        focals_arr[frame_idx] = [smooth_fx[i_seq], smooth_fy[i_seq]]

    return focals_arr

def estimate_geocalib(imgs: torch.Tensor, camera_poses: np.ndarray, device: torch.device) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray]]:
    if not GEOCALIB_AVAILABLE:
        return None, None, []

    try:
        num_frames = len(imgs)
        sample_indices = np.unique(np.linspace(0, num_frames - 1, min(9, num_frames), dtype=int))
        geocalib_frames_cpu = {int(i): imgs[int(i)].detach().cpu() for i in sample_indices}

        geo_model = get_geocalib_model(weights="pinhole", device=device)
        g_worlds, focals_dict = [], {}

        print(f"[GeoCalib] Estimating gravity & intrinsics from {len(geocalib_frames_cpu)} frame(s)...")

        for idx in sample_indices:
            frame_tensor = geocalib_frames_cpu[int(idx)].to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=get_autocast_dtype()):
                results = geo_model.calibrate(frame_tensor, camera_model="pinhole")

            K_obj = results.get("camera", None)
            if K_obj is not None:
                K_mat = getattr(K_obj, "K", K_obj)
                K_np = K_mat.detach().float().cpu().numpy().squeeze() if isinstance(K_mat, torch.Tensor) else np.asarray(K_mat, dtype=np.float32).squeeze()
                focals_dict[idx] = [K_np[0, 0], K_np[1, 1]]

            gravity_obj = results.get("gravity", None)
            if gravity_obj is not None:
                g_tensor = getattr(gravity_obj, "vec3d", getattr(gravity_obj, "vec", gravity_obj))
                if hasattr(gravity_obj, "R"):
                    R_g = gravity_obj.R
                    g_tensor = (R_g.detach().cpu().numpy() if isinstance(R_g, torch.Tensor) else R_g)[..., :, 1]

                g_cam = g_tensor.detach().float().cpu().numpy().reshape(-1, 3)[0] if isinstance(g_tensor, torch.Tensor) else np.asarray(g_tensor, dtype=np.float32).reshape(-1, 3)[0]
                norm_cam = np.linalg.norm(g_cam)
                if np.isfinite(norm_cam) and norm_cam >= 1e-8:
                    g_cam /= norm_cam
                    c2w = np.asarray(camera_poses[idx], dtype=np.float32)
                    g_world = c2w[:3, :3] @ g_cam
                    g_world /= max(np.linalg.norm(g_world), 1e-12)
                    g_worlds.append(g_world)

            del frame_tensor, results

        focals_arr, pp_arr = None, None
        if focals_dict:
            focals_arr = greedy_group_focals(focals_dict, num_frames, rel_tol=0.08)
            img_h, img_w = imgs.shape[-2:]
            pp_arr = make_principal_points(num_frames, img_w, img_h)

        return focals_arr, pp_arr, g_worlds
    finally:
        unload_geocalib_from_cuda(delete_model=False)


# =============================================================================
# Preview GLB & 3D Export
# =============================================================================
def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple[int, int, int], scene_scale: float) -> None:
    scene_scale = max(float(scene_scale), 1e-5)
    cam_w = scene_scale * CONFIG.preview_camera_width_ratio
    cam_h = scene_scale * CONFIG.preview_camera_height_ratio

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot[2, 3] = -cam_h

    opengl_transform = np.diag([1.0, -1.0, -1.0, 1.0])
    complete_transform = transform @ opengl_transform @ rot

    cone = trimesh.creation.cone(cam_w, cam_h, sections=4)
    slight_rot = np.eye(4)
    slight_rot[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    t_vertices = cone.vertices @ slight_rot[:3, :3].T
    v_combined = np.concatenate([cone.vertices, 0.95 * cone.vertices, t_vertices])
    
    v_homo = np.pad(v_combined, ((0, 0), (0, 1)), constant_values=1.0)
    v_transformed = (v_homo @ complete_transform.T)[:, :3]

    num_v = len(cone.vertices)
    faces_list = []
    for v1, v2, v3 in cone.faces:
        if 0 in (v1, v2, v3): continue
        faces_list.extend([
            (v1, v2, v2 + num_v), (v1, v1 + num_v, v3), (v3 + num_v, v2, v3),
            (v1, v2, v2 + 2*num_v), (v1, v1 + 2*num_v, v3), (v3 + 2*num_v, v2, v3)
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]

    camera_mesh = trimesh.Trimesh(vertices=v_transformed, faces=np.array(faces_list))
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)

def predictions_to_glb(predictions: dict, conf_thres: float = CONFIG.default_confidence_percent, filter_by_frames: str = "All", show_cam: bool = True) -> trimesh.Scene:
    vertices_3d, source_rgb, poses = extract_valid_points(predictions, conf_thres, filter_by_frames)
    viewport_display_colors = rgb_float_to_uint8(
        source_rgb, 
        linearize=CONFIG.preview_srgb_to_linear, 
        gain=CONFIG.preview_color_gain
    )

    if vertices_3d.size == 0:
        vertices_3d, viewport_display_colors, scene_scale = np.array([[1.0, 0.0, 0.0]], dtype=np.float32), np.array([[255, 255, 255]], dtype=np.uint8), 1.0
    else:
        p5, p95 = np.percentile(vertices_3d, [5, 95], axis=0)
        scene_scale = max(float(np.linalg.norm(p95 - p5)), 1e-5)

    scene_3d = trimesh.Scene(trimesh.PointCloud(vertices=vertices_3d, colors=viewport_display_colors))

    if show_cam:
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        n_cams = len(poses)
        for i, pose in enumerate(poses):
            color = tuple(int(255 * x) for x in cmap(i / max(n_cams, 1))[:3])
            integrate_camera_into_scene(scene_3d, pose, color, scene_scale)

    return scene_3d


# =============================================================================
# Formal Exports
# =============================================================================
def write_binary_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    vertices, colors = np.asarray(vertices, dtype=np.float32), np.asarray(colors, dtype=np.uint8)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    
    packed = np.empty(len(vertices), dtype=dtype)
    packed["x"], packed["y"], packed["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    packed["red"], packed["green"], packed["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(packed)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        packed.tofile(f)

def export_pointcloud_ply(predictions: dict, export_dir: Path, conf_thres: float) -> tuple[Path, int]:
    vertices, source_rgb, _ = extract_valid_points(predictions, conf_thres, frame_filter="All")
    if len(vertices) == 0:
        raise RuntimeError("No point remains after confidence filtering. Lower Confidence Threshold.")

    raw_standard_colors = rgb_float_to_uint8(source_rgb, linearize=False)
    
    # 坐标系对齐到 Blender/通用点云 (+Y Up -> +Z Up)
    R_cv2blender = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    vertices = vertices @ R_cv2blender.T

    output_path = export_dir / "pointcloud.ply"
    write_binary_ply(output_path, vertices, raw_standard_colors)
    return output_path, len(vertices)

def write_blender_camera_importer(pose_npz_path: Path, output_path: Path) -> None:
    script = f'''\
import bpy, numpy as np
from mathutils import Matrix

NPZ_PATH = r"{pose_npz_path}"
COLLECTION_NAME, CAMERA_PREFIX = "Pi3_Cameras", "Pi3_Camera_"
DEFAULT_SENSOR_WIDTH_MM = {CONFIG.blender_sensor_width_mm}

data = np.load(NPZ_PATH)
poses_cv_c2w = data["camera_to_world_opencv"]
resolutions = data.get("resolutions", np.tile([int(data.get("image_width", 1920)), int(data.get("image_height", 1080))], (len(poses_cv_c2w), 1)))

has_intrinsics = "focals" in data and "principal_points" in data
focals = data["focals"] if has_intrinsics else None
principal_points = data["principal_points"] if has_intrinsics else None

if old_col := bpy.data.collections.get(COLLECTION_NAME):
    for obj in list(old_col.objects): bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old_col)

collection = bpy.data.collections.new(COLLECTION_NAME)
bpy.context.scene.collection.children.link(collection)

opencv_to_blender = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))

for i, pose_np in enumerate(poses_cv_c2w):
    res_x, res_y = int(resolutions[i][0]), int(resolutions[i][1])
    cam = bpy.data.cameras.new(f"{{CAMERA_PREFIX}}{{i:04d}}")
    cam.type = "PERSP"
    cam.sensor_width = DEFAULT_SENSOR_WIDTH_MM
    cam.sensor_fit = 'HORIZONTAL'

    if has_intrinsics:
        fx, fy = focals[i]
        cx, cy = principal_points[i]
        cam.lens = float(fx) * (cam.sensor_width / res_x)
        cam.shift_x = (res_x / 2.0 - float(cx)) / res_x
        cam.shift_y = (float(cy) - res_y / 2.0) / res_x
    else:
        cam.lens = {CONFIG.blender_placeholder_lens_mm}

    try:
        cam.per_camera_resolution.resolution_x, cam.per_camera_resolution.resolution_y = res_x, res_y
    except AttributeError:
        cam["resolution_x"], cam["resolution_y"] = res_x, res_y

    obj = bpy.data.objects.new(cam.name, cam)
    collection.objects.link(obj)
    obj.matrix_world = Matrix(pose_np.tolist()) @ opencv_to_blender
    obj["pi3_index"], obj["pi3_pose_convention"] = i, "OpenCV camera_to_world"

if len(poses_cv_c2w) > 0:
    bpy.context.scene.camera = collection.objects[0]
print(f"Imported {{len(poses_cv_c2w)}} real Blender cameras with GeoCalib FOV.")
'''
    output_path.write_text(script, encoding="utf-8")

def export_camera_poses(predictions: dict, export_dir: Path, original_image_sizes: list[tuple[int, int]] | None = None) -> tuple[Path, Path | None, Path]:
    poses = np.asarray(predictions["camera_poses"], dtype=np.float32)
    num_cameras = len(poses)
    if original_image_sizes and len(original_image_sizes) == num_cameras:
        resolutions = np.array(original_image_sizes, dtype=np.int32)
        img_w, img_h = resolutions[0]
    else:
        imgs = np.asarray(predictions["images"])
        img_h, img_w = (imgs.shape[2:4] if imgs.ndim == 4 and imgs.shape[1] == 3 else imgs.shape[1:3])
        resolutions = np.broadcast_to([img_w, img_h], (num_cameras, 2)).astype(np.int32)

    T_x90 = np.eye(4, dtype=np.float32)
    T_x90[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    poses = transform_camera_poses(poses, T_x90)

    save_data = {
        "camera_to_world_opencv": poses,
        "resolutions": resolutions,
        "image_width": np.int32(img_w),
        "image_height": np.int32(img_h),
        "pose_convention": np.array("OpenCV camera_to_world"),
        "export_alignment_applied": np.bool_(True),
    }

    if predictions.get("focals") is not None:
        save_data["focals"] = np.asarray(predictions["focals"], dtype=np.float32)
        save_data["principal_points"] = np.asarray(predictions["principal_points"], dtype=np.float32)

    npz_path = export_dir / "camera_poses.npz"
    np.savez_compressed(npz_path, **save_data)

    json_path = export_dir / "camera_poses.json" if CONFIG.save_camera_json else None
    if json_path:
        data = {
            "source": "Pi3X",
            "image_width": int(img_w),
            "image_height": int(img_h),
            "pose_convention": "OpenCV camera-to-world; +X right, +Y down, +Z forward",
            "export_alignment_applied": True,
            "cameras": [
                {
                    "index": i,
                    "resolution_x": int(resolutions[i][0]),
                    "resolution_y": int(resolutions[i][1]),
                    "camera_to_world_opencv": pose.tolist(),
                } for i, pose in enumerate(poses)
            ],
        }
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    blender_importer_path = export_dir / "import_cameras_blender.py"
    write_blender_camera_importer(npz_path, blender_importer_path)

    return npz_path, json_path, blender_importer_path

def export_reconstruction(
    session_id: str,
    predictions: dict,
    conf_thres: float,
    frame_filter: str,
    show_cam: bool,
) -> tuple[Path, Path, int]:
    sess = SESSION_MANAGER.get_session(session_id)
    export_dir = create_export_dir()
    sess["last_export_dir"] = export_dir

    _, point_count = export_pointcloud_ply(predictions, export_dir, conf_thres)
    export_camera_poses(predictions, export_dir)

    glb_path = build_glb_path(session_id, conf_thres, frame_filter, show_cam)
    scene = predictions_to_glb(predictions, conf_thres, frame_filter, show_cam)
    scene.export(file_obj=str(glb_path))

    return glb_path, export_dir, point_count


# =============================================================================
# Pipeline Inference Execution
# =============================================================================
def run_pi3_inference(model: Pi3X, imgs: torch.Tensor) -> dict:
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=get_autocast_dtype()):
        predictions = model(imgs[None], with_prior=False)

    predictions["conf"] = torch.sigmoid(predictions["conf"])
    pts_gpu = predictions["points"] if "points" in predictions else predictions["local_points"]

    return {
        "images": imgs[None].permute(0, 1, 3, 4, 2)[0].detach().float().cpu().numpy(),
        "points": pts_gpu[0].detach().float().cpu().numpy(),
        "conf": predictions["conf"][0].detach().float().cpu().numpy(),
        "camera_poses": predictions["camera_poses"][0].detach().float().cpu().numpy(),
        "focals": None,
        "principal_points": None,
    }

def run_model(session_id: str, model: Pi3X) -> dict:
    device = require_cuda()
    imgs = SESSION_MANAGER.load_images_to_ram(session_id).to(device)

    try:
        output = run_pi3_inference(model, imgs)
        try:
            focals, principal_points, g_worlds = estimate_geocalib(imgs, output["camera_poses"], device)
            output["focals"] = focals
            output["principal_points"] = principal_points

            if g_worlds:
                output["points"], output["camera_poses"] = apply_gravity_alignment(output["points"], output["camera_poses"], g_worlds)
        except Exception as e:
            print(f"⚠️ GeoCalib calibration failed: {e}")
            traceback.print_exc()

        sess = SESSION_MANAGER.get_session(session_id)
        pred_path = sess["imgs_dir"].parent / "predictions_preview.npz"

        save_dict = {k: output[k] for k in ("images", "points", "conf", "camera_poses")}
        if output["focals"] is not None:
            save_dict["focals"] = output["focals"]
            save_dict["principal_points"] = output["principal_points"]

        np.savez_compressed(pred_path, **save_dict)
        SESSION_MANAGER.set_predictions(session_id, output, pred_path)
        return output
    finally:
        del imgs
        cleanup_memory(force_cuda=True)


# =============================================================================
# Upload Management
# =============================================================================
def prepare_image_files(input_images, output_dir: Path, interval: int) -> list[str]:
    selected = input_images[::interval] if interval > 0 else input_images
    image_paths = []
    for index, file_data in enumerate(selected):
        src = resolve_input_path(file_data)
        if not src.is_file(): continue
        
        dst = output_dir / f"{index:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        image_paths.append(str(dst))
    return image_paths

def extract_video_frames(input_video, output_dir: Path, interval: int) -> list[str]:
    v_path = resolve_input_path(input_video)
    vs = cv2.VideoCapture(str(v_path))
    if not vs.isOpened():
        raise RuntimeError(f"Cannot open video: {v_path}")

    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = interval if interval > 0 else max(1, int(fps * 1))
    image_paths, frame_index = [], 0

    try:
        while True:
            gotit, frame = vs.read()
            if not gotit: break
            if frame_index % frame_interval == 0:
                img_path = output_dir / f"{len(image_paths):06d}.png"
                if not cv2.imwrite(str(img_path), frame):
                    raise RuntimeError(f"Failed to write frame: {img_path}")
                image_paths.append(str(img_path))
            frame_index += 1
    finally:
        vs.release()
    return image_paths

def handle_uploads(input_video, input_images, interval=-1):
    cleanup_memory()
    ensure_directories()

    temp_dir = CONFIG.work_root / f"tmp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    temp_images_dir = temp_dir / "images"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_images_dir.mkdir(parents=True, exist_ok=True)

    try: interval_int = int(interval)
    except (ValueError, TypeError): interval_int = -1

    image_paths = []
    if input_images:
        image_paths.extend(prepare_image_files(input_images, temp_images_dir, interval_int))
    if input_video:
        image_paths.extend(extract_video_frames(input_video, temp_images_dir, interval_int))

    if not image_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No valid images or video frames were prepared.")

    session_id = SESSION_MANAGER.create_session(temp_dir)
    sess = SESSION_MANAGER.get_session(session_id)
    return session_id, sess["gallery_paths"]

def update_gallery_on_upload(input_video, input_images, interval, old_session_id):
    if old_session_id and old_session_id != "None":
        SESSION_MANAGER.destroy_session(old_session_id)

    if not input_video and not input_images:
        return None, "None", None, "", gr.Dropdown(choices=["All"], value="All")

    try:
        session_id, gallery_paths = handle_uploads(input_video, input_images, interval=interval)
        return None, session_id, gallery_paths, f"Prepared {len(gallery_paths)} images. Click Reconstruct.", gr.Dropdown(choices=["All"], value="All")
    except Exception as error:
        return None, "None", None, f"Error: `{error}`", gr.Dropdown(choices=["All"], value="All")


# =============================================================================
# Reconstruction / Preview Management
# =============================================================================
def build_glb_path(session_id: str, conf_thres: float, frame_filter: str, show_cam: bool) -> Path:
    sess = SESSION_MANAGER.get_session(session_id)
    safe_filter = safe_filename(str(frame_filter), fallback="All")
    return sess["imgs_dir"].parent / f"preview_{float(conf_thres):.2f}_{safe_filter}_cam{bool(show_cam)}.glb"

def get_frame_filter_choices(session_id: str) -> list[str]:
    sess = SESSION_MANAGER.get_session(session_id)
    names = sorted(p.name for p in sess["imgs_dir"].iterdir() if p.is_file())
    return [f"{i}: {name}" for i, name in enumerate(names)]

def gradio_demo(model: Pi3X, session_id: str, conf_thres=CONFIG.default_confidence_percent, frame_filter="All", show_cam=True):
    if not session_id or session_id == "None":
        return None, "Error: no valid uploaded input.", gr.Dropdown(choices=["All"], value="All"), "None"

    start_time = time.time()
    cleanup_memory()

    try:
        predictions = SESSION_MANAGER.get_predictions(session_id) or run_model(session_id, model)
        frame_filter_choices = get_frame_filter_choices(session_id)

        idx = parse_frame_index(frame_filter)
        if idx is None or not (0 <= idx < len(frame_filter_choices)):
            frame_filter = "All"

        glbfile, export_dir, point_count = export_reconstruction(
            session_id, predictions, conf_thres, frame_filter, show_cam
        )

        elapsed = time.time() - start_time
        cleanup_memory()

        return (
            str(glbfile),
            f"Exported {point_count:,} points to: `{export_dir}` ({elapsed:.2f}s)",
            gr.Dropdown(choices=["All"] + frame_filter_choices, value=frame_filter, interactive=True),
            str(export_dir),
        )
    except Exception as error:
        cleanup_memory()
        print("="*50 + "\n🚨 FULL TRACEBACK ERROR:")
        traceback.print_exc()
        print("="*50)
        return None, f"Error: `{error}`", gr.Dropdown(choices=["All"], value="All"), "None"

def update_visualization(session_id, conf_thres, frame_filter, show_cam):
    if not session_id or session_id == "None":
        return None, "No reconstruction cache."

    try:
        predictions = SESSION_MANAGER.get_predictions(session_id)
        if predictions is None:
            return None, "No reconstruction cache."

        glbfile = build_glb_path(session_id, conf_thres, frame_filter, show_cam)
        if not glbfile.exists():
            glbscene = predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, show_cam=show_cam)
            glbscene.export(file_obj=str(glbfile))

        return str(glbfile), ""
    except Exception as error:
        return None, f"Error: `{error}`"

def clear_fields(session_id):
    if session_id and session_id != "None":
        SESSION_MANAGER.destroy_session(session_id)
    return (
        None, "None", "None", None, "", None, None, None,
        CONFIG.default_confidence_percent,
        gr.Dropdown(choices=["All"], value="All"),
        True
    )


# =============================================================================
# UI Build
# =============================================================================
def build_ui(model: Pi3X):
    css_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;700&display=swap');
    @keyframes gradient-animation { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    .gradio-container { font-family: 'Rajdhani', sans-serif; background: linear-gradient(-45deg, #020617, #111827, #082f49, #312e81); background-size: 400% 400%; animation: gradient-animation 20s ease infinite; color: #d1d5db; }
    .gradio-container, .gr-label label, .gr-input, input, textarea, .gr-check-radio label { color: #d1d5db !important; }
    .gr-block.gr-group { background-color: rgba(17, 24, 39, 0.60); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(55, 65, 81, 0.55); border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35); }
    .gr-button { background: linear-gradient(to right, #4f46e5, #7c3aed, #0ea5e9) !important; background-size: 200% auto !important; color: white !important; font-weight: bold !important; border: none !important; border-radius: 10px !important; box-shadow: 0 4px 15px rgba(79, 70, 229, 0.5) !important; font-family: 'Orbitron', sans-serif !important; }
    .intro-content { font-size: 16px !important; line-height: 1.55; color: #cbd5e1 !important; }
    .intro-content p { color: #cbd5e1 !important; }
    .intro-content h1 { font-family: 'Orbitron', sans-serif; font-size: 2.1em !important; font-weight: 900; text-align: center; color: #e0f2fe !important; margin-bottom: 4px; }
    .intro-content .subtitle { text-align: center; margin-bottom: 14px; }
    .export-log { min-height: 24px; margin-top: 6px; }
    .export-log p { margin: 0 !important; padding: 2px 2px !important; font-family: 'Rajdhani', sans-serif !important; font-size: 14px !important; font-weight: 400 !important; line-height: 1.35 !important; text-align: left !important; color: #94a3b8 !important; }
    </style>
    """

    with gr.Blocks() as demo:
        session_id_output = gr.Textbox(label="Session ID", visible=False, value="None")
        export_dir_output = gr.Textbox(label="Export Dir", visible=False, value="None")

        gr.HTML(css_style + '<div class="intro-content"><h1>🌌 π³ Local Point Cloud Export</h1><p class="subtitle">PLY point cloud and Blender camera export.</p></div>')

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Media")
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", type="filepath", label="Or Upload Images", interactive=True)
                    interval = gr.Number(None, label="Frame / Image Interval", info="Empty: video uses about 1 FPS; images use all files.", precision=0)
                    process_btn = gr.Button("Prepare Input", variant="secondary")

                image_gallery = gr.Gallery(label="Image Preview", columns=4, height="300px", show_download_button=True, object_fit="contain", preview=True)

            with gr.Column(scale=2):
                gr.Markdown("### 2. View Reconstruction")
                reconstruction_output = gr.Model3D(height=480, zoom_speed=0.5, pan_speed=0.5, label="3D Output")
                log_output = gr.Markdown("", elem_classes=["export-log"])

                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", scale=3, variant="primary")
                    clear_btn = gr.Button("Clear", scale=1)

                with gr.Group():
                    gr.Markdown("### 3. Adjust Visualization")
                    with gr.Row():
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=CONFIG.default_confidence_percent, step=0.1, label="Confidence Threshold (%)")
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")

        upload_inputs = [input_video, input_images, interval, session_id_output]
        upload_outputs = [reconstruction_output, session_id_output, image_gallery, log_output, frame_filter]
        
        # 绑定触发提交或视频/图片导入，消除 interval 修改引发的多次无效刷新
        process_btn.click(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
        input_video.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)
        input_images.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)

        submit_btn.click(
            fn=lambda: "Reconstructing...", outputs=[log_output]
        ).then(
            fn=partial(gradio_demo, model),
            inputs=[session_id_output, conf_thres, frame_filter, show_cam],
            outputs=[reconstruction_output, log_output, frame_filter, export_dir_output],
        )

        vis_inputs = [session_id_output, conf_thres, frame_filter, show_cam]
        vis_outputs = [reconstruction_output, log_output]
        
        for comp in [conf_thres, frame_filter, show_cam]:
            comp.change(fn=update_visualization, inputs=vis_inputs, outputs=vis_outputs)

        clear_btn.click(
            fn=clear_fields,
            inputs=[session_id_output],
            outputs=[reconstruction_output, session_id_output, export_dir_output, image_gallery, log_output, input_video, input_images, interval, conf_thres, frame_filter, show_cam],
        )

    return demo


# =============================================================================
# Local Model Loading and Entry Point
# =============================================================================
def load_local_model() -> Pi3X:
    if not CONFIG.pi3_checkpoint.exists():
        raise FileNotFoundError(f"\nLocal Pi3X checkpoint was not found.\nExpected: {CONFIG.pi3_checkpoint}\n")

    device = require_cuda()
    print("=" * 72 + f"\nLoading Pi3X local checkpoint\nCheckpoint: {CONFIG.pi3_checkpoint}\nDevice: {device}\n" + "=" * 72)

    model = Pi3X()
    state_dict = load_file(str(CONFIG.pi3_checkpoint), device="cpu") if CONFIG.pi3_checkpoint.suffix.lower() == ".safetensors" else torch.load(str(CONFIG.pi3_checkpoint), map_location="cpu", weights_only=False)

    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing: print(f"⚠️ Missing keys during model load: {len(missing)}")
    if unexpected: print(f"⚠️ Unexpected keys during model load: {len(unexpected)}")

    del state_dict

    model.disable_multimodal(free_cuda_cache=True)
    model._chunked_conv_head = partial(model._chunked_conv_head, chunk_size=8)

    model = model.to(device).eval()
    cleanup_memory(force_cuda=True)
    return model

if __name__ == "__main__":
    try:
        ensure_directories()
        model = load_local_model()
        demo = build_ui(model)

        print("=" * 72 + f"\nPi3 Local UI\nURL: http://{SERVER_NAME}:{SERVER_PORT}\nOutput: {CONFIG.output_root}\n" + "=" * 72)

        demo.queue(max_size=QUEUE_MAX_SIZE).launch(
            server_name=SERVER_NAME,
            server_port=SERVER_PORT,
            share=SHARE,
            inbrowser=True,
            show_error=True,
        )

    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
