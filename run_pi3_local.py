from __future__ import annotations

import gc, glob, json, os, shutil, sys, time, traceback
from datetime import datetime
from pathlib import Path

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

# =============================================================================
# Local configuration
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
PI3_CHECKPOINT = REPO_ROOT / "weights" / "Pi3X" / "model.safetensors"
GEOCALIB_MODELS_DIR = REPO_ROOT / "weights" / "geocalib"
GEOCALIB_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT = REPO_ROOT / "outputs"
WORK_ROOT = REPO_ROOT / "_gradio_work"

# 设置 PyTorch Hub 缓存目录，避免动态 Patching
os.environ["TORCH_HOME"] = str(GEOCALIB_MODELS_DIR)

DEFAULT_INTERVAL = None
DEFAULT_CONFIDENCE_PERCENT = 20.0
SAVE_PER_VIEW_GEOMETRY = False
SAVE_CAMERA_JSON = True

BLENDER_PLACEHOLDER_LENS_MM = 50.0
BLENDER_SENSOR_WIDTH_MM = 36.0

PREVIEW_CAMERA_WIDTH_RATIO = 0.025
PREVIEW_CAMERA_HEIGHT_RATIO = 0.050
PREVIEW_SRGB_TO_LINEAR = True
PREVIEW_COLOR_GAIN = 1.0

SERVER_NAME = os.environ.get("PI3_SERVER_NAME", "127.0.0.1")
SERVER_PORT = int(os.environ.get("PI3_SERVER_PORT", "7860"))
SHARE = os.environ.get("PI3_SHARE", "0").strip().lower() in {"1", "true", "yes"}
QUEUE_MAX_SIZE = 2


# =============================================================================
# Helper functions
# =============================================================================
def get_geocalib_model(weights="pinhole", device="cuda"):
    """获取 GeoCalib 模型（利用 TORCH_HOME，无需动态修补）"""
    if not GEOCALIB_AVAILABLE:
        return None
    if weights not in _GEOCALIB_MODELS:
        print(f"[GeoCalib] Loading model '{weights}' from: {GEOCALIB_MODELS_DIR}")
        _GEOCALIB_MODELS[weights] = GeoCalib(weights=weights).eval().to("cpu")
    return _GEOCALIB_MODELS[weights].to(device).eval()

def unload_geocalib_from_cuda(delete_model: bool = False) -> None:
    """彻底将 GeoCalib 权重从 GPU 中移出并清理显存"""
    global _GEOCALIB_MODELS
    for weights, geo_model in list(_GEOCALIB_MODELS.items()):
        try:
            geo_model.to("cpu")
        except Exception:
            pass
        if delete_model:
            del _GEOCALIB_MODELS[weights]

    if delete_model:
        _GEOCALIB_MODELS.clear()
    cleanup_memory()

def rotation_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """返回 4x4 旋转矩阵，使 source 向量旋转对齐到 target 向量"""
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)

    # 求解把 source 旋转到 target 的旋转矩阵
    R_mat, _ = Rotation.align_vectors(target[None, :], source[None, :])
    
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_mat.as_matrix().astype(np.float32)
    return T

def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_directories() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

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
    export_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_dir.mkdir(parents=True, exist_ok=False)
    return export_dir

def transform_world_points(points: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ world_transform[:3, :3].T + world_transform[:3, 3]

def transform_camera_poses(camera_to_world: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    return world_transform[None, :, :] @ np.asarray(camera_to_world, dtype=np.float32)

def parse_frame_index(filter_str: str | None) -> int | None:
    if not filter_str or str(filter_str).lower() == "all":
        return None
    try:
        return int(str(filter_str).split(":")[0])
    except (ValueError, IndexError):
        return None

def robust_mean_unit_vectors(vectors: list[np.ndarray], max_angle_deg: float = 15.0):
    if len(vectors) == 0:
        return None, None, -1
    
    vecs = np.asarray([v / max(np.linalg.norm(v), 1e-12) for v in vectors], dtype=np.float64)
    if len(vecs) == 1:
        return vecs[0].astype(np.float32), np.ones(1, dtype=bool), 0

    similarity = vecs @ vecs.T
    medoid_idx = int(np.argmax(np.sum(similarity, axis=1)))
    reference = vecs[medoid_idx]

    aligned = vecs.copy()
    aligned[(aligned @ reference) < 0.0] *= -1.0

    cos_thresh = np.cos(np.deg2rad(max_angle_deg))
    inliers = (aligned @ reference) >= cos_thresh

    if not np.any(inliers):
        inliers[medoid_idx] = True

    mean_vec = aligned[inliers].mean(axis=0)
    mean_vec /= max(np.linalg.norm(mean_vec), 1e-12)
    return mean_vec.astype(np.float32), inliers, medoid_idx


# =============================================================================
# Preview GLB
# =============================================================================
def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float) -> None:
    scene_scale = max(float(scene_scale), 1e-5)
    cam_w, cam_h = scene_scale * PREVIEW_CAMERA_WIDTH_RATIO, scene_scale * PREVIEW_CAMERA_HEIGHT_RATIO

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot[2, 3] = -cam_h

    opengl_transform = np.diag([1.0, -1.0, -1.0, 1.0])
    complete_transform = transform @ opengl_transform @ rot

    cone = trimesh.creation.cone(cam_w, cam_h, sections=4)
    slight_rot = np.eye(4)
    slight_rot[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    t_vertices = (cone.vertices @ slight_rot[:3, :3].T) + slight_rot[:3, 3]
    v_combined = np.concatenate([cone.vertices, 0.95 * cone.vertices, t_vertices])
    
    v_homo = np.pad(v_combined, ((0, 0), (0, 1)), constant_values=1.0)
    v_transformed = (v_homo @ complete_transform.T)[:, :3]

    num_v = len(cone.vertices)
    faces_list = []
    for f in cone.faces:
        if 0 in f: continue
        v1, v2, v3 = f
        faces_list.extend([
            (v1, v2, v2 + num_v), (v1, v1 + num_v, v3), (v3 + num_v, v2, v3),
            (v1, v2, v2 + 2*num_v), (v1, v1 + 2*num_v, v3), (v3 + 2*num_v, v2, v3)
        ])
    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]

    camera_mesh = trimesh.Trimesh(vertices=v_transformed, faces=np.array(faces_list))
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

def predictions_to_glb(predictions: dict, conf_thres: float = DEFAULT_CONFIDENCE_PERCENT, filter_by_frames: str = "All", show_cam: bool = True) -> trimesh.Scene:
    idx = parse_frame_index(filter_by_frames)
    
    pts = np.asarray(predictions["points"])
    conf = np.asarray(predictions["conf"])
    imgs = np.asarray(predictions["images"])
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    poses = np.asarray(predictions["camera_poses"])

    if idx is not None and 0 <= idx < len(pts):
        pts, conf, imgs, poses = pts[idx:idx+1], conf[idx:idx+1], imgs[idx:idx+1], poses[idx:idx+1]

    vertices_3d = pts.reshape(-1, 3)
    source_rgb = np.clip(imgs.reshape(-1, 3), 0.0, 1.0)
    preview_rgb = srgb_to_linear(source_rgb) if PREVIEW_SRGB_TO_LINEAR else source_rgb
    colors_rgb = np.clip(preview_rgb * PREVIEW_COLOR_GAIN * 255.0, 0, 255).astype(np.uint8)

    conf_flat = conf.reshape(-1)
    mask = np.isfinite(vertices_3d).all(axis=1) & np.isfinite(conf_flat) & (conf_flat >= max(float(conf_thres) / 100.0, 0.0)) & (conf_flat > 1e-5)

    vertices_3d, colors_rgb = vertices_3d[mask], colors_rgb[mask]

    if vertices_3d.size == 0:
        vertices_3d, colors_rgb, scene_scale = np.array([[1.0, 0.0, 0.0]], dtype=np.float32), np.array([[255, 255, 255]], dtype=np.uint8), 1.0
    else:
        p5, p95 = np.percentile(vertices_3d, [5, 95], axis=0)
        scene_scale = max(float(np.linalg.norm(p95 - p5)), 1e-5)

    scene_3d = trimesh.Scene(trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb))

    if show_cam:
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        n_cams = len(poses)
        for i, pose in enumerate(poses):
            color = tuple(int(255 * x) for x in cmap(i / max(n_cams, 1))[:3])
            integrate_camera_into_scene(scene_3d, pose, color, scene_scale)

    scene_3d.apply_transform(np.eye(4, dtype=np.float32))
    return scene_3d


# =============================================================================
# Formal PLY and camera exports
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
    points = np.asarray(predictions["points"], dtype=np.float32)
    conf = np.asarray(predictions["conf"][..., 0], dtype=np.float32)
    imgs = np.asarray(predictions["images"])
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        imgs = np.transpose(imgs, (0, 2, 3, 1))

    valid = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf >= float(conf_thres) / 100.0)
    vertices = points[valid].reshape(-1, 3)
    colors = np.clip(imgs[valid].reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)

    if len(vertices) == 0:
        raise RuntimeError("No point remains after confidence filtering. Lower Confidence Threshold.")

    # X 轴旋转 +90 度校正
    R_x90 = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    vertices = vertices @ R_x90.T

    output_path = export_dir / "pointcloud.ply"
    write_binary_ply(output_path, vertices, colors)
    return output_path, len(vertices)

def write_blender_camera_importer(pose_npz_path: Path, output_path: Path) -> None:
    script = f'''\
import bpy, numpy as np
from mathutils import Matrix

NPZ_PATH = r"{pose_npz_path}"
COLLECTION_NAME, CAMERA_PREFIX = "Pi3_Cameras", "Pi3_Camera_"
LENS_MM, SENSOR_WIDTH_MM = {BLENDER_PLACEHOLDER_LENS_MM}, {BLENDER_SENSOR_WIDTH_MM}

data = np.load(NPZ_PATH)
poses_cv_c2w = data["camera_to_world_opencv"]
resolutions = data.get("resolutions", np.tile([int(data.get("image_width", 1920)), int(data.get("image_height", 1080))], (len(poses_cv_c2w), 1)))

if old_col := bpy.data.collections.get(COLLECTION_NAME):
    for obj in list(old_col.objects): bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old_col)

collection = bpy.data.collections.new(COLLECTION_NAME)
bpy.context.scene.collection.children.link(collection)

opencv_to_blender = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))

for i, pose_np in enumerate(poses_cv_c2w):
    res_x, res_y = int(resolutions[i][0]), int(resolutions[i][1])
    cam = bpy.data.cameras.new(f"{{CAMERA_PREFIX}}{{i:04d}}")
    cam.type, cam.lens, cam.sensor_width = "PERSP", LENS_MM, SENSOR_WIDTH_MM

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
print(f"Imported {{len(poses_cv_c2w)}} real Blender cameras.")
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

    # 相机位姿 X 轴 +90 度校正
    T_x90 = np.eye(4, dtype=np.float32)
    T_x90[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    poses = transform_camera_poses(poses, T_x90)

    npz_path = export_dir / "camera_poses.npz"
    np.savez_compressed(
        npz_path,
        camera_to_world_opencv=poses,
        resolutions=resolutions,
        image_width=np.int32(img_w),
        image_height=np.int32(img_h),
        pose_convention=np.array("OpenCV camera_to_world"),
        export_alignment_applied=np.bool_(True),
    )

    json_path = export_dir / "camera_poses.json" if SAVE_CAMERA_JSON else None
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

def export_per_view_geometry_if_enabled(predictions: dict, export_dir: Path) -> Path | None:
    local_points = predictions.get("local_points_for_export")
    if not SAVE_PER_VIEW_GEOMETRY or local_points is None:
        return None

    output_path = export_dir / "per_view_geometry.npz"
    np.savez_compressed(
        output_path,
        local_points=np.asarray(local_points, dtype=np.float32),
        conf=np.asarray(predictions["conf"], dtype=np.float32),
        camera_poses=np.asarray(predictions["camera_poses"], dtype=np.float32),
    )
    return output_path


# =============================================================================
# Optimized Execution Function
# =============================================================================
def run_model(target_dir: str, model: Pi3X) -> dict:
    device = require_cuda()

    # 1. 确保在 Pi3X 计算前卸载 GeoCalib
    unload_geocalib_from_cuda(delete_model=False)

    model = model.to(device).eval()

    image_dir = os.path.join(target_dir, "images")
    if not glob.glob(os.path.join(image_dir, "*")):
        raise ValueError("No images found. Check the uploaded input.")

    imgs = load_images_as_tensor(image_dir, interval=1).to(device)

    num_frames = len(imgs)
    sample_indices = np.unique(np.linspace(0, num_frames - 1, min(9, num_frames), dtype=int))

    # 将 GeoCalib 采样帧移至 CPU 暂存
    geocalib_frames_cpu = {int(i): imgs[int(i)].detach().cpu() for i in sample_indices}

    # 2. 模型前向推理
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=get_autocast_dtype()):
        predictions = model(imgs[None], with_prior=False)

    predictions["conf"] = torch.sigmoid(predictions["conf"])

    pts_gpu = predictions["points"] if "points" in predictions else predictions["local_points"]
    poses_gpu = predictions["camera_poses"]
    conf_gpu = predictions["conf"]

    # 3. 转换至 CPU NumPy 斩断 GPU 引用
    output = {
        "images": imgs[None].permute(0, 1, 3, 4, 2).detach().float().cpu().numpy().squeeze(0),
        "points": pts_gpu.detach().float().cpu().numpy().squeeze(0),
        "conf": conf_gpu.detach().float().cpu().numpy().squeeze(0),
        "camera_poses": poses_gpu.detach().float().cpu().numpy().squeeze(0),
    }

    # 4. 清理 GPU 资源
    del predictions, pts_gpu, poses_gpu, conf_gpu, imgs
    cleanup_memory()

    # GeoCalib 校准
    if GEOCALIB_AVAILABLE and len(geocalib_frames_cpu) > 0:
        try:
            geo_model = get_geocalib_model(weights="pinhole", device=device)
            g_worlds = []
            print(f"[GeoCalib] Estimating gravity from {len(geocalib_frames_cpu)} frame(s)...")

            for idx in sample_indices:
                frame_tensor = geocalib_frames_cpu[int(idx)].to(device, non_blocking=True)
                
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=get_autocast_dtype()):
                    results = geo_model.calibrate(frame_tensor, camera_model="pinhole")

                gravity_obj = results.get("gravity", None)
                if gravity_obj is None:
                    del frame_tensor
                    continue

                # 从 GeoCalib 结果提取重力向量
                g_tensor = getattr(gravity_obj, "vec3d", getattr(gravity_obj, "vec", gravity_obj))
                if hasattr(gravity_obj, "R"):
                    R_g = gravity_obj.R
                    g_tensor = (R_g.detach().cpu().numpy() if isinstance(R_g, torch.Tensor) else R_g)[..., :, 1]

                if isinstance(g_tensor, torch.Tensor):
                    g_cam = g_tensor.detach().float().cpu().numpy().reshape(-1, 3)[0]
                else:
                    g_cam = np.asarray(g_tensor, dtype=np.float32).reshape(-1, 3)[0]

                norm_cam = np.linalg.norm(g_cam)
                if not np.isfinite(norm_cam) or norm_cam < 1e-8:
                    del frame_tensor, results
                    continue
                g_cam /= norm_cam

                c2w = np.asarray(output["camera_poses"][idx], dtype=np.float32)
                g_world_i = c2w[:3, :3] @ g_cam
                g_world_i /= max(np.linalg.norm(g_world_i), 1e-12)
                g_worlds.append(g_world_i)

                del frame_tensor, results

            gravity_world, _, _ = robust_mean_unit_vectors(g_worlds, max_angle_deg=20.0)

            if gravity_world is not None:  
                # 明确将重力向量对齐到真实的向下方向 [0, -1, 0]
                target_down_world = np.array([0.0, -1.0, 0.0], dtype=np.float32)
                align_mat = rotation_align_vectors(gravity_world, target_down_world)

                output["points"] = transform_world_points(output["points"], align_mat)
                output["camera_poses"] = transform_camera_poses(output["camera_poses"], align_mat)

                g_after = align_mat[:3, :3] @ gravity_world
                print(f"📐 [GeoCalib Alignment] 旋转对齐完成！")
                print(f"📐 [GeoCalib Alignment] World Gravity Before: {np.round(gravity_world, 4)} -> After: {np.round(g_after, 4)}")
            else:
                print("⚠ [GeoCalib] 未提取到有效重力向量，保持原坐标系。")

        except Exception as e:
            print(f"⚠️ GeoCalib gravity calibration failed: {e}")
            traceback.print_exc()
        finally:
            unload_geocalib_from_cuda(delete_model=False)
            del geocalib_frames_cpu
            cleanup_memory()

    return output


# =============================================================================
# Upload / video sampling
# =============================================================================
def handle_uploads(input_video, input_images, interval=-1):
    cleanup_memory()
    ensure_directories()

    target_dir = WORK_ROOT / f"input_images_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    target_dir_images = target_dir / "images"
    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir_images.mkdir(parents=True, exist_ok=True)

    image_paths = []
    try:
        interval_int = int(interval)
    except (ValueError, TypeError):
        interval_int = -1

    if input_images:
        selected_images = input_images[::interval_int] if interval_int > 0 else input_images
        for file_data in selected_images:
            src = file_data.get("name", file_data) if isinstance(file_data, dict) else str(file_data)
            if not os.path.isfile(src): continue

            dst = target_dir_images / os.path.basename(src)
            if dst.exists():
                dst = target_dir_images / f"{len(image_paths):06d}{Path(src).suffix}"

            shutil.copy(src, dst)
            image_paths.append(str(dst))

    if input_video:
        v_path = input_video.get("name", input_video) if isinstance(input_video, dict) else str(input_video)
        vs = cv2.VideoCapture(v_path)
        if not vs.isOpened():
            raise RuntimeError(f"Cannot open video: {v_path}")

        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = interval_int if interval_int > 0 else max(1, int(fps * 1))
        
        count, video_frame_num = 0, 0
        try:
            while True:
                gotit, frame = vs.read()
                if not gotit: break
                count += 1
                if count % frame_interval == 0:
                    img_path = target_dir_images / f"{video_frame_num:06d}.png"
                    if not cv2.imwrite(str(img_path), frame):
                        raise RuntimeError(f"Failed to write frame: {img_path}")
                    image_paths.append(str(img_path))
                    video_frame_num += 1
        finally:
            vs.release()

    image_paths = sorted(image_paths)
    if not image_paths:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise ValueError("No valid images or video frames were prepared.")

    return str(target_dir), image_paths

def update_gallery_on_upload(input_video, input_images, interval=-1):
    if not input_video and not input_images:
        return None, "None", None, "", gr.Dropdown(choices=["All"], value="All")

    try:
        target_dir, image_paths = handle_uploads(input_video, input_images, interval=interval)
        return None, target_dir, image_paths, f"Prepared {len(image_paths)} images. Click Reconstruct.", gr.Dropdown(choices=["All"], value="All")
    except Exception as error:
        return None, "None", None, f"Error: `{error}`", gr.Dropdown(choices=["All"], value="All")


# =============================================================================
# Reconstruction / preview
# =============================================================================
def build_glb_path(target_dir: str, conf_thres: float, frame_filter: str, show_cam: bool) -> str:
    safe_filter = safe_filename(str(frame_filter), fallback="All")
    return os.path.join(target_dir, f"preview_{float(conf_thres):.2f}_{safe_filter}_cam{bool(show_cam)}.glb")

def save_preview_predictions(target_dir: str, predictions: dict) -> None:
    np.savez_compressed(
        os.path.join(target_dir, "predictions_preview.npz"),
        images=np.asarray(predictions["images"]),
        points=np.asarray(predictions["points"]),
        conf=np.asarray(predictions["conf"]),
        camera_poses=np.asarray(predictions["camera_poses"]),
    )

def load_preview_predictions(target_dir: str) -> dict:
    path = os.path.join(target_dir, "predictions_preview.npz")
    if not os.path.exists(path):
        raise FileNotFoundError("No cached reconstruction. Click Reconstruct first.")
    loaded = np.load(path)
    return {k: np.asarray(loaded[k]) for k in ["images", "points", "conf", "camera_poses"]}

def gradio_demo(target_dir, conf_thres=DEFAULT_CONFIDENCE_PERCENT, frame_filter="All", show_cam=True):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "Error: no valid uploaded input.", gr.Dropdown(choices=["All"], value="All"), "None"

    start_time = time.time()
    cleanup_memory()

    try:
        predictions = run_model(target_dir, model)

        all_files = sorted(os.listdir(os.path.join(target_dir, "images")))
        frame_filter_choices = [f"{i}: {f}" for i, f in enumerate(all_files)]

        idx = parse_frame_index(frame_filter)
        if idx is None or not (0 <= idx < len(all_files)):
            frame_filter = "All"

        if predictions is None:
            raise RuntimeError("run_model() 返回了 None，请检查模型推理逻辑！")
            
        save_preview_predictions(target_dir, predictions)
        export_dir = create_export_dir()

        _, point_count = export_pointcloud_ply(predictions, export_dir, conf_thres)
        export_camera_poses(predictions, export_dir)
        export_per_view_geometry_if_enabled(predictions, export_dir)

        glbfile = build_glb_path(target_dir, conf_thres, frame_filter, show_cam)
        glbscene = predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, show_cam=show_cam)
        glbscene.export(file_obj=glbfile)

        elapsed = time.time() - start_time
        del predictions
        cleanup_memory()

        return (
            glbfile,
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

def update_visualization(target_dir, conf_thres, frame_filter, show_cam):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction cache."

    try:
        predictions = load_preview_predictions(target_dir)
        glbfile = build_glb_path(target_dir, conf_thres, frame_filter, show_cam)

        if not os.path.exists(glbfile):
            glbscene = predictions_to_glb(predictions, conf_thres=conf_thres, filter_by_frames=frame_filter, show_cam=show_cam)
            glbscene.export(file_obj=glbfile)

        return glbfile, ""
    except Exception as error:
        return None, f"Error: `{error}`"

def clear_fields():
    return None, "None", "None", None, "", None, None, DEFAULT_INTERVAL, DEFAULT_CONFIDENCE_PERCENT, gr.Dropdown(choices=["All"], value="All"), True


# =============================================================================
# UI
# =============================================================================
def build_ui():
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
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        export_dir_output = gr.Textbox(label="Export Dir", visible=False, value="None")

        gr.HTML(css_style + '<div class="intro-content"><h1>🌌 π³ Local Point Cloud Export</h1><p class="subtitle">PLY point cloud and Blender camera export.</p></div>')

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 1. Upload Media")
                    input_video = gr.Video(label="Upload Video", interactive=True)
                    input_images = gr.File(file_count="multiple", label="Or Upload Images", interactive=True)
                    interval = gr.Number(DEFAULT_INTERVAL, label="Frame / Image Interval", info="Empty: video uses about 1 FPS; images use all files.", precision=0)

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
                        conf_thres = gr.Slider(minimum=0, maximum=100, value=DEFAULT_CONFIDENCE_PERCENT, step=0.1, label="Confidence Threshold (%)")
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                    frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")

        upload_inputs = [input_video, input_images, interval]
        upload_outputs = [reconstruction_output, target_dir_output, image_gallery, log_output, frame_filter]
        
        for comp in upload_inputs:
            comp.change(fn=update_gallery_on_upload, inputs=upload_inputs, outputs=upload_outputs)

        submit_btn.click(
            fn=lambda: "Reconstructing...", outputs=[log_output]
        ).then(
            fn=gradio_demo,
            inputs=[target_dir_output, conf_thres, frame_filter, show_cam],
            outputs=[reconstruction_output, log_output, frame_filter, export_dir_output],
        )

        vis_inputs = [target_dir_output, conf_thres, frame_filter, show_cam]
        vis_outputs = [reconstruction_output, log_output]
        
        # 批量绑定可视化参数的 change 事件
        for comp in [conf_thres, frame_filter, show_cam]:
            comp.change(fn=update_visualization, inputs=vis_inputs, outputs=vis_outputs)

        clear_btn.click(
            fn=clear_fields,
            outputs=[reconstruction_output, target_dir_output, export_dir_output, image_gallery, log_output, input_video, input_images, interval, conf_thres, frame_filter, show_cam],
        )

    return demo


# =============================================================================
# Local model loading and launch
# =============================================================================
def load_local_model() -> Pi3X:
    if not PI3_CHECKPOINT.exists():
        raise FileNotFoundError(f"\nLocal Pi3X checkpoint was not found.\nExpected: {PI3_CHECKPOINT}\n")

    device = require_cuda()
    print("=" * 72 + f"\nLoading Pi3X local checkpoint\nCheckpoint: {PI3_CHECKPOINT}\nDevice: {device}\n" + "=" * 72)

    model = Pi3X()
    state_dict = load_file(str(PI3_CHECKPOINT), device="cpu") if PI3_CHECKPOINT.suffix.lower() == ".safetensors" else torch.load(str(PI3_CHECKPOINT), map_location="cpu", weights_only=False)

    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.disable_multimodal(free_cuda_cache=True)

    from functools import partial
    model._chunked_conv_head = partial(model._chunked_conv_head, chunk_size=8)

    model.eval()
    cleanup_memory()
    return model

if __name__ == "__main__":
    try:
        ensure_directories()
        model = load_local_model()
        demo = build_ui()

        print("=" * 72 + f"\nPi3 Local UI\nURL: http://{SERVER_NAME}:{SERVER_PORT}\nOutput: {OUTPUT_ROOT}\n" + "=" * 72)

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
