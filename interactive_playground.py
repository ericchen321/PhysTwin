from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Camera vector must be non-zero")
    return v / norm


def camera_setup_to_c2w(camera_setup):
    position = np.array(camera_setup["position"], dtype=np.float64)
    look_at = np.array(camera_setup["look_at"], dtype=np.float64)
    up = np.array(camera_setup["up"], dtype=np.float64)

    z_axis = _normalize_vector(position - look_at)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        raise ValueError("Camera up vector cannot be parallel to view direction")
    x_axis = _normalize_vector(x_axis)
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    c2w[:3, 3] = position
    return c2w


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
    parser.add_argument(
        "--virtual_key_input", action="store_true", help="use virtual key input"
    )
    parser.add_argument(
        "--auto_traj",
        type=str,
        choices=["circle", "swing"],
        default=None,
        help="auto trajectory: 'circle' = lift then clockwise rotation; 'swing' = lift then back-and-forth",
    )
    parser.add_argument(
        "--render_output_root",
        type=str,
        default="./interactive_playground_renders",
    )
    parser.add_argument("--render_capture_fps", type=float, default=10.0)
    parser.add_argument("--no_png_renders", action="store_true")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    render_output_dir = None
    if args.auto_traj is not None and not args.no_png_renders:
        render_output_dir = os.path.join(
            args.render_output_root, f"{case_name}_auto_traj_{args.auto_traj}"
        )

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    base_dir = f"./temp_experiments/{case_name}"

    # Read the first-satage optimized parameters to set the indifferentiable parameters
    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

    camera_setup = {
        "position": [
            -0.36948991180547435,
            -0.33041834779509936,
            -1.1568110038466772,
        ],
        "look_at": [
            -0.6087389312024328,
            -0.5896267181432237,
            -2.2513752675877625,
        ],
        "up": [
            -0.37738162507612366,
            -0.4088114454525919,
            0.8309370079144786,
        ],
    }

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path
    intrinsic_camera_idx = 2
    vis_intrinsic = cfg.intrinsics[intrinsic_camera_idx]
    vis_c2w = camera_setup_to_c2w(camera_setup)
    vis_w2c = np.linalg.inv(vis_c2w)

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.interactive_playground(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
        virtual_key_input=args.virtual_key_input,
        auto_traj=args.auto_traj,
        render_output_dir=render_output_dir,
        render_capture_fps=args.render_capture_fps,
        vis_w2c=vis_w2c,
        vis_intrinsic=vis_intrinsic,
    )
