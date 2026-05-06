import os
from argparse import ArgumentParser
from pathlib import Path
import time
import logging
import json
import glob
import shutil
import subprocess
import sys

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument(
    "--shape_mesh_path",
    type=str,
    default=None,
    help="Path to a user-provided GLB mesh used when --shape_prior is set.",
)
args = parser.parse_args()

# Set the debug flags
PROCESS_SEG = True
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.hand"
CONTROLLER_NAME = "hand"
SHAPE_PRIOR = args.shape_prior
shape_mesh_path = Path(args.shape_mesh_path) if args.shape_mesh_path else None
case_dir = Path(base_path) / case_name

logger = None


def setup_logger(log_file="timer.log"):
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run_command(command):
    subprocess.run(command, check=True)


def script_command(script_path, *script_args):
    return [sys.executable, script_path, *script_args]


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    if shape_mesh_path is None:
        raise ValueError("--shape_mesh_path is required when --shape_prior is set")
    if shape_mesh_path.suffix.lower() != ".glb":
        raise ValueError("--shape_mesh_path must point to a .glb file")
    if not shape_mesh_path.exists():
        raise FileNotFoundError(f"Shape mesh not found: {shape_mesh_path}")

    shape_dir = case_dir / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    target_mesh_path = shape_dir / "object.glb"
    if shape_mesh_path.resolve() != target_mesh_path.resolve():
        shutil.copy2(shape_mesh_path, target_mesh_path)

    matching_dir = shape_dir / "matching"
    if matching_dir.exists():
        shutil.rmtree(matching_dir)


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        run_command(
            script_command(
                "./data_process/segment.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--TEXT_PROMPT",
                TEXT_PROMPT,
            )
        )


if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        run_command(
            script_command(
                "./data_process/dense_track.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            )
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        run_command(
            script_command(
                "./data_process/data_process_pcd.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            )
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        run_command(
            script_command(
                "./data_process/data_process_mask.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--controller_name",
                CONTROLLER_NAME,
            )
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        run_command(
            script_command(
                "./data_process/data_process_track.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
            )
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Align the shape prior with partial observation
    with Timer("Alignment"):
        run_command(
            script_command(
                "./data_process/align.py",
                "--base_path",
                base_path,
                "--case_name",
                case_name,
                "--controller_name",
                CONTROLLER_NAME,
            )
        )

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            run_command(
                script_command(
                    "./data_process/data_process_sample.py",
                    "--base_path",
                    base_path,
                    "--case_name",
                    case_name,
                    "--shape_prior",
                )
            )
        else:
            run_command(
                script_command(
                    "./data_process/data_process_sample.py",
                    "--base_path",
                    base_path,
                    "--case_name",
                    case_name,
                )
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
