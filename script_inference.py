import glob
import os
import json

import argparse

base_path = "./data/different_types"

p = argparse.ArgumentParser()
p.add_argument("--base_path", type=str, default=base_path)
p.add_argument("--case_name", type=str, required=True)
# p.add_argument("--wandb_entity", type=str, required=True)

base_path = p.parse_args().base_path
case_name = p.parse_args().case_name

# Set entity to control where wandb logs are stored
# entity = p.parse_args().wandb_entity
# os.environ["WANDB_ENTITY"] = entity

if (case_name == "all"):
    dir_names = glob.glob(f"experiments/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]

        os.system(
            f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
        )
else:
    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )