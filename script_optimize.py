import glob
import os
import json

import argparse
import subprocess

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")

p = argparse.ArgumentParser()
p.add_argument("--base_path", type=str, default=base_path)
p.add_argument("--case_name", type=str, required=True)
p.add_argument("--wandb_entity", type=str, required=True)
p.add_argument("--max_iter", type=int, default=20)

base_path = p.parse_args().base_path
case_name = p.parse_args().case_name
max_iter = p.parse_args().max_iter

# Set entity to control where wandb logs are stored
entity = p.parse_args().wandb_entity
os.environ["WANDB_ENTITY"] = entity

if (case_name == "all"):
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        
        # Read the train test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)

        train_frame = split["train"][1]

    subprocess.run(["python", "optimize_cma.py", "--base_path", base_path, "--case_name", case_name, "--train_frame", str(train_frame), "--max_iter", str(max_iter)])

else:
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    subprocess.run(["python", "optimize_cma.py", "--base_path", base_path, "--case_name", case_name, "--train_frame", str(train_frame), "--max_iter", str(max_iter)])
