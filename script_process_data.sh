#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate phystwin

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$LD_PRELOAD"

python process_data.py \
    --base_path "$1" \
    --case_name "$2" \
    --category "$3" \
    "${@:4}"
