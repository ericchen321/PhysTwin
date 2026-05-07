#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="phystwin"
PYTHON_VERSION="3.10"
CUDA_12_8_HOME="/usr/local/cuda-12.8"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda was not found on PATH. Initialize conda first, then rerun this script."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
hash -r

if [ -d "${CUDA_12_8_HOME}" ]; then
    export CUDA_HOME="${CUDA_12_8_HOME}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "Using CUDA from ${CUDA_HOME}"
    nvcc --version
else
    cat <<EOF
CUDA 12.8 was not found at ${CUDA_12_8_HOME}.
The PyTorch cu128 wheel can still install, but CUDA extension builds may fail.
EOF
fi

current_python="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${current_python}" != "${PYTHON_VERSION}" ]; then
    cat <<EOF
Conda env "${ENV_NAME}" already exists with Python ${current_python}.
This install script expects Python ${PYTHON_VERSION}.

Remove or rename the existing env before rerunning, for example:
  conda env remove -n ${ENV_NAME}
  bash env_install/env_install.sh
EOF
    exit 1
fi

python_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [ "${python_prefix}" != "${CONDA_PREFIX}" ]; then
    echo "python does not point to the active conda env. Refusing to install."
    exit 1
fi

if ! pip --version | grep -q "${CONDA_PREFIX}"; then
    echo "pip does not point to ${CONDA_PREFIX}. Refusing to install."
    pip --version || true
    exit 1
fi

pip install --upgrade pip setuptools wheel

echo "Installing packages into conda env ${ENV_NAME}:"
which python
python -c 'import sys; print(sys.executable); print(sys.prefix)'
pip --version

conda install -y numpy==1.26.4
pip install warp-lang==1.7.1 # a known version to be compatible
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree
pip install pyrender

# conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# NOTE: use CUDA 12.8-compiled torch for now, since I do not have CUDA 12.1 installed natively
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html
mkdir -p third_party/
cd third_party/
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d/
pip install -e . --no-build-isolation
cd ../../

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for grounded-sam-2
pip install git+https://github.com/IDEA-Research/Grounded-SAM-2.git

# Install GroundingDINO with patch for PyTorch >= 2.x compatibility:
# AT_DISPATCH_FLOATING_TYPES(value.type(), ...) -> value.scalar_type()
# because DeprecatedTypeProperties no longer implicitly converts to c10::ScalarType.
GDINO_TMP="$(mktemp -d)/GroundingDINO"
git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git "${GDINO_TMP}"
sed -i \
    's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    "${GDINO_TMP}/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
pip install --no-build-isolation "${GDINO_TMP}"

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

# Install the env for trellis
# NOTE: we use OmniPart-generated mesh, so won't call TRELLIS here to generate mesh
# cd data_process
# git clone --recurse-submodules https://github.com/microsoft/TRELLIS
# cd TRELLIS
# . ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# cd ../..

pip install gsplat==1.4.0
pip install kornia
# cd gaussian_splatting/
# pip install submodules/diff-gaussian-rasterization/
# pip install submodules/simple-knn/
# cd ..
cd gaussian_splatting/submodules/diff-gaussian-rasterization/
python setup.py build_ext --inplace
pip install --no-build-isolation -e .
cd ../simple-knn/
python setup.py build_ext --inplace
pip install --no-build-isolation -e .
cd ../../../

pip install plyfile einops