#!/usr/bin/env bash
# ==============================================================================
# RIPL Lab assignment - ManiSkill 3 environment BUILD
#
# Scope: install the machine + graphics + python layers, and verify only that
#        far. Runtime verification (sim, rendering, data, training) belongs to
#        smoke_test_e2e.sh and is deliberately NOT duplicated here.
#
# Assumes: Ubuntu 22.04 container, NVIDIA GPU, /workspace writable, root.
#
# Usage:   bash setup_runpod.sh
# Then:    source /workspace/ripl/env.sh && bash smoke_test_e2e.sh
# ==============================================================================
set -euo pipefail

ROOT=/workspace/ripl
VENV=$ROOT/venv
PY_VERSION=3.11

fail() { echo ""; echo "FAILED: $*"; exit 1; }

echo "=============================================================="
echo "STEP 1 — preflight"
echo "=============================================================="

[ -d /workspace ] || fail "/workspace does not exist. Attach storage or pick a
template that provides it - everything else on this container is wiped on stop."

nvidia-smi || fail "nvidia-smi failed - no GPU visible to this container."

# Inherit the container's real environment. Interactive shells on RunPod (web
# terminal, SSH) frequently do NOT inherit PID 1's env, so a bare
# `echo $NVIDIA_DRIVER_CAPABILITIES` returns empty on pods where graphics was
# granted correctly. Read it from PID 1 instead, and treat it as a hint only.
if [ -r /proc/1/environ ]; then
  CAPS=$(tr '\0' '\n' < /proc/1/environ | grep '^NVIDIA_DRIVER_CAPABILITIES=' \
         | cut -d= -f2- || true)
  export NVIDIA_DRIVER_CAPABILITIES="${CAPS:-}"
  echo "  PID 1 reports NVIDIA_DRIVER_CAPABILITIES = ${CAPS:-<not set>}"
fi

# THIS is the fact. The env var is an input to container creation; the presence
# of the driver's graphics libraries is evidence about the container you got.
GLX_COUNT=$(ldconfig -p | grep -c 'libGLX_nvidia' || true)
EGL_COUNT=$(ldconfig -p | grep -c 'libEGL_nvidia' || true)
echo "  libGLX_nvidia entries: $GLX_COUNT"
echo "  libEGL_nvidia entries: $EGL_COUNT"

if [ "$GLX_COUNT" -eq 0 ]; then
  echo ""
  echo "  !! libGLX_nvidia.so is not present. The container runtime did not mount"
  echo "  !! the driver's graphics libraries, which means the 'graphics' driver"
  echo "  !! capability was not granted to this container."
  echo "  !!"
  echo "  !! This CANNOT be fixed from inside a running pod. Terminate it and"
  echo "  !! redeploy with NVIDIA_DRIVER_CAPABILITIES=all set in the deploy"
  echo "  !! page's Environment Variables section (more reliable than template"
  echo "  !! inheritance). If it still fails, the host itself is not configured"
  echo "  !! for graphics - try a different machine or datacenter."
  echo ""
  fail "no graphics capability"
fi

echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
mkdir -p "$ROOT"

echo "=============================================================="
echo "STEP 2 — system packages"
echo "=============================================================="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  libvulkan1 vulkan-tools libglvnd-dev libgl1 libegl1 \
  ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
  git git-lfs curl wget tmux htop unzip build-essential
git lfs install --skip-repo || true

echo "=============================================================="
echo "STEP 3 — Vulkan ICD / EGL vendor config"
echo "=============================================================="
# Normally injected by the NVIDIA container runtime alongside the graphics libs.
# Written by hand here for the cases where the libs arrived but the manifests
# did not.
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d /etc/vulkan/implicit_layer.d

if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  echo "  writing nvidia_icd.json"
  cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
EOF
else
  echo "  nvidia_icd.json already present (good sign - runtime injected it)"
fi

if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
  echo "  writing 10_nvidia.json"
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
fi

# Optional in general; required on some datacenter GPUs (A100 in particular).
if [ ! -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then
  echo "  writing nvidia_layers.json"
  cat > /etc/vulkan/implicit_layer.d/nvidia_layers.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": { "__NV_PRIME_RENDER_OFFLOAD": "1" },
        "disable_environment": { "DISABLE_LAYER_NV_OPTIMUS_1": "" }
    }
}
EOF
fi

echo "=============================================================="
echo "STEP 4 — Vulkan verification (hard gate)"
echo "=============================================================="
if vulkaninfo --summary > "$ROOT/vulkaninfo.txt" 2>&1; then
  grep -E "deviceName|driverName|apiVersion" "$ROOT/vulkaninfo.txt" | head -6 | sed 's/^/  /'
  echo "  Vulkan OK"
else
  echo "  Full output: $ROOT/vulkaninfo.txt"
  tail -20 "$ROOT/vulkaninfo.txt" | sed 's/^/  /'
  echo ""
  echo "  ErrorInitializationFailed / missing-extension / segfault all mean the"
  echo "  same family of problem: the loader cannot reach a working NVIDIA ICD."
  echo "  The manifests above are now correct, so if this still fails the driver"
  echo "  libraries are the issue -> terminate and redeploy."
  fail "vulkaninfo did not enumerate a device"
fi

echo "=============================================================="
echo "STEP 5 — python environment (on persistent storage)"
echo "=============================================================="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d "$VENV" ] || uv venv -p "$PY_VERSION" "$VENV"

uv pip install --python "$VENV/bin/python" \
  torch torchvision --index-url https://download.pytorch.org/whl/cu124

uv pip install --python "$VENV/bin/python" \
  mani_skill \
  h5py tqdm wandb tensorboard \
  imageio imageio-ffmpeg opencv-python-headless \
  matplotlib pandas scipy \
  diffusers huggingface_hub \
  tyro "gymnasium>=1.1" \
  ipykernel jupyterlab

# The pip package ships no examples/baselines. Clone for the diffusion_policy
# and ppo baseline code used in T-I and T-IV.
[ -d "$ROOT/ManiSkill" ] || git clone https://github.com/haosulab/ManiSkill.git "$ROOT/ManiSkill"

echo "=============================================================="
echo "STEP 6 — env file"
echo "=============================================================="
cat > "$ROOT/env.sh" <<EOF
# source this in every new shell
export PATH="\$HOME/.local/bin:\$PATH"
source $VENV/bin/activate

export MS_ASSET_DIR=$ROOT/maniskill_data
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
unset DISPLAY                      # headless; SAPIEN renders offscreen

export RIPL_ROOT=$ROOT
export MANISKILL_REPO=$ROOT/ManiSkill

# re-inherit container env (interactive shells often don't)
if [ -r /proc/1/environ ]; then
  _c=\$(tr '\\0' '\\n' < /proc/1/environ | grep '^NVIDIA_DRIVER_CAPABILITIES=' | cut -d= -f2- || true)
  [ -n "\$_c" ] && export NVIDIA_DRIVER_CAPABILITIES="\$_c"
  unset _c
fi
EOF

mkdir -p "$ROOT/maniskill_data"
# shellcheck disable=SC1091
source "$ROOT/env.sh"

echo "=============================================================="
echo "STEP 7 — version report (record these for your README)"
echo "=============================================================="
python - <<'PY'
import torch, mani_skill, sapien
print(f"  torch       {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"  mani_skill  {mani_skill.__version__}")
print(f"  sapien      {sapien.__version__}")
PY
{
  echo "gpu:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "vulkan: $(grep -m1 deviceName "$ROOT/vulkaninfo.txt" | xargs)"
  "$VENV/bin/python" -c "import torch,mani_skill,sapien; print(f'torch: {torch.__version__}'); print(f'mani_skill: {mani_skill.__version__}'); print(f'sapien: {sapien.__version__}')"
} > "$ROOT/ENVIRONMENT.txt"
echo "  saved to $ROOT/ENVIRONMENT.txt"

echo ""
echo "=============================================================="
echo "BUILD COMPLETE. The graphics + python layers are verified."
echo "Nothing about the simulator, data, or training is verified yet."
echo ""
echo "  source $ROOT/env.sh"
echo "  tmux new -s smoke"
echo "  bash smoke_test_e2e.sh"
echo "=============================================================="
