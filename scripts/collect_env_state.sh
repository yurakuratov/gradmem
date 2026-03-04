#!/bin/bash

COLLECT_ENV_STATE_SUBDIR="${COLLECT_ENV_STATE_SUBDIR:-metadata}"

# Emit metadata lines to stdout for file/stdout reuse.
_emit_meta_lines() {
  local exp_path="${1:-N/A}"
  local np="${2:-N/A}"
  local cpu_model="unknown"
  local cpu_count="unknown"
  local ram_total="unknown"
  local gpu_count="0"
  local gpu_models="none"
  local nvidia_driver_version="unknown"
  local cuda_version="unknown"
  local pytorch_version="unavailable"
  local torch_cuda_version="unavailable"
  local torch_cuda_available="unavailable"
  local torch_cuda_device_count="unavailable"
  local torch_gpu_compute_capability="unavailable"
  local git_commit="unknown"
  local git_branch="unknown"
  local git_upstream="none"
  local git_remote_url="none"
  local git_ahead_count="unknown"
  local git_behind_count="unknown"
  local git_is_pushed="unknown"
  local git_counts=""
  local git_remote_name=""
  local py_exec=""
  local torch_probe_output=""

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
    git_commit="${git_commit:-unknown}"

    git_branch="$(git symbolic-ref --short -q HEAD 2>/dev/null || true)"
    if [[ -z "$git_branch" ]]; then
      git_branch="detached"
    fi

    git_upstream="$(git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>/dev/null || true)"
    git_upstream="${git_upstream:-none}"

    if [[ "$git_upstream" != "none" ]]; then
      git_counts="$(git rev-list --left-right --count "HEAD...$git_upstream" 2>/dev/null || true)"
      git_ahead_count="$(awk '{print $1}' <<< "$git_counts" 2>/dev/null || true)"
      git_behind_count="$(awk '{print $2}' <<< "$git_counts" 2>/dev/null || true)"
      git_ahead_count="${git_ahead_count:-unknown}"
      git_behind_count="${git_behind_count:-unknown}"

      if [[ "$git_ahead_count" =~ ^[0-9]+$ ]]; then
        if [[ "$git_ahead_count" -eq 0 ]]; then
          git_is_pushed="yes"
        else
          git_is_pushed="no"
        fi
      fi

      git_remote_name="${git_upstream%%/*}"
      if [[ -n "$git_remote_name" && "$git_remote_name" != "$git_upstream" ]]; then
        git_remote_url="$(git remote get-url "$git_remote_name" 2>/dev/null || true)"
        git_remote_url="${git_remote_url:-none}"
      fi
    fi
  fi

  if command -v lscpu >/dev/null 2>&1; then
    cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name:/{gsub(/^[ \t]+/, "", $2); print $2; exit}' || true)"
    cpu_count="$(lscpu 2>/dev/null | awk -F: '/^CPU\(s\):/{gsub(/^[ \t]+/, "", $2); print $2; exit}' || true)"
  fi
  if [[ -z "$cpu_model" || "$cpu_model" == "unknown" ]]; then
    cpu_model="$(awk -F: '/model name/{gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null || true)"
    cpu_model="${cpu_model:-unknown}"
  fi
  if [[ -z "$cpu_count" || "$cpu_count" == "unknown" ]]; then
    cpu_count="$(awk '/^processor/{count+=1} END{if (count > 0) print count}' /proc/cpuinfo 2>/dev/null || true)"
    cpu_count="${cpu_count:-unknown}"
  fi

  if command -v free >/dev/null 2>&1; then
    ram_total="$(free -h 2>/dev/null | awk '/^Mem:/{print $2; exit}' || true)"
    ram_total="${ram_total:-unknown}"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NF{count+=1} END{print count+0}' || true)"
    gpu_models="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | awk 'NF{models = models ? models "; " $0 : $0} END{if (models) print models; else print "none"}' || true)"
    nvidia_driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | awk 'NF{print; exit}' || true)"
    cuda_version="$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' 'NF > 1 {split($2, a, " "); print a[1]; exit}' || true)"
    nvidia_driver_version="${nvidia_driver_version:-unknown}"
    cuda_version="${cuda_version:-unknown}"
  fi

  if command -v python >/dev/null 2>&1; then
    py_exec="python"
  elif command -v python3 >/dev/null 2>&1; then
    py_exec="python3"
  fi

  if [[ -n "$py_exec" ]]; then
    torch_probe_output="$("$py_exec" - <<'PY'
try:
    import torch
    print(f"pytorch_version={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        cap = torch.cuda.get_device_capability(0)
        print(f"torch_gpu_compute_capability={cap[0]}.{cap[1]}")
    else:
        print("torch_gpu_compute_capability=unavailable")
except Exception:
    print("pytorch_version=unavailable")
    print("torch_cuda_version=unavailable")
    print("torch_cuda_available=unavailable")
    print("torch_cuda_device_count=unavailable")
    print("torch_gpu_compute_capability=unavailable")
PY
)" || true

    while IFS='=' read -r key value; do
      case "$key" in
        pytorch_version) pytorch_version="${value:-unavailable}" ;;
        torch_cuda_version) torch_cuda_version="${value:-unavailable}" ;;
        torch_cuda_available) torch_cuda_available="${value:-unavailable}" ;;
        torch_cuda_device_count) torch_cuda_device_count="${value:-unavailable}" ;;
        torch_gpu_compute_capability) torch_gpu_compute_capability="${value:-unavailable}" ;;
      esac
    done <<< "$torch_probe_output"
  fi

  {
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "pwd: $(pwd)"
    echo "user: $(whoami)"
    echo "exp_path: $exp_path"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
    echo "NP=${np}"
    echo "git_commit: $git_commit"
    echo "git_branch: $git_branch"
    echo "git_upstream: $git_upstream"
    echo "git_remote_url: $git_remote_url"
    echo "git_ahead_count: $git_ahead_count"
    echo "git_behind_count: $git_behind_count"
    echo "git_is_pushed: $git_is_pushed"
    echo "cpu_model: $cpu_model"
    echo "cpu_count: $cpu_count"
    echo "ram_total: $ram_total"
    echo "gpu_count: $gpu_count"
    echo "gpu_models: $gpu_models"
    echo "nvidia_driver_version: $nvidia_driver_version"
    echo "cuda_version: $cuda_version"
    echo "pytorch_version: $pytorch_version"
    echo "torch_cuda_version: $torch_cuda_version"
    echo "torch_cuda_available: $torch_cuda_available"
    echo "torch_cuda_device_count: $torch_cuda_device_count"
    echo "torch_gpu_compute_capability: $torch_gpu_compute_capability"
  }
}

# Collect reproducibility metadata for an experiment run.
# Usage: collect_env_state <lock_dir> <launch_script_path> <exp_path> <np>
collect_env_state() {
  local lock_dir="${1:?lock_dir is required}"
  local launch_script_path="${2:?launch_script_path is required}"
  local exp_path="${3:?exp_path is required}"
  local np="${4:?np is required}"
  local repro_dir="$lock_dir/$COLLECT_ENV_STATE_SUBDIR"

  mkdir -p "$repro_dir" || true

  cp -a "$launch_script_path" "$repro_dir/launch_script.sh" 2>/dev/null || true
  (env | sort) > "$repro_dir/env.txt" || true

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git status --porcelain=v1 > "$repro_dir/git_status.txt" 2>/dev/null || true
    git diff > "$repro_dir/git.diff" 2>/dev/null || true
  fi

  _emit_meta_lines "$exp_path" "$np" > "$repro_dir/meta.txt"
}

# Prepare lock-dir based run state and collect metadata.
# Usage: prepare_locked_run <exp_path> <launch_script_path> <np>
prepare_locked_run() {
  local exp_path="${1:?exp_path is required}"
  local launch_script_path="${2:?launch_script_path is required}"
  local np="${3:?np is required}"
  local lock_dir="${exp_path}.lock"

  mkdir -p "$(dirname "$exp_path")"

  # If already completed / running / created, skip.
  if [[ -d "$exp_path" ]]; then
    echo "[SKIP] exists: $exp_path"
    return 1
  fi

  # Atomically claim this run slot (lock).
  if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "[SKIP] locked: $exp_path"
    return 1
  fi

  RUN_LOCK_DIR="$lock_dir"
  RUN_LOCK_METADATA_DIR="$RUN_LOCK_DIR/$COLLECT_ENV_STATE_SUBDIR"
  RUN_EXP_METADATA_DIR="$exp_path/$COLLECT_ENV_STATE_SUBDIR"
  RUN_LOCK_LOG="$RUN_LOCK_DIR/stdout_stderr.log"

  collect_env_state "$RUN_LOCK_DIR" "$launch_script_path" "$exp_path" "$np"
}

# Print standardized run header and command line.
# Usage: print_run_header <exp_path> <port> <np> <mixed_precision> <cmd...>
print_run_header() {
  local exp_path="${1:?exp_path is required}"
  local port="${2:?port is required}"
  local np="${3:?np is required}"
  local mixed_precision="${4:?mixed_precision is required}"
  shift 4

  echo "[RUN] $exp_path"
  echo "[RUN] port=$port CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-} NP=$np mixed_precision=$mixed_precision"
  printf '[RUN] cmd: '
  printf '%q ' "$@"
  echo
}

# Finalize lock-dir artifacts and lock retention/cleanup.
# Usage: finalize_locked_run <exp_path> <lock_dir> <lock_log> <exit_code>
finalize_locked_run() {
  local exp_path="${1:?exp_path is required}"
  local lock_dir="${2:?lock_dir is required}"
  local lock_log="${3:?lock_log is required}"
  local exit_code="${4:?exit_code is required}"
  local lock_metadata_dir="$lock_dir/$COLLECT_ENV_STATE_SUBDIR"
  local exp_metadata_dir="$exp_path/$COLLECT_ENV_STATE_SUBDIR"

  finish_run_timer "$exit_code" "$lock_dir"

  echo "[RUN] duration=${RUN_TIMER_DURATION_DHMS} (d:hh:mm:ss) exit_code=$exit_code"

  # After python creates EXP_PATH, copy metadata + logs into it.
  if [[ -d "$exp_path" ]]; then
    mkdir -p "$exp_metadata_dir"
    cp -a "$lock_metadata_dir/." "$exp_metadata_dir/" 2>/dev/null || true

    # Merge logs: keep lock log as canonical, also copy to EXP_PATH/train.log.
    cp -a "$lock_log" "$exp_path/train.log" 2>/dev/null || true
  fi

  if [[ "$exit_code" -eq 0 ]]; then
    rm -rf "$lock_dir"
  else
    # Keep lockdir, indicating that run did not complete successfully.
    echo "[FAIL] exit_code=$exit_code (leaving lock dir for inspection): $lock_dir"
  fi
}

# Start per-run wall-clock timer.
start_run_timer() {
  RUN_TIMER_START_EPOCH="$(date +%s)"
  RUN_TIMER_START_ISO="$(date -Is)"
}

# Finish timer, write timing artifact, and expose formatted duration.
# Usage: finish_run_timer <exit_code> <lock_dir>
finish_run_timer() {
  local exit_code="${1:?exit_code is required}"
  local lock_dir="${2:?lock_dir is required}"
  local metadata_dir="$lock_dir/$COLLECT_ENV_STATE_SUBDIR"
  local end_epoch
  local end_iso
  local duration_seconds
  local duration_days
  local duration_hours
  local duration_minutes
  local duration_secs
  local duration_dhms

  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  duration_seconds=$((end_epoch - RUN_TIMER_START_EPOCH))
  duration_days=$((duration_seconds / 86400))
  duration_hours=$(((duration_seconds % 86400) / 3600))
  duration_minutes=$(((duration_seconds % 3600) / 60))
  duration_secs=$((duration_seconds % 60))
  duration_dhms="$(printf "%d:%02d:%02d:%02d" \
    "$duration_days" "$duration_hours" "$duration_minutes" "$duration_secs")"

  mkdir -p "$metadata_dir" || true
  {
    echo "run_start: $RUN_TIMER_START_ISO"
    echo "run_end: $end_iso"
    echo "run_duration_dhms: $duration_dhms"
    echo "exit_code: $exit_code"
  } > "$metadata_dir/run_timing.txt" || true

  RUN_TIMER_END_ISO="$end_iso"
  RUN_TIMER_DURATION_DHMS="$duration_dhms"
}

find_free_port() {
python - <<'PY'
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
PY
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  if [[ $# -eq 0 ]]; then
    _emit_meta_lines "${EXP_PATH:-N/A}" "${NP:-N/A}"
  else
    echo "Usage: bash scripts/collect_env_state.sh"
    echo "Runs a standalone env snapshot to stdout."
    exit 1
  fi
fi
