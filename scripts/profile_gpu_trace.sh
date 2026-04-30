#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   Simple GPU/CPU timeline:
#     scripts/profile_gpu_trace.sh python placement.py --device cuda --num-epochs 100
#
#   Short benchmark case:
#     scripts/profile_gpu_trace.sh python test.py --device cuda --test-case-id 1 --num-epochs 50
#
#   Add GPU hardware metrics:
#     NSYS_GPU_METRICS=1 scripts/profile_gpu_trace.sh python placement.py --device cuda
#
#   Add syscall and file-access tracing:
#     NSYS_SYSCALL=1 NSYS_OSRT_FILE_ACCESS=1 scripts/profile_gpu_trace.sh python placement.py --device cuda
#
#   Full tracing preset for deeper CPU-side diagnosis:
#     NSYS_GPU_METRICS=1 NSYS_SYSCALL=1 NSYS_OSRT_FILE_ACCESS=1 \
#     NSYS_PYTORCH=autograd-shapes-nvtx \
#     NSYS_CUDA_BACKTRACE=all:0 \
#     NSYS_STATS_REPORTS=nvtx_sum,osrt_sum,cuda_api_sum,cuda_api_trace,cuda_kern_exec_sum,cuda_kern_exec_trace,syscall_sum \
#     scripts/profile_gpu_trace.sh python placement.py --device cuda --num-epochs 100

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "example: $0 python placement.py --device cuda --num-epochs 100" >&2
  exit 2
fi

export PATH="${HOME}/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:${PATH}"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${NSYS_OUTPUT_DIR:-profile/nsys}"
output_base="${NSYS_OUTPUT_BASE:-${output_dir}/gpu_trace_${timestamp}}"
cpu_sample="${NSYS_CPU_SAMPLING:-1}"
backtrace="${NSYS_BACKTRACE:-fp}"
samples_per_backtrace="${NSYS_SAMPLES_PER_BACKTRACE:-1}"
nsys_trace="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn,python-gil}"
python_sampling="${NSYS_PYTHON_SAMPLING:-true}"
python_sampling_frequency="${NSYS_PYTHON_SAMPLING_FREQUENCY:-1000}"
python_backtrace="${NSYS_PYTHON_BACKTRACE:-cuda}"
pytorch_trace="${NSYS_PYTORCH:-autograd-nvtx}"
cuda_backtrace="${NSYS_CUDA_BACKTRACE:-kernel:80000,memory:80000,sync:80000}"
osrt_threshold="${NSYS_OSRT_THRESHOLD:-1000}"
stats_reports="${NSYS_STATS_REPORTS:-nvtx_sum,osrt_sum,cuda_api_sum,cuda_kern_exec_sum,syscall_sum}"

mkdir -p "${output_dir}"

if [[ "${cpu_sample}" == "1" ]]; then
  sample_mode="process-tree"
else
  sample_mode="none"
fi

nsys_args=(
  --force-overwrite=true \
  --output="${output_base}" \
  --trace="${nsys_trace}" \
  --sample="${sample_mode}" \
  --cpuctxsw="${sample_mode}" \
  --backtrace="${backtrace}" \
  --samples-per-backtrace="${samples_per_backtrace}" \
  --python-sampling="${python_sampling}" \
  --python-sampling-frequency="${python_sampling_frequency}" \
  --python-backtrace="${python_backtrace}" \
  --pytorch="${pytorch_trace}" \
  --cudabacktrace="${cuda_backtrace}" \
  --osrt-threshold="${osrt_threshold}" \
  --stats=true \
  --export=sqlite
)

if [[ "${NSYS_GPU_METRICS:-0}" == "1" ]]; then
  nsys_args+=(--gpu-metrics-devices=all)
fi

if [[ "${NSYS_SYSCALL:-0}" == "1" ]]; then
  nsys_args+=(--syscall=process-tree)
fi

if [[ "${NSYS_OSRT_FILE_ACCESS:-0}" == "1" ]]; then
  nsys_args+=(--osrt-file-access=true)
fi

echo "Nsight Systems output base: ${output_base}"
echo "CPU sampling: ${sample_mode}, backtrace=${backtrace}, python_sampling=${python_sampling}"
echo "Trace domains: ${nsys_trace}"
echo

nsys profile "${nsys_args[@]}" "$@"

if [[ "${NSYS_STATS_SUMMARY:-1}" == "1" ]]; then
  summary_base="${output_base}_summary"
  if nsys stats \
    --force-overwrite true \
    --report "${stats_reports}" \
    --format table \
    --output "${summary_base}" \
    "${output_base}.sqlite"; then
    echo "Text summaries: ${summary_base}_*.txt"
  else
    echo "Warning: failed to generate nsys stats summaries." >&2
  fi
fi

echo
echo "Nsight Systems report: ${output_base}.nsys-rep"
echo "SQLite export: ${output_base}.sqlite"
echo "Open GUI with: nsys-ui ${output_base}.nsys-rep"
