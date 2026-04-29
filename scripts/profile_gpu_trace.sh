#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "example: $0 python test.py" >&2
  exit 2
fi

export PATH="${HOME}/.local/bin:/usr/local/cuda-13.2/bin:/usr/local/bin:${PATH}"

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${NSYS_OUTPUT_DIR:-profile/nsys}"
output_base="${NSYS_OUTPUT_BASE:-${output_dir}/gpu_trace_${timestamp}}"
cpu_sample="${NSYS_CPU_SAMPLING:-0}"

mkdir -p "${output_dir}"

if [[ "${cpu_sample}" == "1" ]]; then
  sample_mode="process-tree"
else
  sample_mode="none"
fi

nsys_args=(
  --force-overwrite=true \
  --output="${output_base}" \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample="${sample_mode}" \
  --cpuctxsw="${sample_mode}" \
  --stats=true \
  --export=sqlite
)

if [[ "${NSYS_GPU_METRICS:-0}" == "1" ]]; then
  nsys_args+=(--gpu-metrics-devices=all)
fi

nsys profile "${nsys_args[@]}" "$@"

echo
echo "Nsight Systems report: ${output_base}.nsys-rep"
echo "SQLite export: ${output_base}.sqlite"
echo "Open GUI with: nsys-ui ${output_base}.nsys-rep"
