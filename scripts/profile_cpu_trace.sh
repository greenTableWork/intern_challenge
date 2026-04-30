#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   Configure and build with frame pointers plus gprof instrumentation:
#     cmake --preset profile -S cpp
#     cmake --build cpp/build_profile --target placement placement_test placement_unit_tests
#
#   Simple C++ CPU profile:
#     scripts/profile_cpu_trace.sh cpp/build_profile/placement --help
#
#   Profile the C++ benchmark/test executable:
#     scripts/profile_cpu_trace.sh cpp/build_profile/placement_test
#
#   Profile C++ unit tests:
#     scripts/profile_cpu_trace.sh cpp/build_profile/placement_unit_tests
#
#   Higher-frequency perf sampling:
#     PERF_FREQ=997 scripts/profile_cpu_trace.sh cpp/build_profile/placement_test
#
#   Hardware-cycle sampling when perf permissions allow it:
#     PERF_EVENTS=cycles:u scripts/profile_cpu_trace.sh cpp/build_profile/placement_test
#
#   Full CPU tracing preset:
#     PERF_FREQ=997 PERF_EVENTS=cycles:u PERF_CALL_GRAPH=dwarf PERF_SCRIPT=1 \
#     scripts/profile_cpu_trace.sh cpp/build_profile/placement_test

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <cpp-command> [args...]" >&2
  echo "example: $0 cpp/build_profile/placement_test" >&2
  exit 2
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "error: perf was not found on PATH" >&2
  exit 1
fi

if ! command -v gprof >/dev/null 2>&1; then
  echo "error: gprof was not found on PATH" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${CPU_PROFILE_OUTPUT_DIR:-profile/cpu}"
output_base="${CPU_PROFILE_OUTPUT_BASE:-${output_dir}/cpu_trace_${timestamp}}"
perf_freq="${PERF_FREQ:-199}"
perf_call_graph="${PERF_CALL_GRAPH:-fp}"
perf_events="${PERF_EVENTS:-cpu-clock}"
perf_report_limit="${PERF_REPORT_LIMIT:-200}"

mkdir -p "${output_dir}"

command_path="$1"
shift
if [[ "${command_path}" == */* ]]; then
  command_path="$(realpath "${command_path}")"
fi

perf_data="${output_base}.perf.data"
perf_report="${output_base}.perf.report.txt"
perf_stat="${output_base}.perf.stat.txt"
perf_script="${output_base}.perf.script.txt"
gmon_prefix="${output_base}.gmon"
gprof_report="${output_base}.gprof.txt"

echo "CPU profile output base: ${output_base}"
echo "perf events: ${perf_events}, frequency=${perf_freq}, call_graph=${perf_call_graph}"
echo "command: ${command_path} $*"
echo

GMON_OUT_PREFIX="${gmon_prefix}" perf record \
  --output="${perf_data}" \
  --freq="${perf_freq}" \
  --event="${perf_events}" \
  --call-graph="${perf_call_graph}" \
  -- "${command_path}" "$@"

if [[ ! -r "${perf_data}" ]]; then
  echo "error: perf data was created but is not readable by ${USER:-the current user}: ${perf_data}" >&2
  echo "fix ownership with:" >&2
  echo "  sudo chown ${USER:-$(id -un)}:$(id -gn) ${perf_data}" >&2
  exit 1
fi

perf report \
  --stdio \
  --input="${perf_data}" \
  --max-stack="${perf_report_limit}" \
  >"${perf_report}"

if [[ "${PERF_STAT:-0}" == "1" ]]; then
  echo
  echo "Running a second pass for perf stat because perf stat and perf record collect separately."
  perf stat \
    --output="${perf_stat}" \
    --event="${PERF_STAT_EVENTS:-task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses}" \
    -- "${command_path}" "$@"
fi

if [[ "${PERF_SCRIPT:-0}" == "1" ]]; then
  perf script --input="${perf_data}" >"${perf_script}"
fi

shopt -s nullglob
gmon_files=("${gmon_prefix}".*)
if [[ ${#gmon_files[@]} -gt 0 ]]; then
  gprof "${command_path}" "${gmon_files[@]}" >"${gprof_report}"
else
  echo "Warning: no gprof output found at ${gmon_prefix}.*" >&2
  echo "Rebuild the C++ profile preset after this change so binaries are compiled and linked with -pg." >&2
fi

echo
echo "perf data: ${perf_data}"
echo "perf report: ${perf_report}"
if [[ -f "${perf_stat}" ]]; then
  echo "perf stat: ${perf_stat}"
fi
if [[ -f "${perf_script}" ]]; then
  echo "perf script: ${perf_script}"
fi
if [[ -f "${gprof_report}" ]]; then
  echo "gprof report: ${gprof_report}"
fi
