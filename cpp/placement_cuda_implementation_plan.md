# Plan: Full CUDA Placement Implementation

## Summary
- `placement_cuda` is the active path for moving placement generation to CUDA-first code.
- `placement_pytorch_cuda` remains the TorchScript/JIT experiment and historical comparison point.
- Keep the default example fixed at `3` macros, `10` standard cells, seed `42`, and no CLI arguments.

## Current CUDA Progress
- CUDA tensor setup exists in `placement_cuda_tensor_setup.{h,cu}`.
- The CUDA setup path allocates tensors in C++ and fills them in a cuRAND-backed CUDA kernel.
- Current CUDA-filled tensors: `macro_areas`, `std_area_indices`, `std_cell_areas`, `areas`, `num_pins_per_cell`, `cell_widths`, `cell_heights`, and initial `cell_features`.
- CUDA now initializes cell `X`/`Y` positions with a cuRAND-backed kernel after setup, using a CUDA-side area reduction for the spread radius.
- `placement_cuda::CellFeatureIdx` names the cell feature columns used by the CUDA kernel.

## Migration Milestones
- [x] Create the initial CUDA tensor setup path for areas, dimensions, and initial `cell_features`.
- [x] Generate macro and standard-cell pin counts in CUDA and write `NumPins` directly into `cell_features`.
- [x] Add debug-mode rendering for the CUDA-initialized placement state.
- [ ] Compute total pin counts and offsets with CUDA-friendly prefix-sum primitives.
- [ ] Allocate and fill `pin_features` in CUDA.
- [ ] Generate `edge_list` in CUDA, initially accepting a simple bounded edge-count strategy before optimizing duplicate handling.
- [x] Initialize cell positions in CUDA without host scalar extraction.
- [x] Add rendering capabilities to the CUDA path for generated placement states.
- [ ] Remove remaining `.item<>`, host-loop, `std::vector`, and `from_blob` sync points from the CUDA path.

## CUDA Training Milestones
- [ ] Add CUDA implementation of the wirelength attraction loss.
- [ ] Add CUDA implementation of the overlap repulsion loss.
- [ ] Add a CUDA-resident gradient tracking state for trainable cell positions.
- [ ] Add forward-pass kernels that evaluate losses from CUDA placement tensors.
- [ ] Add backward-pass kernels that accumulate gradients into the CUDA gradient state.
- [ ] Add an on-device optimizer step for updating cell positions without host synchronization.
- [ ] Add a training epoch loop that runs forward, backward, optimizer, and gradient reset on device.
- [ ] Add debug/profiling checks for per-epoch loss values and position updates.
- [ ] Render the CUDA placement after training epochs once the on-device training loop is active.

## Build And Profiling Flags
- Add a `placement_cuda` profiling toggle in CMake, defaulting on for Debug-style builds and off for Release unless explicitly enabled.
- Use explicit compile definitions for implementation checks:
  - `PLACEMENT_CUDA_DEBUG_BUILD=1` for `$<CONFIG:Debug>`.
  - `PLACEMENT_CUDA_RELEASE_BUILD=1` for `$<CONFIG:Release>`.
  - `PLACEMENT_CUDA_ENABLE_PROFILING=1` when the profiling toggle is enabled.
  - `PLACEMENT_CUDA_ENABLE_DEBUG_RENDER=1` for debug-mode cell rendering.
- Use existing standard `NDEBUG` semantics only as a secondary signal; CUDA path code should prefer the explicit `PLACEMENT_CUDA_*` macros above.

## Test Plan
- Build both paths:
  - `cmake --build cpp/build_debug --target placement_pytorch_cuda`
  - `cmake --build cpp/build_debug --target placement_cuda`
- Run both paths:
  - `env cpp/build_debug/placement_pytorch_cuda`
  - `env cpp/build_debug/placement_cuda`
- Acceptance checks:
  - `placement_pytorch_cuda` continues to run the JIT setup path.
  - `placement_cuda` has no TorchScript include or JIT call.
  - CUDA tensors remain CUDA-resident and preserve expected shapes and dtypes.
  - Debug builds of `placement_cuda` render `placement_cuda_tensor_setup_debug.png` from the CUDA-initialized positions as the initial placement.
  - CUDA training keeps loss tensors, gradients, optimizer state, and position updates on device.
  - Training renders distinguish initial CUDA placement from post-training CUDA placement once epochs are implemented.

## Assumptions
- `placement_cuda` is allowed to diverge implementation details from the JIT prototype while preserving the default tensor schema.
- The CUDA path should favor explicit CUDA kernels and CUDA library primitives over TorchScript.
- The old JIT plan remains as historical context; this file is the active CUDA progress tracker.
