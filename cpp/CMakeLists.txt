cmake_minimum_required(VERSION 3.25)

project(placement_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# LibTorch is provided by the active Python environment rather than vcpkg.
# Prefer the repo-local virtualenv when present, then ask Python where the
# installed torch package exposes its CMake config files.
set(_repo_venv_python "${CMAKE_CURRENT_LIST_DIR}/../.venv/bin/python")
if(NOT Python3_EXECUTABLE AND EXISTS "${_repo_venv_python}")
  set(Python3_EXECUTABLE "${_repo_venv_python}" CACHE FILEPATH "Python interpreter")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_CMAKE_PREFIX_RESULT
)
if(NOT TORCH_CMAKE_PREFIX_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to import torch from ${Python3_EXECUTABLE}. Install torch in .venv or set Python3_EXECUTABLE.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)

add_library(
  placement_core
  benchmark.cpp
  generation.cpp
  losses.cpp
  metrics.cpp
  training.cpp
)
target_include_directories(placement_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include")
target_link_libraries(placement_core PUBLIC "${TORCH_LIBRARIES}")
target_compile_features(placement_core PUBLIC cxx_std_20)

add_executable(placement_smoke main.cpp)
target_link_libraries(placement_smoke PRIVATE placement_core CLI11::CLI11)
target_compile_features(placement_smoke PRIVATE cxx_std_20)

if(MSVC)
  target_compile_options(placement_core PRIVATE /W4)
  target_compile_options(placement_smoke PRIVATE /W4)
else()
  target_compile_options(placement_core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(placement_smoke PRIVATE -Wall -Wextra -Wpedantic)
endif()
