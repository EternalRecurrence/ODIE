cmake_minimum_required(VERSION 3.24)
project(odie_runtime_probe LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
foreach(lib IN ITEMS tiny-cuda-nn tiny-cuda-nn-resources)
  add_library(${lib} STATIC IMPORTED)
  set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${TCNN_PREBUILT_DIR}/lib${lib}.a")
endforeach()
add_library(tcnn-fmt STATIC IMPORTED)
set_target_properties(tcnn-fmt PROPERTIES IMPORTED_LOCATION "${TCNN_PREBUILT_DIR}/dependencies/fmt/libfmt.a")
add_library(odie_runtime_probe SHARED runtime.cu)
target_compile_definitions(odie_runtime_probe PRIVATE ODIE_LEVELS=${ODIE_LEVELS} ODIE_RAW_WIDTH=${ODIE_RAW_WIDTH} ODIE_RGB8=${ODIE_RGB8} ODIE_GATE_K=${ODIE_GATE_K} ODIE_SOFTPLUS=${ODIE_SOFTPLUS} ODIE_ROUTER_ENABLED=${ODIE_ROUTER_ENABLED})
target_include_directories(odie_runtime_probe PRIVATE
  "${TCNN_SOURCE_DIR}/include" "${TCNN_SOURCE_DIR}/dependencies"
  "${TCNN_SOURCE_DIR}/dependencies/cutlass/include"
  "${TCNN_SOURCE_DIR}/dependencies/cutlass/tools/util/include"
  "${TCNN_SOURCE_DIR}/dependencies/fmt/include")
target_compile_definitions(odie_runtime_probe PRIVATE TCNN_MIN_GPU_ARCH=${TCNN_MIN_GPU_ARCH} TCNN_HALF_PRECISION=1 TCNN_RTC TCNN_RTC_USE_FAST_MATH TCNN_CMRC FMT_UNICODE=0)
target_compile_options(odie_runtime_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;--expt-extended-lambda;-lineinfo>)
target_link_libraries(odie_runtime_probe PRIVATE tiny-cuda-nn tiny-cuda-nn-resources tcnn-fmt CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::cublas CUDA::cublasLt)
