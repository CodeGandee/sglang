/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/all.h>

namespace sglang::shadowkv {

inline void check_operation_device(
    const at::Tensor& tensor,
    const char* operation) {
  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, tensor.get_device()));
  const bool supported =
      (properties.major == 8 && properties.minor == 0) ||
      (properties.major == 10 && properties.minor == 0);
  TORCH_CHECK(
      supported,
      operation,
      " requires compute capability 8.0 or 10.0; found ",
      properties.major,
      ".",
      properties.minor);
}

}  // namespace sglang::shadowkv
