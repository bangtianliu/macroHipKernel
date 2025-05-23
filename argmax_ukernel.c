// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <float.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __device__ __attribute__((const)) half __ockl_wfred_max_f16(half);
extern "C" __device__ __attribute__((const))
int64_t __ockl_wfred_min_i64(int64_t);
extern "C" __device__ __attribute__((const))
int32_t __ockl_wfred_min_i32(int32_t);

/*
Constraint/Tiling note:
For simplicity, we distribute all parallel dim across different workgroup, and
only use single subgroup/warp per workgroup. This constraint is also set during
tiling phase in KernelConfig.
*/

// extern "C" __global__ void argmax_F32I64(const float *__restrict__
// inputBuffer,
//                                          int64_t *__restrict__ outputBuffer,
//                                          int reductionSize) {
//   uint laneID = __builtin_amdgcn_workitem_id_x();
//   // Set identity value to handle problem non divisible by subgroupSize.
//   float laneMax = laneID >= reductionSize ? -FLT_MAX : inputBuffer[laneID];
//   int64_t laneResult = laneID;

//   uint numBatches = (reductionSize + warpSize - 1) / warpSize;
//   for (int i = 1; i < numBatches; ++i) {
//     uint idx = warpSize * i + laneID;
//     float newIn = idx >= reductionSize ? -FLT_MAX : inputBuffer[idx];
//     if (newIn == laneMax)
//       continue;
//     laneMax = __ocml_fmax_f32(newIn, laneMax);
//     laneResult = newIn == laneMax ? idx : laneResult;
//   }
//   // Final reduction with one subgroup
//   float wgMax = laneMax;
//   for (int i = 1; i < warpSize; i *= 2) {
//     wgMax = __ocml_fmax_f32(__shfl_xor(wgMax, i), wgMax);
//   }
//   // Check if there are multiple max value holders.
//   uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);
//   // if there is only one max value holder, write and exit.
//   if (__popcll(laneHasMaxValmask) == 1) {
//     if (wgMax == laneMax)
//       outputBuffer[0] = laneResult;
//     return;
//   }
//   // if there are multiple max value holder, find smallest index (argmax
//   // semantics).
//   int64_t indexVal =
//       wgMax == laneMax ? laneResult : std::numeric_limits<int64_t>::max();
//   for (int i = 1; i < warpSize; i *= 2) {
//     indexVal = min(__shfl_xor(indexVal, i), indexVal);
//   }
//   if (laneID == 0)
//     outputBuffer[0] = indexVal;
// }

extern "C" __global__ void
argmax_BF16_I64(const hip_bfloat16 *__restrict__ inputBuffer,
                int64_t *__restrict__ outputBuffer, int reductionSize) {
  uint laneID = __builtin_amdgcn_workitem_id_x();
  float laneMax =
      laneID >= reductionSize ? -FLT_MAX : float(inputBuffer[laneID]);
  int64_t laneResult = laneID;

  uint numBatches = (reductionSize + warpSize - 1) / warpSize;
  for (int i = 1; i < numBatches; ++i) {
    uint idx = i * warpSize + laneID;
    float newIn = idx >= reductionSize ? -FLT_MAX : float(inputBuffer[idx]);
    if (newIn == laneMax)
      continue;
    laneMax = __ocml_fmax_f32(newIn, laneMax);
    laneResult = newIn == laneMax ? idx : laneResult;
  }

  // Final reduction with one subgroup
  float wgMax = laneMax;
  for (int i = 1; i < warpSize; i *= 2) {
    wgMax = __ocml_fmax_f32(__shfl_xor(wgMax, i), wgMax);
  }
  // Check if there are multiple max value holders.
  uint64_t laneHasMaxValmask = __ballot(wgMax == laneMax);
  // if there is only one max value holder, write and exit.
  if (__popcll(laneHasMaxValmask) == 1) {
    if (wgMax == laneMax)
      outputBuffer[0] = laneResult;
    return;
  }
  // if there are multiple max value holder, find smallest index (argmax
  // semantics).
  int64_t indexVal =
      wgMax == laneMax ? laneResult : std::numeric_limits<int64_t>::max();
  for (int i = 1; i < warpSize; i *= 2) {
    indexVal = min(__shfl_xor(indexVal, i), indexVal);
  }
  if (laneID == 0)
    outputBuffer[0] = indexVal;
}