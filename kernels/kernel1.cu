#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda_bf16.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda/barrier>

#define barrier  cuda::barrier<cuda::thread_scope_block>

namespace M1 {
    namespace cde = cuda::device::experimental;

    __device__ static inline uint64_t matrixPropEncoder(uint64_t x) {return (((x) & 0x3FFFF)) >> 0x4;}
}