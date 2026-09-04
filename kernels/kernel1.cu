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

    __device__ static inline uint64_t matrixPropEncoder(uint64_t x) {return (((x) & 0x3FFFF)) >> 0x4;} /*TTake the lowest 18 bits of x, then discard the lowest 4 of those bits by shifting right 4 positions.*/

    __device__ uint64_t smemDescriptor(uint32_t *ptr) {
        uint32_t addr = static_cast<uint32_t>(cvta_generic_to_shared(ptr));
        uint64_t desc = 0xFFFF;
        
        desc |= matrixPropEncoder(addr); //combines the matrix start address to the desciptor 
        desc |= matrixPropEncoder(16) << 16; //The encoded leading-dimension offset is shifted left by 16 so that its descriptor field begins at bit 16
        desc |= matrixPropEncoder(1024) << 32; //combines the stride dimension bytes offset from bit 32
        desc |= 1llu << 62; //sets bits 62 and 63 to 1 and 0 respectively. 

    }
}