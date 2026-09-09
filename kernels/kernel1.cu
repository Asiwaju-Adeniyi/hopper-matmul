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
        uint64_t desc = 0;
        
        desc |= matrixPropEncoder(addr); //combines the matrix start address to the desciptor 
        desc |= matrixPropEncoder(16) << 16; //The encoded leading-dimension offset is shifted left by 16 so that its descriptor field begins at bit 16
        desc |= matrixPropEncoder(1024) << 32; //combines the stride dimension bytes offset from bit 32
        desc |= 1llu << 62; //sets bits 62 and 63 to 1 and 0 respectively. 

    }
}


__device__ void WGfence() {
    asm volatile ("wgmma.fence.sync.aligned; \n" ::: "memory");
}

__device__ void WGcommit() {
    asm volatile("wgmma.commit_group.sync.aligned; \n" ::: "memory");
}

template <uint N> 
__device__ void WGwait() {
    static_assert(N >= 0 && N <= 7, "Wait instruction: N must be between 0 and 7");
    asm volatile ("wgmma.wait_group.sync.aligned;\n" :: "n"(N) : "memory");
}


template <int outDim, int reducDim> 

void tmaMAP(CutensorMap *tma_map, bf16* gmem_ptr, int height, int width) {
    void* gmem_address = (void*) gmem_ptr; //casting to void because TMA expects a generic pointer
    uint64_t gmem_shape[5] = {(uint64_t) reducDim * width, (uint64_t outDim * height), 1, 1, 1}; //this is the shape of the whole matrix. 
    uint32_t gmem_stride = {sizeof(bf16), sizeof(bf16) * reducDim * width, 0, 0, 0}; /*basic stride formula or (col, row) is that if threadIdx.x increases by 1 in the x-direction, y increases by number of entries in the x-direction * col's stride*/
    uint32_t smem_shape[5] = {(uint32_t)outDim, (uint32_t reducDim)};
    uint32_t smem_stride[5] = {1,1,1,1,1};  

        CUresult output = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_shape,
        gmem_stride + 1, smem_shape, smem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(output == CUDA_SUCCESS);
}


CUtensorMap *d_tmaMap1 = 0;
CUtensorMap *d_tmaMap2 = 0;