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

#define barrier cuda::barrier<cuda::thread_scope_block>

namespace M1 {
    namespace cde = cuda::device::experimental;

    __device__ static inline uint64_t matrixPropEncoder(uint64_t x) {return (((x) & 0x3FFFF)) >> 0x4;} /*TTake the lowest 18 bits of x, then discard the lowest 4 of those bits by shifting right 4 positions.*/

    __device__ uint64_t smemDescriptor(bf16 *ptr) {
        uint32_t addr = static_cast<uint32_t>(cvta_generic_to_shared(ptr));
        uint64_t desc = 0;
        
        desc |= matrixPropEncoder(addr); //combines the matrix start address to the desciptor 
        desc |= matrixPropEncoder(16) << 16; //The encoded leading-dimension offset is shifted left by 16 so that its descriptor field begins at bit 16
        desc |= matrixPropEncoder(1024) << 32; //combines the stride dimension bytes offset from bit 32
        desc |= 1llu << 62; //sets bits 62 and 63 to 1 and 0 respectively. 

        return desc;

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
    uint64_t gmem_shape[5] = {(uint64_t) reducDim * width, (uint64_t) outDim * height, 1, 1, 1}; //this is the shape of the whole matrix. 
    uint32_t gmem_stride[5] = {sizeof(bf16), sizeof(bf16) * reducDim * width, 0, 0, 0}; /*basic stride formula or (col, row) is that if threadIdx.x increases by 1 in the x-direction, y increases by number of entries in the x-direction * col's stride*/
    uint32_t smem_shape[5] = {(uint32_t)outDim, (uint32_t) reducDim};
    uint32_t smem_stride[5] = {1,1,1,1,1};  

        CUresult output = cuTensorMapEncodeTiled(
        tma_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gmem_address, gmem_shape,
        gmem_stride + 1, smem_shape, smem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    assert(output == CUDA_SUCCESS);
}


CUtensorMap *d_tmaMap1 = 0;
CUtensorMap *d_tmaMap2 = 0;

int _prev_m=0, _prev_n=0, _prev_k=0;

template <int outDim, int reducDim> 

__host__ static inline CUtensormap* allocateAndCreateTensorMap(bf16 *src, int height, int width) {
    CUtensorMap *tmaMapD; 
    CUtensorMap tmaMapH;

    cudaMalloc(&tmaMapD, sizeof(CUtensorMap));
    tmaMAP<outDim, reducDim>(&tmaMapH, src, blocks_height, blocks_width);
    cudaMemcpy(tmaMapD, &tmaMapHost, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    return tmaMapD;
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma64(float d[4][8], bf16* sA, bf16* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0]);
    uint64_t desc_b = make_smem_desc(&sB[0]);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31},"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
        "}\n"
        : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]),
          "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]),
          "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]),
          "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]),
          "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]),
          "+f"(d[3][6]), "+f"(d[3][7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}




