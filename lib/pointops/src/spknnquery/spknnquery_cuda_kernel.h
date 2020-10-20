#ifndef _SPKNNQUERY_CUDA_KERNEL
#define _SPKNNQUERY_CUDA_KERNEL

#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

void spknnquery_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor comp_tensor, at::Tensor new_comp_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor);  // modify

#ifdef __cplusplus
extern "C" {
#endif

void spknnquery_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, const long int *comp, const long int *new_comp, int *idx, float *dist2, cudaStream_t stream); // modify

#ifdef __cplusplus
}
#endif

#endif
