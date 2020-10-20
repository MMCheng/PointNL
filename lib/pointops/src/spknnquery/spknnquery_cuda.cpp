#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

#include "spknnquery_cuda_kernel.h" // modify

extern THCState *state;

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void spknnquery_cuda(int b, int n, int m, int nsample, at::Tensor xyz_tensor, at::Tensor new_xyz_tensor, at::Tensor comp_tensor, at::Tensor new_comp_tensor, at::Tensor idx_tensor, at::Tensor dist2_tensor) // modify
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(comp_tensor);       // add
    CHECK_INPUT(new_comp_tensor);   // add

    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    const long int *comp = comp_tensor.data<long int>();          // add
    const long int *new_comp = new_comp_tensor.data<long int>();  // add
    int *idx = idx_tensor.data<int>();
    float *dist2 = dist2_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);

    spknnquery_cuda_launcher(b, n, m, nsample, xyz, new_xyz, comp, new_comp, idx, dist2, stream); // modify
}
