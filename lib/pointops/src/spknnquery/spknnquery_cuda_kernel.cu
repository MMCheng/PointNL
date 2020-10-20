#include "../cuda_utils.h"
#include "spknnquery_cuda_kernel.h"

// input: xyz (b, n, 3) new_xyz (b, m, 3) comp (b, n, 1) // keep n==m
// output: idx (b, m, nsample) dist2 (b, m, nsample)

// modify
__global__ void spknnquery_cuda_kernel(int b, int n, int m, int nsample, const float *__restrict__ xyz, const float *__restrict__ new_xyz, 
                                       const long int *__restrict__ comp, const long int *__restrict__ new_comp,
                                       int *__restrict__ idx, float *__restrict__ dist2) {
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    new_comp += bs_idx * m * 1 + pt_idx * 1;    // add

    xyz += bs_idx * n * 3;
    comp += bs_idx * n * 1;     // add

    idx += bs_idx * m * nsample + pt_idx * nsample;

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    //long int new_c = new_comp[0];    // add

    //double* best = new double[nsample];
    //int* besti = new int[nsample];
    double best[200];
    int besti[200];
    for(int i = 0; i < nsample; i++){
        best[i] = 1e40;
        besti[i] = -1;  // modify: 0 to -1
    }
    for(int k = 0; k < n; k++){
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        //long int comp_i = comp[k * 1 + 0];   // add
        //if (comp_i != new_c) {  // add: belong to different super point
        if (comp[k * 1 + 0] != new_comp[0]) {  // add: belong to different super point
            continue;
        }
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        for(int j = 0; j < nsample; j++){
            if(d2 < best[j]){
                for(int i = nsample - 1; i > j; i--){
                    best[i] = best[i - 1];
                    besti[i] = besti[i - 1];
                }
                best[j] = d2;
                besti[j] = k;
                break;
            }
        }
    }
    int pos = nsample;
    for(int i = 0; i < nsample; i++){
        if (besti[i] == -1) { // add
            pos = i;
            break;
        }
        idx[i] = besti[i];
        dist2[i] = best[i];
    }
    // add--------
    int j = 0;
    for (int i = pos; i < nsample; i++) {  // copy to nsample
        idx[i] = idx[j];
        dist2[i] = dist2[j];
        j += 1;
        if (j >= pos) {
            j = 0;
        }
    }
    for (int i = pos; i < nsample; i++) {  // copy to nsample
        if (idx[i] == -1) {
            idx[i] = 0;
        }
    }
    //delete []best;
    //delete []besti;
}


void spknnquery_cuda_launcher(int b, int n, int m, int nsample, const float *xyz, const float *new_xyz, 
                              const long int *comp, const long int *new_comp,
                              int *idx, float *dist2, cudaStream_t stream) { // modify
    // param new_xyz: (B, m, 3)
    // param xyz: (B, n, 3)
    // param comp: (B, n, 1)            // add
    // param new_comp: (B, m, 1)        // add
    // param idx: (B, m, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);    // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    spknnquery_cuda_kernel<<<blocks, threads, 0, stream>>>(b, n, m, nsample, xyz, new_xyz, comp, new_comp, idx, dist2);     // modify
    // cudaDeviceSynchronize();  // for using printf in kernel function

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
