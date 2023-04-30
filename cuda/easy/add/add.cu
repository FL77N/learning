#include <iostream>
#include <cuda_runtime.h>

#define M 2000
#define N 1000

#define CHECK(call)\
{\
    const cudaError_t error=call;\
    if(error!=cudaSuccess)\
    {\
        printf("ERROR: %s:%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
        exit(1);\
    }\
}

using namespace std;

__global__ void m_add(float *c, float *a, float *b, int m, int n){
    // get ind
    const int block_ind = blockIdx.x;
    const int thread_ind = threadIdx.x;
    const int step = blockDim.x * gridDim.x;
    int curr_ind = block_ind * blockDim.x + thread_ind;

    while (curr_ind < m * n) {
        c[curr_ind] = a[curr_ind] + b[curr_ind];
	curr_ind += step;
    }
}

int main() {
    float *h_a, *h_b, *h_o;
    float *d_a, *d_b, *d_o;

    // define timer
    cudaEvent_t start, stop;

    // initialize data
    h_a = (float*) malloc(sizeof(float) * (M * N));
    h_b = (float*) malloc(sizeof(float) * (M * N));
    h_o = (float*) malloc(sizeof(float) * (M * N));

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int ind = i * N + j;
	    h_a[ind] = ind + 324.4553f;
	    h_b[ind] = ind + 2.3232f;
	}
    }

    // GPU memory allco
    CHECK(cudaMalloc((void**) &d_a, sizeof(float) * (M * N)));
    CHECK(cudaMalloc((void**) &d_b, sizeof(float) * (M * N)));
    CHECK(cudaMalloc((void**) &d_o, sizeof(float) * (M * N)));

    // copy from CPU to GPU
    CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * (M * N), cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // cuda add
    m_add<<<M, N>>>(d_o, d_a, d_b, M, N);
    CHECK(cudaDeviceSynchronize());
    // copy results back to CPU
    CHECK(cudaMemcpy(h_o, d_o, sizeof(float) * (M * N), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "The kernel function is during: " << elapsedTime << endl;

    cout << "The final number (M * N): " << h_o[234];
    // free GPU mem
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    free(h_a);
    free(h_b);
    cudaDeviceReset();
    return 0;

}
