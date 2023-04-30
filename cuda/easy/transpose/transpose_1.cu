#include <iostream>

#define M 1000
#define N 1000

using namespace std;

static const bool align = true;
#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;                   \
       i += blockDim.x * gridDim.x)


__global__ void mat_transpose_1_1 (float *c, float *a, int m, int n) {
    CUDA_1D_KERNEL_LOOP(i, n*m) {
        int col = i % m;
	int row = i / m;

	c[col*m+row] = a[row*n+col];
    }
}


__global__ void mat_transpose_1_2 (float *c, float *a, int m, int n) {
    CUDA_1D_KERNEL_LOOP(i, n*m) {
        int col = i % m;
        int row = i / m;

        c[row*m+col] = a[col*n+row];
    }
}


int main() {
    float *h_a, *h_o;
    float *d_a, *d_o;

    // define timer
    cudaEvent_t start, stop;

    // init data
    h_a = (float*) malloc(sizeof(float) * M * N);
    h_o = (float*) malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
	    h_a[idx] = idx % 10 + 3.3233f;
	}
    }

    size_t p_a, p_o;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // GPU memory allco and copy data
    if (align) {
	cudaMallocPitch((void**) &d_a, &p_a, sizeof(float) * N, M);
	cudaMallocPitch((void**) &d_o, &p_o, sizeof(float) * M, N);

	cudaMemcpy2D(d_a, p_a, h_a, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_transpose_1_2<<<512, 512>>>(d_o, d_a, p_a / sizeof(float), N);
    } else {
        cudaMalloc((void**) &d_a, sizeof(float) * (M * N)); 
        cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

        cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_transpose_1_1<<<512, 512>>>(d_o, d_a, M, N);
    }
 
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    if (!align) cudaMemcpy(h_o, d_o, sizeof(float) * (M * N), cudaMemcpyDeviceToHost);
    else cudaMemcpy2D(h_o, sizeof(float) * N, d_o, p_o, sizeof(float) * N, M, cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "The kernel function is during: " << elapsedTime << " ms" << endl;
    cout << "The final number (M * N): " << h_o[N*M-2] << endl;
    cout << "input " << h_a[(M-1)*N-1] << endl;

    cudaFree(d_a);
    cudaFree(d_o);
    free(h_a);
    free(h_o);
    cudaDeviceReset();
    return 0;
}
