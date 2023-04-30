#include <iostream>

#define M 1000
#define N 1000
#define L 8000

using namespace std;

static const bool align = false;


__global__ void mat_mul (float *c, size_t n_c, const float *a, size_t n_a, const float *b, size_t n_b, int n) {
    extern __shared__ float data[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    int i, j;
    for (i = tid; i < n; i += blockDim.x) {
        data[i] = a[row * n_a + i];
    }
    __syncthreads();
    double tmp = 0.0;
    for (j = tid; j < n; j += blockDim.x) {
        tmp = 0.0;
	for (i = 0; i < n; ++i) {
            tmp += data[i] * b[i * n_b + j];
	}
	c[row * n_c + j] = tmp;
    }
}


int main() {
    float *h_a, *h_b, *h_o;
    float *d_a, *d_b, *d_o;

    // define timer
    cudaEvent_t start, stop;

    // init data
    h_a = (float*) malloc(sizeof(float) * M * N);
    h_b = (float*) malloc(sizeof(float) * M * N);
    h_o = (float*) malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
	    h_a[idx] = idx % 10 + 3.3233f;
	    h_b[idx] = idx % 5 + 2.32f;
	}
    }

    size_t p_a, p_b, p_o;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // GPU memory allco and copy data
    if (align) {
	cudaMallocPitch((void**) &d_a, &p_a, sizeof(float) * M, N);
	cudaMallocPitch((void**) &d_b, &p_b, sizeof(float) * M, N);
	cudaMallocPitch((void**) &d_o, &p_o, sizeof(float) * M, N);

	cudaMemcpy2D(d_a, p_a, h_a, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_b, p_b, h_b, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_mul<<<M, N, sizeof(float) * N>>>(d_o, p_o / sizeof(float), d_a, p_a / sizeof(float), d_b, p_b / sizeof(float), N);
    } else {
        cudaMalloc((void**) &d_a, sizeof(float) * (M * N)); 
        cudaMalloc((void**) &d_b, sizeof(float) * (N * N));
        cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

        cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_mul<<<M, N, sizeof(float) * N>>>(d_o, M, d_a, M, d_b, M, N);
    }
 
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    if (!align) cudaMemcpy(h_o, d_o, sizeof(float) * (M * N), cudaMemcpyDeviceToHost);
    else cudaMemcpy2D(h_o, sizeof(float) * N, d_o, p_o, sizeof(float) * N, M, cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "The kernel function is during: " << elapsedTime << " ms" << endl;
    cout << "The final number (M * N): " << h_o[0] << endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    free(h_a);
    free(h_b);
    free(h_o);
    cudaDeviceReset();
    return 0;
}
