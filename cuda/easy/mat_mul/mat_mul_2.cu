#include <iostream>

#define M 2048
#define N 2048
#define L 2048

using namespace std;

static const bool align = false;
__device__ const int threadnum = 32;


// Kahan's Summation Formula
__global__ void mat_mul_k (float *c, size_t n_c, const float *a, size_t n_a, const float *b, size_t n_b, int n) {
    __shared__ float matA[threadnum][threadnum];
    __shared__ float matB[threadnum][threadnum];

    const int tid_c = threadIdx.x;
    const int tid_r = threadIdx.y;
    const int bid_c = blockIdx.x * threadnum;
    const int bid_r = blockIdx.y * threadnum;
    int i, j;
    float res = 0.0;
    float comp = 0.0;
    float t = 0.0;
    for (j = 0; j < n; j += threadnum) {
        matA[tid_r][tid_c] = a[(bid_r+tid_r)*n_a+tid_c+j];
        matB[tid_r][tid_c] = b[(tid_r+j)*n_b+bid_c+tid_c];
        __syncthreads();
        for (i = 0; i < threadnum; ++i) {
            comp -= matA[tid_r][i] * matB[i][tid_c];
	    t = res - comp;
	    res = t;
	}
        __syncthreads();
    }

    if (tid_r+bid_r<n && tid_c+bid_c<n) {
        c[(bid_r+tid_r)*n_c+bid_c+tid_c] = res;
    }
}


// chessboard array matmul
__global__ void mat_mul (float *c, size_t n_c, const float *a, size_t n_a, const float *b, size_t n_b, int n) {
    __shared__ float matA[threadnum][threadnum];
    __shared__ float matB[threadnum][threadnum];

    const int tid_c = threadIdx.x;
    const int tid_r = threadIdx.y;
    const int bid_c = blockIdx.x * threadnum;
    const int bid_r = blockIdx.y * threadnum;
    int i, j;
    double res = 0.0;
    for (j = 0; j < n; j += threadnum) {
        matA[tid_r][tid_c] = a[(bid_r+tid_r)*n_a+tid_c+j];
	matB[tid_r][tid_c] = b[(tid_r+j)*n_b+bid_c+tid_c];
	__syncthreads();
	for (i = 0; i < threadnum; ++i) res +=  matA[tid_r][i] * matB[i][tid_c];
	__syncthreads();
    }

    if (tid_r+bid_r<n && tid_c+bid_c<n) {
        c[(bid_r+tid_r)*n_c+bid_c+tid_c] = res;
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
    int bx = (N + threadnum - 1) / threadnum;
    dim3 blocks(bx, bx);
    dim3 threads(threadnum, threadnum);

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
	mat_mul<<<blocks, threads>>>(d_o, p_o/sizeof(float), d_a, p_a/sizeof(float), d_b, p_b/sizeof(float), N);
    } else {
        cudaMalloc((void**) &d_a, sizeof(float) * (M * N)); 
        cudaMalloc((void**) &d_b, sizeof(float) * (N * N));
        cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

        cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_mul<<<blocks, threads>>>(d_o, M, d_a, M, d_b, M, N);
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
