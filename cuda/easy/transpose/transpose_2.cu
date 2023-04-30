#include <iostream>

#define M 1000
#define N 1000

using namespace std;

static const bool align = true;
static const int BLOCK_DIM = 32;


__global__ void mat_transpose_2_1(float *c, float *a, int m, int n)
{
    const unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    if ((xIndex < n) && (yIndex < m))
    {
        unsigned int ind_a = yIndex * n + xIndex;
	unsigned int ind_c = xIndex * m + yIndex;
	c[ind_c] = a[ind_a];
    }
}


__global__ void mat_transpose_2_2(float *c, float *a, int m, int n)
{
    __shared__ float tmp[BLOCK_DIM][BLOCK_DIM+1];
    unsigned int xIndex, yIndex, bidx, bidy, ind_a, ind_c;

    const int mm = (m + BLOCK_DIM - 1) / BLOCK_DIM;
    const int nn = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    for (bidy=blockIdx.y; bidy<mm; bidy+=gridDim.y)
    {
        for (bidx=blockIdx.x; bidx<nn; bidx+=gridDim.x)
	{
	    xIndex = bidx * blockDim.x + threadIdx.x;
	    yIndex = bidy * blockDim.y + threadIdx.y;
	    if ((xIndex < n) && (yIndex < m))
	    {
	        ind_a = yIndex * n + xIndex;
		tmp[threadIdx.y][threadIdx.x] = a[ind_a];
	    }
	    __syncthreads();
	    xIndex = bidy * blockDim.x + threadIdx.x;
	    yIndex = bidx * blockDim.y + threadIdx.y;
	    if ((xIndex < m) && (yIndex < n))
	    {
	        ind_c = yIndex * m + xIndex;
		c[ind_c] = tmp[threadIdx.x][threadIdx.y];
	    }
	}
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
    dim3 threads_shape(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 blocks_shape((N+BLOCK_DIM-1) / BLOCK_DIM, (M+BLOCK_DIM-1) / BLOCK_DIM, 1);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // GPU memory allco and copy data
    if (align) {
	cudaMallocPitch((void**) &d_a, &p_a, sizeof(float) * N, M);
	cudaMallocPitch((void**) &d_o, &p_o, sizeof(float) * M, N);

	cudaMemcpy2D(d_a, p_a, h_a, sizeof(float) * N, sizeof(float) * N, M, cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_transpose_2_2<<<blocks_shape, threads_shape>>>(d_o, d_a, p_a / sizeof(float), N);
    } else {
        cudaMalloc((void**) &d_a, sizeof(float) * (M * N)); 
        cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

        cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	mat_transpose_2_2<<<blocks_shape, threads_shape>>>(d_o, d_a, M, N);
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
