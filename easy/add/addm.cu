#include <iostream>

#define M 2000
#define N 1000

using namespace std;

__global__ void m_add(float c[][N], float a[][N], float b[][N], int m, int n){
    // get ind
    int x_i = threadIdx.x + blockIdx.x;
    int y_i = threadIdx.y + blockIdx.y;

    if (x_i < m && y_i < n) {
        c[x_i][y_i] = a[x_i][y_i] + b[x_i][y_i];
    }
}

int main() {
    float (*h_a)[N] = new float[M][N];
    float (*h_b)[N] = new float[M][N];
    float (*h_o)[N] = new float[M][N];
    float (*d_a)[N], (*d_b)[N], (*d_o)[N];

    // define timer
    cudaEvent_t start, stop;

    // initialize data
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
	    h_a[i][j] = i + j + 2.232f;
	    h_b[i][j] = i + 2 * j;
	}
    }

    // GPU memory allco
    cudaMalloc((void**) &d_a, sizeof(float) * (M * N));
    cudaMalloc((void**) &d_b, sizeof(float) * (M * N));
    cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

    // copy from CPU to GPU
    cudaMemcpy(d_a, h_a, sizeof(float) * (M * N), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * (M * N), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // cuda add
    dim3 block_num(M, N);
    m_add<<<block_num, 1>>>(d_o, d_a, d_b, M, N);

    // copy results back to CPU
    cudaMemcpy(h_o, d_o, sizeof(float) * (M * N), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "The kernel function is during: " << elapsedTime << endl;

    cout << "The final number (M * N): " << h_o[234][21];
    // free GPU mem
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    free(h_a);
    free(h_b);

    return 0;

}
