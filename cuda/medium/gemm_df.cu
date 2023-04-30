#include <iostream>

using namespace std;

__device__ const int threadnum = 32;


// chessboard array matmul
template <
    const int BLOCK_SIZE_M,
    const int BLOCK_SIZE_N,
    const int BLOCK_SIZE_K,
    const int thread_x,
    const int thread_y
    >
__global__ void mat_mul (float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, const int M, const int N, const int K) {
    // bid
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // block tid
    const int b_tx = threadIdx.x;
    const int b_ty = threadIdx.y;

    // block abs tid
    const int tid = b_ty * thread_x + b_tx;

    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // copy A
    int AIterRow = tid / (32 / 4);
    int AIterCol = tid % (32 / 4) * 4;
    float a_reg[2][4];

    // copy B
    int BIterRow = tid / (128 / 4);
    int BIterCol = tid % (128 / 4) * 4;
    float b_reg[2][4];

    // cal index
    int ACalIndex = tid / 32 * 4;
    int BCalIndex = tid % 32 * 4;

    A = &A[(BLOCK_SIZE_M * by) * K];
    B = &B[BLOCK_SIZE_N * bx];

    // register for C
    float accum[4][4];
    for (int i = 0; i < 4; i++)
    {
         for (int j = 0; j < 4; j++)
	 {
	     accum[i][j] = 0.0;
	 }
    }

    int Iter = 0;
    while(Iter < K / BLOCK_SIZE_K)
    {
        reinterpret_cast<float4*>(&a_reg[0][0])[0] = reinterpret_cast<float4*>(
	    &A[AIterRow * K + AIterCol + Iter * BLOCK_SIZE_K])[0];
	As[AIterCol][AIterRow] = a_reg[0][0];
	As[AIterCol+1][AIterRow] = a_reg[0][1];
	As[AIterCol+2][AIterRow] = a_reg[0][2];
	As[AIterCol+3][AIterRow] = a_reg[0][3];
	reinterpret_cast<float4*>(&Bs[BIterRow][BIterCol])[0] = reinterpret_cast<float4*>(
	    &B[(BIterRow + Iter * BLOCK_SIZE_K) * N + BIterCol])[0];
	__syncthreads();

        reinterpret_cast<float4*>(&a_reg[0][0])[0] = reinterpret_cast<float4*>(
	    &As[0][ACalIndex])[0];
	reinterpret_cast<float4*>(&b_reg[0][0])[0] = reinterpret_cast<float4*>(
	    &Bs[0][BCalIndex])[0];

	int load_idx = 0;
        #pragma unroll
	for (int l = 0; l < BLOCK_SIZE_K-1; ++l)
	{
            reinterpret_cast<float4*>(&a_reg[load_idx^1][0])[0] = reinterpret_cast<float4*>(
	        &As[l+1][ACalIndex])[0];
	    reinterpret_cast<float4*>(&b_reg[load_idx^1][0])[0] = reinterpret_cast<float4*>(
	        &Bs[l+1][BCalIndex])[0];
            #pragma unroll
	    for (int i = 0; i < 4; ++i)
	    {
                 #pragma unroll
	         for (int j = 0; j < 4; ++j)
		 {
                      accum[i][j] += a_reg[load_idx][i] * b_reg[load_idx][j];
		 }
	    }
	    load_idx ^= 1;
        }

        #pragma unroll
	for (int i = 0; i < 4; ++i)
	    #pragma unroll
	    for (int j = 0; j < 4; ++j)
            {
	        accum[i][j] += a_reg[load_idx][i] * b_reg[load_idx][j];
	    }

	__syncthreads();
	++Iter;
    }

    const int c_block_row = ACalIndex;
    const int c_block_col = BCalIndex;
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<float4*>(&C[
	    (BLOCK_SIZE_M*by+c_block_row+i)*N+BLOCK_SIZE_N*bx+c_block_col
	])[0] = reinterpret_cast<float4*>(&accum[i][0])[0];
    }
}


int main() {
    const int M = 2048, N = 2048, K = 2048;
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

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128; 
    const int BLOCK_SIZE_K = 32;

    dim3 blocks(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
    dim3 threads(threadnum, threadnum);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // GPU memory allco and copy data
    cudaMalloc((void**) &d_a, sizeof(float) * (M * K)); 
    cudaMalloc((void**) &d_b, sizeof(float) * (K * N));
    cudaMalloc((void**) &d_o, sizeof(float) * (M * N));

    cudaMemcpy(d_a, h_a, sizeof(float) * (M * K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * (K * N), cudaMemcpyHostToDevice);
     // const int thread_ = threadnum; // thread_ = threadnum;
    cudaEventRecord(start, 0);
    mat_mul<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, threadnum, threadnum><<<blocks, threads>>>(d_o, d_a, d_b, M, N, K);
 
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_o, d_o, sizeof(float) * (M * N), cudaMemcpyDeviceToHost);

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
