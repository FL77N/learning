#include <iostream>

#define N 1000000

using namespace std;

__device__ const int thread_num = 256;
static const int block_num = 256;
static const bool cpu_sum = true;

__global__ void  vector_dot_product_1(float *c, float *a, float *b, int n){
    // alloc shared momery for effictive computing
    __shared__ float tmp_arr[thread_num];
    const int t_idx = threadIdx.x;
    const int b_dim = blockDim.x;
    int tid = t_idx;
    double temp = 0.0;
    
    while(tid<n) {
        temp += a[tid] * b[tid];
	tid += b_dim;
    }
    tmp_arr[t_idx] = temp;

    // wait all threads for temp
    __syncthreads();
    int i = 2, j = 1;
    while(i <= thread_num) {
        if ((t_idx % i) == 0) {
	    tmp_arr[t_idx] += tmp_arr[t_idx + j];
	}
	__syncthreads();
	i *= 2;
	j *= 2;
    }
    if (t_idx == 0) c[0] = tmp_arr[0];
}


__global__ void  vector_dot_product_2(float *c, float *a, float *b, int n){
    // alloc shared momery for effictive computing
    __shared__ float tmp_arr[thread_num];
    const int t_idx = threadIdx.x;
    const int b_dim = blockDim.x;
    int tid = t_idx;
    double temp = 0.0;

    while(tid<n) {
        temp += a[tid] * b[tid];
        tid += b_dim;
    }
    tmp_arr[t_idx] = temp;

    // wait all threads for temp
    __syncthreads();
    int i = thread_num / 2;
    while(i != 0) {
        if (t_idx < i) {
            tmp_arr[t_idx] += tmp_arr[t_idx + i];
        }
        __syncthreads();
       i /= 2;
    }
    if (t_idx == 0) c[0] = tmp_arr[0];
}


__global__ void  vector_dot_product_3(float *c, float *a, float *b, int n){
	    // alloc shared momery for effictive computing
    __shared__ float tmp_arr[thread_num];
    const int t_idx = threadIdx.x;
    const int b_idx = blockIdx.x;
    const int step = blockDim.x * gridDim.x;
    int tid = t_idx + b_idx * blockDim.x;
    double temp = 0.0;

    while(tid<n) {
        temp += a[tid] * b[tid];
        tid += step;
    }
    tmp_arr[t_idx] = temp;

    // wait all threads for temp
    __syncthreads();
    int i = thread_num / 2;
    while(i != 0) {
        if (t_idx<i) {
            tmp_arr[t_idx] += tmp_arr[t_idx + i];
        }
        __syncthreads();
	i /= 2;
    }
    if (t_idx == 0) c[b_idx] = tmp_arr[0];
}


// atom
__global__ void vector_dot_product_4(float *c, float *a, float *b, int n) {
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        c[0] = 0.0;
    }
    __shared__ float tmp[thread_num];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
	tid += t_n;
    }
    tmp[tidx] = temp;

    __syncthreads();
    int i = blockDim.x / 2;
    while(i != 0) {
        if (tidx < i) {
	   tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;
    }

    if (tidx == 0) {
        atomicAdd(c, tmp[0]);
    }
}


__device__ void vector_dot(float *t, volatile float *tmp) {
    const int tidx = threadIdx.x;
    int i = blockDim.x / 2;
    while (i != 0) {
        if (tidx < i) {
	    tmp[tidx] += tmp[tidx + i];
	}
	__syncthreads();
	i /= 2;

	if (tidx == 0) t[0] = tmp[0];
    }
}

__device__ unsigned int lockcount = 0;

__global__ void vector_dot_product_5(float *c, float *a, float *b, float *t, int n) {
    __shared__ float tmp[thread_num];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int t_n = blockDim.x * gridDim.x;
    int tid = bidx * blockDim.x + tidx;
    double temp = 0.0;
    while (tid < n) {
        temp += a[tid] * b[tid];
	tid += t_n;
    }
    tmp[tidx] = temp;

    __syncthreads();

    vector_dot(&t[bidx], tmp);

    __shared__ bool lock;
    __threadfence();
    if (tidx == 0) {
        unsigned int lockiii = atomicAdd(&lockcount, 1);
	lock = (lockcount==gridDim.x);
    }
    __syncthreads();
    if(lock) {
	// block_num must less than thread_num
        tmp[tidx] = t[tidx];
	__syncthreads();
	vector_dot(c, tmp);
//	lockcount=0;
    }
}


int main() {
    float *h_a, *h_b, *h_o;
    float *d_a, *d_b, *d_o;

    // define timer
    cudaEvent_t start, stop;

    // initialize data
    h_a = (float*) malloc(sizeof(float) * N);
    h_b = (float*) malloc(sizeof(float) * N);
    h_o = (float*) malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
    	int ind = i;
	h_a[ind] = ind % 2313 + 4.4553f;
	h_b[ind] = ind % 2431 + 2.3232f;
    }


    // GPU memory allco
    cudaMalloc((void**) &d_a, sizeof(float) * N);
    cudaMalloc((void**) &d_b, sizeof(float) * N);
    cudaMalloc((void**) &d_o, sizeof(float) * N);

    // copy from CPU to GPU
    cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // cuda add
 //   vector_dot_product_4<<<block_num, thread_num>>>(d_o, d_a, d_b, N);

    // 计数法
    float *h_t;
    cudaMalloc((void**) &h_t, sizeof(float) * N);
    vector_dot_product_5<<<block_num, thread_num>>>(d_o, d_a, d_b, h_t, N);

    // copy results back to CPU
    cudaMemcpy(h_o, d_o, sizeof(float) * N, cudaMemcpyDeviceToHost);

    if (cpu_sum) {
        float temp = 0;
	for (int i = 0; i < block_num; ++i) {
            temp += h_o[i];
	}
	h_o[0] = temp;
    }
 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "The kernel function is during: " << elapsedTime << "ms" << endl;

    cout << "The final results (M * N): " << h_o[0] << endl;
    // free GPU mem
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    free(h_a);
    free(h_b);

    return 0;

}
