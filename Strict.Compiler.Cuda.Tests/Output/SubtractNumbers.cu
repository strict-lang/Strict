#include <assert.h>

#define N 10

extern "C" __global__ void Subtract(const float *first, const float *second, float *output, const int count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = first[idx] - second[idx];
}

int main(){
    // Allocate host memory
    float *first = (float*)malloc(sizeof(float) * N);
		float *second = (float*)malloc(sizeof(float) * N);
    // Initialize host arrays
    for(int i = 0; i < N; i++){
        first[i] = 2.0f;
				second[i] = 1.0f;
    }

		// Allocate device memory
		float *d_first;
		float *d_second;
    cudaMalloc((void**)&d_first, sizeof(float) * N);
    cudaMalloc((void**)&d_second, sizeof(float) * N);
	
    // Transfer data from host to device memory
    cudaMemcpy(d_first, first, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_second, second, sizeof(float) * N, cudaMemcpyHostToDevice);

		// Assign const values
		int count = 1;
		float *d_output;
    cudaMalloc((void**)&d_output, sizeof(float) * N);
    // Executing kernel 
    Subtract<<<N,1>>>(d_first, d_second, d_output, count);

		float *output = (float*)malloc(sizeof(float) * N);
    // Transfer data back to host memory
    cudaMemcpy(output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
				assert(fabs(output[i] - 1.0f) < 0.000001f);
    }
    // Deallocate device memory
    cudaFree(d_first);
    cudaFree(d_second);
    cudaFree(d_output);

    // Deallocate host memory
    free(first); 
    free(second); 
    free(output);
		return 0;
}