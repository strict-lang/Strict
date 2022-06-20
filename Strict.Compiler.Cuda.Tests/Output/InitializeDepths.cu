#include <assert.h>

#define N 10

extern "C" __global__ void Process(const float *input, const int Width, const int Height, const float initialDepth, float *output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = initialDepth;
}

int main(){
    // Allocate host memory
    float *input = (float*)malloc(sizeof(float) * N);
    // Initialize host arrays
    for(int i = 0; i < N; i++){
        input[i] = 1.0f;
    }

		// Allocate device memory
		float *d_input;
    cudaMalloc((void**)&d_input, sizeof(float) * N);
	
    // Transfer data from host to device memory
    cudaMemcpy(d_input, input, sizeof(float) * N, cudaMemcpyHostToDevice);

		// Assign const values
		float initialDepth = 5.0f;
		int Width = 1;
		int Height = 1;
		float *d_output;
    cudaMalloc((void**)&d_output, sizeof(float) * N);
    // Executing kernel 
    Process<<<N,1>>>(d_input, Width, Height, initialDepth, d_output);

		float *output = (float*)malloc(sizeof(float) * N);
    // Transfer data back to host memory
    cudaMemcpy(output, d_output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
				assert(fabs(output[i] - 5.0f) < 0.000001f);
    }
    // Deallocate device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Deallocate host memory
    free(input); 
    free(output);
		return 0;
}