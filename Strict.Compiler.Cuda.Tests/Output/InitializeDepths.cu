extern "C" __global__ void Process(const float *input, const int Width, const int Height, const float initialDepth, float *output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = initialDepth;
}