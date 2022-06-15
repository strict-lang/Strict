extern "C" __global__ void Subtract(const float *first, const float *second, float *output, const int count)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = first[idx] - second[idx];
}