extern "C" __global__ void AddNumbers(const int *first, const int *second, int* output, const int count)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * blockDim.x + ix;
	output[idx] = first[idx] + second[idx];
}