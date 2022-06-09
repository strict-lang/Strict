extern "C" __global__ void InitializeDepth(const float *input, float *output, const int Width, const int Height, const float initialDepth)
{   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	output[y * Width + x] = initialDepth;
}