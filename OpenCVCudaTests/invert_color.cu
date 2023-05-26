#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define MAX_VALUE 255

__global__ void invert_CUDA(unsigned char* image, int channels);

void inver_color_CUDA(unsigned char* image, int height, int width, int channels) {
	unsigned char* cuda_image = NULL;

	int image_size = height * width * channels;
	
	//allocate
	cudaMalloc((void**)&cuda_image, image_size);
	cudaMemcpy(cuda_image, image, image_size, cudaMemcpyHostToDevice);

	dim3 grid_image(width, height);
	invert_CUDA << <grid_image, 1 >> > (cuda_image, channels);
	cudaMemcpy(image, cuda_image, image_size, cudaMemcpyDeviceToHost);

	//Free up GPU
	cudaFree(cuda_image);
}


__global__ void invert_CUDA(unsigned char* image, int channels) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * channels;
	for (int i = 0; i < channels; i++) {
		image[idx + i] = MAX_VALUE - image[idx + i];
	}
}