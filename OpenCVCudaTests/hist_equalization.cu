#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hist_equalization_CUDA.h"

#define TARGET_MIN 0
#define TARGET_MAX 255

__global__ void calc_min_max(unsigned char* image, int channels, int* Min, int* Max);
__global__ void histogram_equalization(unsigned char* image, int channels, int* Min, int* Max);
__device__ int New_Pixel_Value(int Value, int Min, int Max);

void hist_equalization_CUDA(unsigned char* image, int height, int width, int channels) {
	unsigned char* cuda_image = NULL;
	int* cuda_Min = NULL;
	int* cuda_Max = NULL;

	int image_size = height * width * channels;
	int min_max_size = sizeof(int) * channels;

	//allocate
	cudaMalloc((void**)&cuda_image, image_size);
	cudaMalloc((void**)&cuda_Min, min_max_size);
	cudaMalloc((void**)&cuda_Max, min_max_size);

	
	int Min[3] = { 255, 255, 255 };
	int Max[3] = { 0, 0, 0 };

	cudaMemcpy(cuda_image, image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_Min, Min, min_max_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_Max, Max, min_max_size, cudaMemcpyHostToDevice);


	dim3 grid_image(width, height);
	calc_min_max <<<grid_image, 1>>> (cuda_image, channels, cuda_Min, cuda_Max);
	histogram_equalization <<<grid_image, 1>>> (cuda_image, channels, cuda_Min, cuda_Max);

	cudaMemcpy(image, cuda_image, image_size, cudaMemcpyDeviceToHost);

	//Free up GPU
	cudaFree(cuda_image);
	cudaFree(cuda_Min);
	cudaFree(cuda_Max);



}


__global__ void calc_min_max(unsigned char* image, int channels, int* Min, int* Max) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * channels;
	for (int i = 0; i < channels; i++) {
		atomicMin(&Min[i], image[idx + i]);
		atomicMax(&Max[i], image[idx + i]);
	}

}

__global__ void histogram_equalization(unsigned char* image, int channels, int* Min, int* Max) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * channels;
	for (int i = 0; i < channels; i++) {
		image[idx + i] = New_Pixel_Value(image[idx + i], Min[i], Max[i]);
	}
}


__device__ int New_Pixel_Value(int Value, int Min, int Max) {
	return (TARGET_MIN + (Value - Min) * (int)((TARGET_MAX - TARGET_MIN) / (Max - Min)));
}
