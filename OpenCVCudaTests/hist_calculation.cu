#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hist_calculation_CUDA.h"

__global__ void calc_histogram_CUDA(unsigned char* image, int* histogram);
__global__ void calc_histogram_CUDA(unsigned char* image, int channels, int* B_hist, int* G_hist, int* R_hist);

void hist_calculation_CUDA(unsigned char* image, int height, int width, int channels, int* hist_grayscale) {
	unsigned char* cuda_image = NULL;
	int* cuda_hist = NULL;

	int image_size = height * width * channels;
	int hist_size = sizeof(int) * BIN_COUNT;
	//allocate
	cudaMalloc((void**)&cuda_image, image_size);
	cudaMalloc((void**)&cuda_hist, hist_size);

	//copy
	cudaMemcpy(cuda_image, image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_hist, hist_grayscale, hist_size, cudaMemcpyHostToDevice);

	//
	dim3 grid_image(width, height);
	calc_histogram_CUDA<<<grid_image, 1>>> (cuda_image, cuda_hist);
	cudaMemcpy(hist_grayscale, cuda_hist, hist_size, cudaMemcpyDeviceToHost);

	//Free up GPU
	cudaFree(cuda_image);
	cudaFree(cuda_hist);

}



void hist_calculation_CUDA(unsigned char* image, int height, int width, int channels, int* B_hist, int* G_hist, int* R_hist) {
	unsigned char* cuda_image = NULL;
	int* cuda_B_hist = NULL;
	int* cuda_G_hist = NULL;
	int* cuda_R_hist = NULL;

	int image_size = height * width * channels;
	int hist_size = sizeof(int) * BIN_COUNT;

	//allocate
	cudaMalloc((void**)&cuda_image, image_size);
	cudaMalloc((void**)&cuda_B_hist, hist_size);
	cudaMalloc((void**)&cuda_G_hist, hist_size);
	cudaMalloc((void**)&cuda_R_hist, hist_size);


	cudaMemcpy(cuda_image, image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_B_hist, B_hist, hist_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_G_hist, G_hist, hist_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_R_hist, R_hist, hist_size, cudaMemcpyHostToDevice);


	dim3 grid_image(width, height);
	calc_histogram_CUDA << <grid_image, 1 >> > (cuda_image, channels, cuda_B_hist, cuda_G_hist, cuda_R_hist);
	cudaMemcpy(B_hist, cuda_B_hist, hist_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(G_hist, cuda_G_hist, hist_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_hist, cuda_R_hist, hist_size, cudaMemcpyDeviceToHost);

	//Free up GPU
	cudaFree(cuda_image);
	cudaFree(cuda_B_hist);
	cudaFree(cuda_G_hist);
	cudaFree(cuda_R_hist);


}

__global__ void calc_histogram_CUDA(unsigned char* image, int* histogram) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = x + y * gridDim.x;

	atomicAdd(&histogram[image[idx]], 1);
}


__global__ void calc_histogram_CUDA(unsigned char* image, int channels, int* B_hist, int* G_hist, int* R_hist) {
	int x = blockIdx.x;
	int y = blockIdx.y;

	int idx = (x + y * gridDim.x) * channels;

	atomicAdd(&B_hist[image[idx]], 1);
	atomicAdd(&G_hist[image[idx+1]], 1);
	atomicAdd(&R_hist[image[idx+2]], 1);

}