#pragma once

#ifndef BIN_COUNT
	#define BIN_COUNT 256
#endif
void hist_calculation_CUDA(unsigned char* image, int height, int width, int channels, int* hist_grayscale);
void hist_calculation_CUDA(unsigned char* image, int height, int width, int channels, int* B_hist, int* G_hist, int* R_hist);