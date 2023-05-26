#pragma once
#ifndef BIN_COUNT
	#define BIN_COUNT 256
#endif

void hist_equalization_CUDA(unsigned char* image, int height, int width, int channels);
