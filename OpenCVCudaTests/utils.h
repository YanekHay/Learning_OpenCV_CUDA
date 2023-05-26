#pragma once
#include <iostream>
#include <vector>

#include <Windows.h>;
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "hist_calculation_CUDA.h"
#include "hist_equalization_CUDA.h"
#include "invert_color_CUDA.h"

namespace fs = std::filesystem;


std::vector<std::string> get_image_paths(fs::path dir, const std::vector<std::string>& extensions = { ".jpeg", ".png", ".jpg", ".webp"});
cv::Mat _resize(cv::Mat image, const cv::Size& windowSize);

std::array<int, BIN_COUNT> calc_grayscale_hist(cv::Mat& image);
std::tuple<std::array<int, BIN_COUNT>, std::array<int, BIN_COUNT>, std::array<int, BIN_COUNT>> calc_bgr_hist(cv::Mat& image);
void equalize_hist(cv::Mat& image);