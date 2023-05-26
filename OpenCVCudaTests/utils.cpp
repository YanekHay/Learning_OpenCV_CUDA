#include "utils.h"
#include "hist_calculation_CUDA.h"
#include <tuple>

std::vector<std::string> get_image_paths(fs::path dir, const std::vector<std::string>& extensions) {
	std::vector<std::string> image_paths;
	for (const auto& entry : fs::directory_iterator(dir)) {
		if (entry.is_regular_file()) {
			fs::path file_path = entry.path();
			if (std::find(extensions.begin(), extensions.end(), file_path.extension()) != extensions.end()) {
				image_paths.push_back(file_path.string());
			}
		}
	}

	return image_paths;
}

//Resize the given image by keeping the aspect ratio
cv::Mat _resize(cv::Mat image, const cv::Size& windowSize) {
	// Maybe change in the future to take the image by refference (cv::Mat& image)
	double widthRatio = static_cast<double>(windowSize.width) / image.cols;
	double heightRatio = static_cast<double>(windowSize.height) / image.rows;
	double scale = std::min(widthRatio, heightRatio);

	cv::Size newSize(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale));
	cv::resize(image, image, newSize);

	return image;
}


std::array<int, BIN_COUNT> calc_grayscale_hist(cv::Mat& image) {
	std::array<int, BIN_COUNT> hist_grayscale = { 0 };

	hist_calculation_CUDA(image.data, image.rows, image.cols, image.channels(), hist_grayscale.data());

	for (int i = 0; i < BIN_COUNT; i++) {
		std::cout << "Histogram[" << i << "]: " << hist_grayscale[i] << std::endl;
	}

	return hist_grayscale;
}


std::tuple<std::array<int, BIN_COUNT>, std::array<int, BIN_COUNT>, std::array<int, BIN_COUNT>> calc_bgr_hist(cv::Mat& image) {
	std::array<int, BIN_COUNT> B_hist = { 0 };
	std::array<int, BIN_COUNT> G_hist = { 0 };
	std::array<int, BIN_COUNT> R_hist = { 0 };

	hist_calculation_CUDA(image.data, image.rows, image.cols, image.channels(), B_hist.data(), G_hist.data(), R_hist.data());

	for (int i = 0; i < BIN_COUNT; i++) {
		std::cout << "Histogram[" << i << "]: B:" << B_hist[i] << " G:" << G_hist[i] << " R:" << R_hist[i] << std::endl;
	}

	return std::make_tuple(B_hist, G_hist, R_hist);
}


void equalize_hist(cv::Mat& image) {
	hist_equalization_CUDA(image.data, image.rows, image.cols, image.channels());
}