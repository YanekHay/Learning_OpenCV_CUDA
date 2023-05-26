#include "utils.h"

#define IMAGE_ID 1
#define WINDOW_WIDTH 320
#define WINDOW_HEIGHT 320

int main() {
	fs::path root = fs::current_path() / "images/low";
	std::vector<std::string> image_paths = get_image_paths(root);
	cv::Mat image = cv::imread(image_paths[IMAGE_ID]);
	if (image.empty()) {
		std::cout << "Unable to read the image from " << image_paths[IMAGE_ID] << std::endl;
		return -1;
	}
	image = _resize(image, cv::Size{ WINDOW_WIDTH, WINDOW_HEIGHT });
	std::cout << "Image size: " << image.rows * image.cols * image.channels() << std::endl;
	cv::imshow("Original", image);
	cv::moveWindow("Original", 100 , 100);
	
	//auto [B_hist, G_hist, R_hist] = calc_bgr_hist(image);
	//auto gray_hist = calc_grayscale_hist(image);
	equalize_hist(image);
	cv::imshow("Equalized", image);
	cv::moveWindow("Equalized", WINDOW_WIDTH, 100);


	inver_color_CUDA(image.data, image.rows, image.cols, image.channels());
	cv::imshow("inverted", image);
	cv::moveWindow("inverted", WINDOW_WIDTH*2, 100);

	cv::waitKey(0);
}



