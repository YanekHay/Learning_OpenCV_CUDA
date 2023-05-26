#include "utils.h"

#define ID 1
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 640

int main() {
	fs::path root = fs::current_path() / "images/low";
	std::vector<std::string> image_paths = get_image_paths(root);
	cv::Mat image = cv::imread(image_paths[ID]);
	if (image.empty()) {
		std::cout << "Unable to read the image from " << image_paths[ID] << std::endl;
		return -1;
	}
	image = _resize(image, cv::Size{ WINDOW_WIDTH, WINDOW_HEIGHT });
	std::cout << "Image size: " << image.rows * image.cols * image.channels() << std::endl;
	cv::imshow("Original", image);
	cv::moveWindow("Original", 100, 100);

	//auto [B_hist, G_hist, R_hist] = calc_bgr_hist(image);
	//auto gray_hist = calc_grayscale_hist(image);
	equalize_hist(image);

	cv::imshow("Equalized", image);
	cv::moveWindow("Equalized", WINDOW_WIDTH+100, 100);

	cv::waitKey(0);
}



