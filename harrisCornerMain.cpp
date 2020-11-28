#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <stdio.h>

#include "config.h"

#define ARG_IMG 1
#define ARG_OUT 2

using namespace cv;
using namespace std;

void harrisCornerDetector(unsigned char *input, unsigned char *output, int height, int width);
void harrisCornerDetectorStaged(float *pinput, float *output, int height, int width);
void printCudaInfo();

void output_sobel_response(const char *file_path, float *buf, int height, int width) {
    Mat output(height, width, CV_32FC1, buf, Mat::AUTO_STEP);
    // Take absolute value and normalize
    output = abs(output);
    normalize(output, output, 0x00, 0xFF, NORM_MINMAX, CV_8UC1);

    imwrite(file_path, output);
}

void output_cornerness_response(const char *file_path, float *buf, int height, int width) {
    Mat output(height, width, CV_32FC1, buf, Mat::AUTO_STEP);
    for(int i = 0; i < width * height; i++) {
        printf("%f, ", buf[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    if(argc < 3) {
        printf("Usage: ./harrisCorner image_path output_path\n");
        exit(-1);
    }

    const char *img_path = argv[ARG_IMG];
    const char *out_path = argv[ARG_OUT];
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    if(img.empty()) {
        printf("Image %s not found\n", img_path);
        exit(-1);
    }

    img.convertTo(img, CV_32FC1, 1.0/255.0, 0);

    printCudaInfo();
    printf("Size: %i, %i\n", img.rows, img.cols);

    // Pad image according to the gradient and detection window sizes
    Mat img_padded;
    const int border_size = TOTAL_PADDING_SIZE;
    printf("Padding size: %i\n", border_size);
    copyMakeBorder(img, img_padded, border_size, border_size, border_size, border_size, BORDER_REPLICATE);

    float *img_buf = new float[img_padded.rows * img_padded.cols];
    float *output_buf = new float[img.rows * img.cols];
    memcpy(img_buf, img_padded.data, sizeof(float) * img_padded.rows * img_padded.cols);

    // harrisCornerDetector(img_buf, output_buf, img.rows, img.cols);

    harrisCornerDetectorStaged(img_buf, output_buf, img.rows, img.cols);
    
    delete[] img_buf;

    //output_sobel_response(out_path, output_buf, img.rows, img.cols);
    output_cornerness_response(out_path, output_buf, img.rows, img.cols);
    delete[] output_buf;
    return 0;
}