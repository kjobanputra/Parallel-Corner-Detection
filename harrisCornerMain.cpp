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
void harrisCornerDetectorStaged(float *pinput, float *output, int height, int width, int pheight, int pwidth);
void printCudaInfo();


int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: ./harrisCorner image_path\n");
        exit(-1);
    }

    const char *img_path = argv[ARG_IMG];
    const char *out_path = argv[ARG_OUT];
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    if(img.empty()) {
        printf("Image %s not found\n", img_path);
        exit(-1);
    }

    img.convertTo(img, CV_32FC1, 1.0/255.0);

    printCudaInfo();
    printf("Image type %i\n", img.type());
    printf("CV_32FC1: %i\n", CV_32FC1);
    printf("CV_8UC1: %i\n", CV_8UC1);
    printf("Size: %i, %i\n", img.rows, img.cols);

    // Pad image according to the gradient and detection window sizes
    Mat img_padded;
    const int border_size = 1;
    printf("Padding size: %i\n", border_size);
    copyMakeBorder(img, img_padded, border_size, border_size, border_size, border_size, BORDER_REPLICATE);

    float *img_buf = new float[img_padded.rows * img_padded.cols];
    float *output_buf = new float[img.rows * img.cols];
    memcpy(img_buf, img_padded.data, sizeof(unsigned char) * img_padded.rows * img_padded.cols);

    // harrisCornerDetector(img_buf, output_buf, img.rows, img.cols);

    harrisCornerDetectorStaged(img_buf, output_buf, img.rows, img.cols, 1, 1);
    
    delete[] img_buf;

    Mat output(img.rows, img.cols, CV_32FC1, output_buf);

    imwrite(out_path, output);
    return 0;
}