#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <tuple>

#include "CycleTimer.h"
#include "config.h"

#include <chrono>

#define ARG_IMG 1
#define ARG_OUT 2

using namespace cv;
using namespace std;
using namespace std::chrono;

void harrisCornerDetector(unsigned char *input, unsigned char *output, int height, int width);
std::tuple<long int, long int, long int> harrisCornerDetectorStaged(float *pinput, int *output, int height, int width);
void benchSobel(float *pinput, float *output_x, float *output_y, int width, int height, int mode);
void benchCorner(float *pinput, float *output_c, float *output_ix, float *output_iy, float *output_ixy, int height, int width, int mode);
void benchNMS(float *input, int *output, int height, int width, int mode);
void printCudaInfo();
void init_cuda();
void free_cuda();
void deviceQuery();

void output_sobel_response(const char *file_path, float *buf, int height, int width) {
    Mat output(height, width, CV_32FC1, buf, Mat::AUTO_STEP);
    // Take absolute value and normalize
    output = abs(output);
    normalize(output, output, 0x00, 0xFF, NORM_MINMAX, CV_8UC1);

    imwrite(file_path, output);
}

void output_cornerness_response(const char *file_path, float *buf, int height, int width) {
    Mat output(height, width, CV_32FC1, buf, Mat::AUTO_STEP);
    normalize(output, output, 0x00, 0xFF, NORM_MINMAX, CV_8UC1);

    imwrite(file_path, output);
}

void output_corners(const char *file_path, Mat &source, unsigned char *harris, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (harris[i * width + j] == 1) {
                // points out edges that stick out
                circle(source, Point(j,i), 3, Scalar(255));
            }
        }
  }

  imwrite(file_path, source);
}

void serialSobelY(Mat &image, Mat &output) {
    Mat padded;
    copyMakeBorder(image, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    const float kern[3][3] = {{-0.125, -0.25, -0.125},
                         {0.0, 0.0, 0.0},
                         {0.125, 0.25, 0.125}};

    for(int i = 1; i <= image.rows; i++) {
        for(int j = 1; j <= image.cols; j++) {
            for(int k = -1; k <= 1; k++) {
                for(int l = -1; l <= 1; l++) {
                    output.at<float>(i-1, j-1) += padded.at<float>(i+k, j+l) * kern[k+1][l+1];
                }
            }
        }
    }
}

void serialSobelX(Mat &image, Mat &output) {
    Mat padded;
    copyMakeBorder(image, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    const float kern[3][3] = {{-0.125, 0.0, 0.125},
                         {-0.25, 0.0, 0.25},
                         {-0.125, 0.0, 0.125}};

    for(int i = 1; i <= image.rows; i++) {
        for(int j = 1; j <= image.cols; j++) {
            for(int k = -1; k <= 1; k++) {
                for(int l = -1; l <= 1; l++) {
                    output.at<float>(i-1, j-1) += padded.at<float>(i+k, j+l) * kern[k+1][l+1];
                }
            }
        }
    }
}

// takes a non-padded image
void verifySobel(Mat &image, float *grad_x, float *grad_y) {
    Mat output_x = Mat::zeros(image.size(), CV_32FC1), output_y = Mat::zeros(image.size(), CV_32FC1);
    Sobel(image, output_x, CV_32FC1, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    Sobel(image, output_y, CV_32FC1, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
    //serialSobelX(image, output_x);
    //serialSobelY(image, output_y);
    output_x = output_x / 8.0;
    output_y = output_y / 8.0;

    //cout << output_x << "\n";

    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            //printf("%f, ", grad_x[i * image.cols + j]);
            
            if(std::abs((output_x.at<float>(i, j) - grad_x[i * image.cols + j])) > 0.000001) {
                printf("Error in xgrad at %d, %d\n", i, j);
                printf("Expected val %f, actual val %f\n", output_x.at<float>(i, j), grad_x[i * image.cols + j]);
                return;
            }
            if(std::abs((output_y.at<float>(i, j) - grad_y[i * image.cols + j])) > 0.000001) {
                printf("Error in ygrad at %d, %d\n", i, j);
                printf("Expected val %f, actual val %f\n", output_y.at<float>(i, j), grad_y[i * image.cols + j]);
                return;
            }
        }
        //printf("\n");
    }

    printf("Sobel results verified!\n");
}

int main(int argc, char **argv) {
    if(argc < 3) {
        printf("Usage: ./harrisCorner image_path output_path\n");
        exit(-1);
    }

    const char *img_path = argv[ARG_IMG];
    const char *out_path = argv[ARG_OUT];
    Mat img_color = imread(img_path);
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    if(img.empty()) {
        printf("Image %s not found\n", img_path);
        exit(-1);
    }
    img.convertTo(img, CV_32FC1, 1.0/255.0, 0);

    printCudaInfo();
    deviceQuery();
    printf("Size: %i, %i\n", img.rows, img.cols);

    // Pad image according to the gradient and detection window sizes
    Mat img_padded;
    const int border_size = 2;
    printf("Padding size: %i\n", border_size);
    copyMakeBorder(img, img_padded, border_size, border_size, border_size, border_size, BORDER_REPLICATE);

    float *img_buf = new float[img_padded.rows * img_padded.cols];
    int *output_buf = new int[img.rows * img.cols];
    memcpy(img_buf, img_padded.data, sizeof(float) * img_padded.rows * img_padded.cols);

    init_cuda();
    printf("Initialized cuda\n");
    const int iter = 10;
    long int k_dur_sum = 0;
    long int m1_dur_sum = 0;
    long int m2_dur_sum = 0;
    for(int i = 0; i < iter; i++) {
        auto tuple = harrisCornerDetectorStaged(img_buf, output_buf, img.rows, img.cols);
        k_dur_sum += std::get<0>(tuple);
        m1_dur_sum += std::get<1>(tuple);
        m2_dur_sum += std::get<2>(tuple);
    }
    delete[] img_buf;

    //output_sobel_response(out_path, output_buf, img.rows, img.cols);
    // output_cornerness_response(out_path, output_buf, img.rows, img.cols);
    output_corners(out_path, img_color, (unsigned char *)output_buf, img.rows, img.cols);
    delete[] output_buf;

    printf("Avg Kernel Time: %ld us\n", k_dur_sum / iter);
    printf("Avg Mem1 Time: %ld us\n", m1_dur_sum / iter);
    printf("Avg Mem2 Time: %ld us\n", m2_dur_sum / iter);
    printf("Avg Total Time: %ld us\n", (k_dur_sum + m1_dur_sum + m2_dur_sum) / iter);

    Mat out_x = Mat::zeros(img.size(), CV_32FC1);
    Mat out_y = Mat::zeros(img.size(), CV_32FC1);
    Mat out_ixx = Mat::zeros(img_padded.size(), CV_32FC1);
    Mat out_iyy = Mat::zeros(img_padded.size(), CV_32FC1);
    Mat out_ixy = Mat::zeros(img_padded.size(), CV_32FC1);

    // benchSobel(img_buf, (float *)out_x.data, (float *)out_y.data, img.rows, img.cols, 5);
    // verifySobel(img, (float *)out_x.data, (float *)out_y.data);
    // output_sobel_response(out_path, (float *)out_x.data, out_x.rows, out_x.cols);

    //benchNMS(img_buf, NULL, img_padded.rows, img_padded.cols, 0);

    //benchCorner(img_buf, NULL, (float *)out_ixx.data, NULL, NULL, img.rows, img.cols, 1);
    // for(int i = 0; i < img_padded.rows; i++) {
    //     for(int j = 0; j < img_padded.cols; j++) {
    //         printf("%f, ", out_x.data[i * out_ixx.rows + j]);
    //     }
    //     printf("\n");
    // }
    free_cuda();
    return 0;
}