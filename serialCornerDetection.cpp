#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#define ARG_IMG 1

using namespace cv;
using namespace std;
Mat src, src_gray;

const float gaussianKernel[3][3] = {
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)},
    {(1.0 / 8), (1.0 / 4), (1.0 / 8)},
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)}
};

const float k = 0.04;
const int thresholdVal = 100; // How sensitive it is to detecting corners
const int convolutionWindowSize = 3;

float partialX(int i, int j) {
  if (j > 0 && j < src_gray.cols-1) {
     return (src_gray.at<float>(i, j+1) - src_gray.at<float>(i, j-1))/2;
  }

  return 0;
}

float partialY(int i, int j) {
  if (i > 0 && i < src_gray.cols-1) {
    return (src_gray.at<float>(i+1, j) - src_gray.at<float>(i-1, j))/2;
  }

  return 0;
}

float det(int gxx, int gxy, int gyy) {
  return gxx*gyy + gxy*gxy;
}

float trace(int gxx, int gyy) {
  return gxx + gyy;
}

// Computer convolution centered at (i,j) with size convolutionWindowSize by
// convolutionWindowSize.
float c(int i, int j) {
  int halfWindow = convolutionWindowSize/2;
  float gxx = 0;
  float gxy = 0;
  float gyy = 0;

  for (int l = i - halfWindow; l <= i + halfWindow; l++) {
    for (int m = j - halfWindow; m <= j + halfWindow; m++) {
      int partialXVal = partialX(l, m);
      int partialYVal = partialY(l, m);
      int gaussianVal = gaussianKernel[l - (i - halfWindow)][m - (j - halfWindow)];
      gxx += partialXVal*partialXVal*gaussianVal;
      gxy += partialXVal*partialYVal*gaussianVal;
      gyy += partialYVal*partialYVal*gaussianVal;
    }
  }

  float traceVal = trace(gxx, gyy);
  return det(gxy, gxy, gyy) - k * traceVal * traceVal;

}

void cornerHarris(Mat harris) {
  for (int i = 0; i < src_gray.rows; i++) {
    for (int j = 0; j < src_gray.cols; j++) {
      harris.at<float>(i, j) = c(i, j);
    }
  }
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: ./harrisCorner image_path\n");
        exit(-1);
    }

    const char *img_path = argv[ARG_IMG];
    Mat src = imread(img_path);
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    if(src_gray.empty()) {
        printf("Image %s not found\n", img_path);
        exit(-1);
    }

  //cvtColor(src, src_gray, COLOR_BGR2GRAY);
  Mat harris = Mat::zeros(src.size(), CV_32FC1);
  cornerHarris(harris);
  Mat harrisNorm, harrisNormScaled;
  for (int i = 0; i < harris.rows; i++) {
    for (int j = 0; j < harris.cols; j++) {
      if (harris.at<float>(i,j) > thresholdVal) {
        // points out edges that stick out
        circle(harrisNormScaled, Point(j,i), 3, Scalar(0));
      }
    }
  }

  //const char* window = "window";
  //namedWindow(window);
  //imshow(window, harris);
  imwrite("output/callibration.jpg", harris);
  return 0;
}