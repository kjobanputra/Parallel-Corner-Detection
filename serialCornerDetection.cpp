#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#include "CycleTimer.h"

// For timing
#include <chrono>

#define ARG_IMG 1
#define ARG_OUT 2

//TODO: potential seg fault in code
using namespace cv;
using namespace std;
Mat srcGray, paddedSrcGray, gaussianConvolvedMatrix;

const float gaussianKernel[3][3] = {
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)},
    {(1.0 / 8), (1.0 / 4), (1.0 / 8)},
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)}
};

const float gx[3][3] = {
    {0.125, 0, -0.125},
    {0.25, 0, -0.25},
    {0.125, 0, -0.125}
};

const float gy[3][3] = {
    {0.125, 0.25, 0.125},
    {0, 0, 0},
    {-0.125, -0.25, -0.125}
};

const float k = 0.04;
const float thresholdVal = 0.4; // How sensitive it is to detecting corners
const int convolutionWindowSize = 3;

// Using Sobel Operator
float partialX(int i, int j) {
  float gxVal;
  float partialXVal = 0;
  for (int l = i-1; l <= i+1; l++) {
    for (int m = j-1; m <= j+1; m++) {
      gxVal = gx[l-(i-1)][m-(j-1)];
      partialXVal += gxVal * paddedSrcGray.at<float>(l,m);
    }
  }

  return partialXVal;
}

// Using Sobel operator
float partialY(int i, int j) {
  float gyVal;
  float partialYVal = 0;
  for (int l = i-1; l <= i+1; l++) {
    for (int m = j-1; m <= j+1; m++) {
      gyVal = gy[l-(i-1)][m-(j-1)];
      partialYVal += gyVal * paddedSrcGray.at<float>(l,m);
    }
  }

  return partialYVal;
}

float det(int gxx, int gxy, int gyy) {
  return gxx*gyy - gxy*gxy;
}

float trace(int gxx, int gyy) {
  return gxx + gyy;
}

// Computer convolution centered at (i,j) with size convolutionWindowSize by
// convolutionWindowSize.
float c(int i, int j) {
  float partialXVal = partialX(i, j);
  float partialYVal = partialY(i, j);
  float gxx = partialXVal*partialXVal;
  float gxy = partialXVal*partialYVal;
  float gyy = partialYVal*partialYVal;

  float traceVal = trace(gxx, gyy);
  return det(gxy, gxy, gyy) - k * traceVal * traceVal;
}

void gaussianConvolution(Mat &output) {
  int halfWindow = convolutionWindowSize/2;
  float gaussianVal;
  float convolvedVal;
  for (int i = 1; i < paddedSrcGray.rows-1; i++) {
    for (int j = 1; j < paddedSrcGray.cols-1; j++) {
      convolvedVal = 0;
      for (int l = i-halfWindow; l <= i+halfWindow; l++) {
        for (int m = j-halfWindow; m <= j+halfWindow; m++) {
          gaussianVal = gaussianKernel[l-(i-halfWindow)][m-(j-halfWindow)];
          convolvedVal += paddedSrcGray.at<float>(l, m) * gaussianVal;
        }
      }
      output.at<float>(i-1,j-1) = convolvedVal;
    }
  }
}

void nonMaxSupression(Mat cMatrix, Mat &harris) {
  for (int i = 1; i < cMatrix.rows-1; i++) {
    for (int j = 1; j < cMatrix.cols-1; j++) {
      float maxVal = 0;
      int maxK = -1;
      int maxL = -1;
      bool max = true;
      for (int k = -1; k <= 1; k++) {
        for (int l = -1; l <= 1; l++) {
          if (cMatrix.at<float>(i + k, j + l) > cMatrix.at<float>(i, j)) {
            harris.at<float>(i, j) = 0.0f;
            max = false;
            break;
          }
        }
      }

      if (max && cMatrix.at<float>(i, j) > 0) {
        harris.at<float>(i, j) = cMatrix.at<float>(i, j);
      }
    }
  }
}

void cornerHarris(Mat &harris) {
  //gaussianConvolvedMatrix = Mat::zeros(srcGray.rows+2, srcGray.cols+2, CV_32FC1);
  //gaussianConvolution(gaussianConvolvedMatrix);

  Mat cMatrix = Mat::zeros(harris.size(), CV_32FC1);

  for (int i = 1; i < srcGray.rows + 1; i++) {
    for (int j = 1; j < srcGray.cols + 1; j++) {
      cMatrix.at<float>(i-1, j-1) =  c(i, j);
    }
  }

  // NMS 
  nonMaxSupression(cMatrix, harris);
}

int main(int argc, char **argv) {
  if(argc < 3) {
    printf("Usage: ./harrisCorner image_path output_path\n");
    exit(-1);
  }

  const char *img_path = argv[ARG_IMG];
  const char *out_path = argv[ARG_OUT];
  Mat src = imread(img_path);

  if (src.empty())
  {
    cout << "Image not found: " << img_path << endl;
    return -1;
  }

  cvtColor(src, srcGray, COLOR_BGR2GRAY);
  srcGray.convertTo(srcGray, CV_32FC1, 0.0115, 0);
  if(srcGray.empty()) {
      printf("Image %s not found\n", img_path);
      exit(-1);
  }

  copyMakeBorder(srcGray, paddedSrcGray, 2, 2, 2, 2, BORDER_REPLICATE);

  Mat harris = Mat::zeros(src.size(), CV_32FC1);
  auto startTime = std::chrono::high_resolution_clock::now();
  cornerHarris(harris);
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

  //cout << harris << "\n";
  for (int i = 0; i < harris.rows; i++) {
    for (int j = 0; j < harris.cols; j++) {
      if (harris.at<float>(i,j) > thresholdVal) {
        // points out edges that stick out
        circle(src, Point(j,i), 3, Scalar(255));
      }
    }
  }

  imwrite(out_path, src);
  cout << "Serial: " << duration.count() << endl;
  return 0;
}