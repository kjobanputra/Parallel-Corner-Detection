#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

#define ARG_IMG 1

using namespace cv;
using namespace std;
const char* window = "window";
Mat srcGray, paddedSrcGray, gaussianConvolvedMatrix;

const float gaussianKernel[3][3] = {
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)},
    {(1.0 / 8), (1.0 / 4), (1.0 / 8)},
    {(1.0 / 16), (1.0 / 8), (1.0 / 16)}
};

const float gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const float gy[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

const float k = 0.04;
const int thresholdVal = 100; // How sensitive it is to detecting corners
const int convolutionWindowSize = 3;

float partialX(int i, int j) {
  float gxVal;
  float partialXVal = 0;
  for (int l = i-1; l <= i+1; l++) {
    for (int m = j-1; m <= j+1; j++) {
      gxVal = gx[l-(i-1)][m-(j-1)];
      partialXVal += gxVal * gaussianConvolvedMatrix.at<float>(l,m);
    }
  }

  return partialXVal;
}

float partialY(int i, int j) {
  float gyVal;
  float partialYVal = 0;
  for (int l = i-1; l <= i+1; l++) {
    for (int m = j-1; m <= j+1; j++) {
      gyVal = gy[l-(i-1)][m-(j-1)];
      partialYVal += gyVal * gaussianConvolvedMatrix.at<float>(l,m);
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
      output.at<float>(i,j) = convolvedVal;
    }
  }
}

void nonMaxSupression(Mat cMatrix, Mat &harris) {
  for (int i = 0; i < cMatrix.rows; i++) {
    for (int j = 0; j < cMatrix.cols; j++) {
      float maxVal = 0;
      int maxK = -1;
      int maxL = -1;
      for (int k = i; k < min(i + convolutionWindowSize, cMatrix.rows); k++) {
        for (int l = j; l < min(j + convolutionWindowSize, cMatrix.cols); l++) {
          if (cMatrix.at<float>(k, l) > maxVal) {
            maxVal = cMatrix.at<float>(k,l);
            maxK = k;
            maxL = l;
          }
        }
      }

      if (maxVal > 0) {
        harris.at<float>(maxK, maxL) = cMatrix.at<float>(maxK, maxL);
      }
    }
  }
}

void cornerHarris(Mat &harris) {
  gaussianConvolvedMatrix = Mat::zeros(srcGray.rows+2, srcGray.cols+2, CV_32FC1);
  gaussianConvolution(gaussianConvolvedMatrix);

  for (int i = 1; i < gaussianConvolvedMatrix.rows-1; i++) {
    for (int j = 1; j < gaussianConvolvedMatrix.cols-1; j++) {
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

  if (src.empty())
  {
    cout << "Image not found: " << img_path << endl;
    return -1;
  }

  cvtColor(src, srcGray, COLOR_BGR2GRAY);
  if(srcGray.empty()) {
      printf("Image %s not found\n", img_path);
      exit(-1);
  }

  copyMakeBorder(srcGray, paddedSrcGray, 2, 2, 2, 2, BORDER_REPLICATE);

  Mat harris = Mat::zeros(src.size(), CV_32FC1);
  cornerHarris(harris);
  for (int i = 0; i < harris.rows; i++) {
    for (int j = 0; j < harris.cols; j++) {
      cout << "Harris at " << i;
      cout << " " << j << " is " << harris.at<float>(i,j) << endl;
      if (harris.at<float>(i,j) > thresholdVal) {
        // points out edges that stick out
        circle(src, Point(j,i), 3, Scalar(100));
      }
    }
  }

  //namedWindow(window);
  //imshow(window, src);
  //waitKey();
  imwrite("output/callibration.jpg", src);
  return 0;
}