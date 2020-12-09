// Put global configuration settings in this file
#include <algorithm>
// The window size to use when querying a corner
#define WINDOW_SIZE 3
#define WINDOW_PADDING_SIZE (WINDOW_SIZE / 2)
// The Sobel kernel size to use for computing image gradients
#define GRAD_SIZE 3
#define GRAD_PADDING_SIZE (GRAD_SIZE / 2)
// The window size to use for non-maximum suppression
#define NMS_WINDOW_SIZE 3
// The constant used to scale corner responses
#define K 0.04f
#define THRESH 0.2f
#define TOTAL_PADDING_SIZE (WINDOW_PADDING_SIZE + GRAD_PADDING_SIZE)
#define PADDING_SIZE std::max(WINDOW_SIZE, GRAD_SIZE)

#define MAX_IMAGE_HEIGHT 1440
#define MAX_IMAGE_WIDTH 2556