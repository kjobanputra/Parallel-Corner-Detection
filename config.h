// Put global configuration settings in this file
#include <algorithm>
// The window size to use when querying a corner
#define WINDOW_SIZE 3
#define WINDOW_PADDING_SIZE (WINDOW_SIZE / 2)
// The Sobel kernel size to use for computing image gradients
#define GRAD_SIZE 3
#define GRAD_PADDING_SIZE (GRAD_SIZE / 2)
// The constant used to scale corner responses
#define K 0.04f
#define TOTAL_PADDING_SIZE (WINDOW_PADDING_SIZE + GRAD_PADDING_SIZE)
#define PADDING_SIZE std::max(WINDOW_SIZE, GRAD_SIZE)