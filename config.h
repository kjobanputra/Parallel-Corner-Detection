// Put global configuration settings in this file

// The window size to use when querying a corner
#define WINDOW_SIZE 3
// The Sobel kernel size to use for computing image gradients
#define GRAD_SIZE 3

#define PADDING_SIZE std::max(WINDOW_SIZE, GRAD_SIZE)