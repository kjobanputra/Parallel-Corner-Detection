#include <stdio.h>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "config.h"

#include <chrono>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)
#define TILE_WIDTH 30

#define SHARED_PADDING (WINDOW_SIZE / 2) + (GRAD_SIZE / 2) 

#define dceil(a, b) ((a) % (b) != 0 ? ((a) / (b) + 1) : ((a) / (b)))

#define INSIDE (1.0f)
#define OUTSIDE (0.0f)
#define UNDEFINED (-1.0f)

#define MAX_F(a, b) ((a) > (b) ? (a) : (b))
#define MIN_F(a, b) ((a) < (b) ? (a) : (b))

using namespace std::chrono;

// Constant pointer to the image data
__constant__ float *image_data;
__constant__ float *image_x_grad;
__constant__ float *image_y_grad;
char *c_image_data;
float *c_image_x_grad;
float *c_image_y_grad;

// __global__
// void harrisCornerDetector_kernel(float *input, float *output, int height, int width) {
//     const int shared_size = THREADS_PER_BLOCK + 2 * THREADS_PER_BLOCK * SHARED_PADDING + SHARED_PADDING * SHARED_PADDING;
//     __shared__ float image_dx[shared_size];
//     __shared__ float image_dy[shared_size];

//     uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//     uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
// }

__global__
void gaussian_kernel(float *image, float *output, int height, int width, int pheight, int pwidth) {
    // TODO:
}

__global__
void sobel_kernel(int h, int w) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    const int dxl = (pX == 0) ? 0 : -1;
    const int dyu = (pY == 0) ? 0 : -1;
    const int dxr = (pX == w - 1) ? 0 : 1;
    const int dyd = (pY == h - 1) ? 0 : 1;

    if(pX < w && pY < h) {
        float x_grad = 0.125f * image_data[(pY + dyu) * w + pX + dxl] + 0.25f * image_data[pY * w + pX + dxl] + 0.125f * image_data[(pY + dyd) * w + pX + dxl] +
                    -0.125f * image_data[(pY + dyu) * w + pX + dxr] + -0.25f * image_data[pY * w + pX + dxr] + -0.125f * image_data[(pY + dyd) * w + pX + dxr];
        float y_grad = 0.125f * image_data[(pY + dyu) * w + pX + dxl] + 0.25f * image_data[(pY + dyu) * w + pX] + 0.125f * image_data[(pY + dyu) * w + pX + dxr] + 
                    -0.125f * image_data[(pY + dyd) * w + pX + dxl] + -0.25f * image_data[(pY + dyd) * w + pX] + -0.125f * image_data[(pY + dyd) * w + pX + dxr];

        image_x_grad[pY * w + pX] = x_grad;
        image_y_grad[pY * w + pX] = y_grad;
    }
}

__global__
void sobel_kernel_padded(int height, int width, int pheight, int pwidth) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppY = pY + pheight;
    const uint ppX = pX + pwidth;
    const uint w = 2 * pwidth + width;

    if(pX < width && pY < height) {
        float x_grad = 0.125f * image_data[(ppY - 1) * w + ppX - 1] + 0.25f * image_data[ppY * w + ppX - 1] + 0.125f * image_data[(ppY + 1) * w + ppX - 1] +
                    -0.125f * image_data[(ppY - 1) * w + ppX + 1] + -0.25f * image_data[ppY * w + ppX + 1] + -0.125f * image_data[(ppY + 1) * w + ppX + 1];
        float y_grad = 0.125f * image_data[(ppY - 1) * w + ppX - 1] + 0.25f * image_data[(ppY - 1) * w + ppX] + 0.125f * image_data[(ppY - 1) * w + ppX + 1] + 
                    -0.125f * image_data[(ppY + 1) * w + ppX - 1] + -0.25f * image_data[(ppY + 1) * w + ppX] + -0.125f * image_data[(ppY + 1) * w + ppX + 1];

        image_x_grad[pY * width + pX] = x_grad;
        image_y_grad[pY * width + pX] = y_grad;
    }
}

// Expects padding
__global__
void sobel_kernel_shared(int height, int width) {
    __shared__ float image_shared[THREADS_PER_BLOCK];
    const uint pY = blockIdx.y * (TILE_WIDTH) + threadIdx.y;
    const uint pX = blockIdx.x * (TILE_WIDTH) + threadIdx.x;
    const uint ty = threadIdx.y;
    const uint tx = threadIdx.x;

    if(pX <= width && pY <= height) {
        image_shared[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = image_data[pY * (width + 2) + pX];
    }
    __syncthreads();

    if(threadIdx.x > 0 && threadIdx.x < BLOCK_WIDTH - 1 && threadIdx.y > 0 && threadIdx.y < BLOCK_WIDTH - 1) {
        float x_grad = 0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx - 1] + 0.25f * image_shared[ty * BLOCK_WIDTH + tx - 1] + 0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx - 1] +
                    -0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx + 1] + -0.25f * image_shared[ty * BLOCK_WIDTH + tx + 1] + -0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx + 1];
        float y_grad = 0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx - 1] + 0.25f * image_shared[(ty - 1) * BLOCK_WIDTH + tx] + 0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx + 1] + 
                    -0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx - 1] + -0.25f * image_shared[(ty + 1) * BLOCK_WIDTH + tx] + -0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx + 1];

        image_x_grad[pY * width + pX] = x_grad;
        image_y_grad[pY * width + pX] = y_grad;
    }
}

__global__
void sobel_x_kernel(int height, int width, int pheight, int pwidth) {
    const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppixelY = pixelY + pheight;
    const uint ppixelX = pixelX + pwidth;
    

    if(pixelX < width && pixelY < height) {
        float value = 0.0f; 
        // left horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
            value += weight * image_data[(ppixelY + i) * (2 * pwidth + width) + ppixelX - 1];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image_data[(ppixelY + i) * (2 * pwidth + width) + ppixelX + 1];
        }
        image_x_grad[pixelY * width + pixelX] = value;
    }
}

__global__
void sobel_y_kernel(int height, int width, int pheight, int pwidth) {
    const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppixelY = pixelY + pheight;
    const uint ppixelX = pixelX + pwidth;

    if(pixelX < width && pixelY < height) {
        float value = 0.0f; 
        // left horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
            value += weight * image_data[(ppixelY - 1) * (2 * pwidth + width) + ppixelX + i];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image_data[(ppixelY + 1) * (2 * pwidth + width) + ppixelX + i];
        }
        image_y_grad[pixelY * width + pixelX] = value;
    }
}

__global__
void cornerness_kernel(int height, int width) {
    const int padding = WINDOW_PADDING_SIZE;
    const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppixelY = pixelY + padding;
    const uint ppixelX = pixelX + padding;

    if(pixelX < width && pixelY < height) {
        float gxx = 0.0f, gyy = 0.0f, gxy = 0.0f;
        for(int i = -padding; i <= padding; i++) {
            for(int j = -padding; j <= padding; j++) {
                const uint pos = (ppixelY + i) * (2 * padding + width) + ppixelX + j;
                gxx += image_x_grad[pos] * image_x_grad[pos];
                gyy += image_y_grad[pos] * image_y_grad[pos];
                gxy += image_x_grad[pos] * image_y_grad[pos];
            }
        }
        const float det = gxx * gyy - gxy * gxy;
        const float trace = gxx + gyy;
        //const float val = (det - K * trace * trace >= THRESH) ? 1.0f : 0.0f;
        image_data[pixelY * width + pixelX] = det - K * trace * trace;
    }
}

/* algorithm borrowed from: http://www.bmva.org/bmvc/2008/papers/45.pdf */
__global__
void non_maximum_suppression_kernel(float *cornerness, float *input, float *output, int height, int width, bool *done) {
    
}

/* input is a grayscale image of size height by width */
std::tuple<long int, long int, long int> harrisCornerDetectorStaged(float *pinput, float *output, int height, int width) {
    auto start_time = high_resolution_clock::now();
    const size_t padding = TOTAL_PADDING_SIZE;

    const int input_image_size = sizeof(float) * (height + 2 * padding) * (width + 2 * padding);
    const int grad_image_width = width + 2 * padding;
    const int grad_image_height = height + 2 * padding;
    const int output_image_size = sizeof(float) * height * width;

    // Copy input arrays to the GPU
    auto mem_start_time1 = high_resolution_clock::now();
    cudaMemcpy(c_image_data, pinput, input_image_size, cudaMemcpyHostToDevice);
    auto mem_end_time1 = high_resolution_clock::now();

    const dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    const dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
    auto kernel_start_time = high_resolution_clock::now();
    //sobel_x_kernel<<<grid, threadBlock>>>(grad_image_height, grad_image_width, GRAD_PADDING_SIZE, GRAD_PADDING_SIZE);
    //sobel_y_kernel<<<grid, threadBlock>>>(grad_image_height, grad_image_width, GRAD_PADDING_SIZE, GRAD_PADDING_SIZE);
    sobel_kernel<<<grid, threadBlock>>>(grad_image_height, grad_image_width);
    cudaDeviceSynchronize();
    cornerness_kernel<<<grid, threadBlock>>>(height, width);
    cudaDeviceSynchronize();
    auto kernel_end_time = high_resolution_clock::now();

    // Copy result to CPU
    auto mem_start_time2 = high_resolution_clock::now();
    cudaMemcpy(output, c_image_data, output_image_size, cudaMemcpyDeviceToHost);
    auto mem_end_time2 = high_resolution_clock::now();
    auto end_time = high_resolution_clock::now();

    return std::make_tuple(duration_cast<microseconds>(kernel_end_time - kernel_start_time).count(), 
                                    duration_cast<microseconds>(mem_end_time1 - mem_start_time1).count(), 
                                    duration_cast<microseconds>(mem_end_time2 - mem_start_time2).count());
}

// __global__
// void sobel_x_kernel_shared(int height, int width) {
//     __shared__ float image_shared[THREADS_PER_BLOCK + 4 * (BLOCK_HEIGHT + 1)];

//     const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//     const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint pixelSL = (blockIdx.y + 1) * BLOCK_WIDTH + blockIdx.x + 1;

//     if(pixelX < width && pixelY < height) {
//         // Load image value into shared memory
//         image_shared[pixelSL] = image_data[pixelY * width + pixelX];
//         if(threadIdx.x == 0 && threadIdy.y == 0) {
//             image_shared[]
//         }
//         float value = 0.0f; 
//         // left horizontal sweep
//         for(int i = -1; i <= 1; i++) {
//             const float weight = i != 0 ? 0.125f : 0.25f;
//             const int x = MAX_F(pixelX - 1, 0);
//             const int bx = MAX_F(threadIdx.x - 1, 0);
//             const int y = MAX_F(pixelY + i, 0);
//             y = MIN_F(y, height);
//             const int by = MAX_F(threadIdx.y + i, 0);
//             by = MIN_F(threadIdx.y + i, BLOCK_WIDTH);
//             // Nearest padding
//             const float image_value;
//             if(pixelX == 0 || pixelY + i >= height || pixelY + i < 0) {
//                 image_value = image_shared[blockIdx.y * BLOCK_WIDTH + blockIdx.x];
//             }
//             else if(blockIdx.x == 0 || blockIdx.y + i < 0 || blockIdx.y + i >= BLOCK_HEIGHT) {
//                 image_value = image_data[y * width + x];
//             }
//             else {
//                 image_value = image_shared[x];
//             }
//             value += weight * image_value;
//         }

//         // right horizontal sweep
//         for(int i = -1; i <= 1; i++) {
//             const float weight = i != 0 ? -0.125f : -0.25f;
//             // Nearest padding
//             const float image_value;
//             if(pixelX + 1 >= width || pixelY + i >= height || pixelY + i < 0) {
//                 // TODO: technically use the "nearest", but since this is a 3x3 kernel
//                 // the nearest is always the current pixel under consideration.
//                 image_value = image_shared[blockIdx.y * BLOCK_WIDTH + blockIdx.x];
//             }
//             else if(blockIdx.x + 1 >= BLOCK_WIDTH || blockIdx.y + i < 0 || blockIdx.y + i >= BLOCK_HEIGHT) {
//                 image_value = image_data[(pixelY + i) * width + pixelX + 1];
//             }
//             else {
//                 image_value = image_shared[(blockIdx.y + i) * BLOCK_WIDTH + blockIdx.x + 1];
//             }
//             value += weight * image_value;
//         }
//         // Global write
//         image_x_grad[pixelY * width + pixelX] = value;
//     }
// }

// __global__
// void sobel_x_kernel_shared(int height, int width) {
//     __shared__ float image_shared[THREADS_PER_BLOCK];

//     const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
//     const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint pixelSL = blockIdx.y * BLOCK_WIDTH + blockIdx.x;

//     if(pixelX < width && pixelY < height) {
//         image_shared[pixelSL] = image_data[pixelY * width + pixelX];
//         float value = 0.0f; 
//         // left horizontal sweep
//         for(int i = -1; i <= 1; i++) {
//             const float weight = i != 0 ? 0.125f : 0.25f;
//             const int x = MAX_F(pixelX - 1, 0);
//             const int bx = MAX_F(threadIdx.x - 1, 0);
//             const int y = MAX_F(pixelY + i, 0);
//             y = MIN_F(y, height);
//             const int by = MAX_F(threadIdx.y + i, 0);
//             by = MIN_F(threadIdx.y + i, BLOCK_WIDTH);
//             // Nearest padding
//             const float image_value;
//             if(pixelX == 0 || pixelY + i >= height || pixelY + i < 0) {
//                 image_value = image_shared[blockIdx.y * BLOCK_WIDTH + blockIdx.x];
//             }
//             else if(blockIdx.x == 0 || blockIdx.y + i < 0 || blockIdx.y + i >= BLOCK_HEIGHT) {
//                 image_value = image_data[y * width + x];
//             }
//             else {
//                 image_value = image_shared[x];
//             }
//             value += weight * image_value;
//         }

//         // right horizontal sweep
//         for(int i = -1; i <= 1; i++) {
//             const float weight = i != 0 ? -0.125f : -0.25f;
//             // Nearest padding
//             const float image_value;
//             if(pixelX + 1 >= width || pixelY + i >= height || pixelY + i < 0) {
//                 // TODO: technically use the "nearest", but since this is a 3x3 kernel
//                 // the nearest is always the current pixel under consideration.
//                 image_value = image_shared[blockIdx.y * BLOCK_WIDTH + blockIdx.x];
//             }
//             else if(blockIdx.x + 1 >= BLOCK_WIDTH || blockIdx.y + i < 0 || blockIdx.y + i >= BLOCK_HEIGHT) {
//                 image_value = image_data[(pixelY + i) * width + pixelX + 1];
//             }
//             else {
//                 image_value = image_shared[(blockIdx.y + i) * BLOCK_WIDTH + blockIdx.x + 1];
//             }
//             value += weight * image_value;
//         }
//         // Global write
//         image_x_grad[pixelY * width + pixelX] = value;
//     }
// }

void init_cuda() {
    cudaSetDevice(0);
    cudaFree(0);

    // Allocate space for our input images and intermediate results
    cudaMalloc(&c_image_data, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_image_x_grad, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_image_y_grad, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMemcpyToSymbol(image_data, &c_image_data, sizeof(float *));
    cudaMemcpyToSymbol(image_x_grad, &c_image_x_grad, sizeof(float *));
    cudaMemcpyToSymbol(image_y_grad, &c_image_y_grad, sizeof(float *));
}

void free_cuda() {
    cudaFree(c_image_data);
    cudaFree(c_image_x_grad);
    cudaFree(c_image_y_grad);
}

void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}