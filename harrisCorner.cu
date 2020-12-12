#include <stdio.h>
#include <tuple>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "CycleTimer.h"
#include "config.h"

#include <chrono>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)
#define TILE_WIDTH 30
#define TILE_HEIGHT 30

#define SHARED_PADDING (WINDOW_SIZE / 2) + (GRAD_SIZE / 2) 

#define dceil(a, b) ((a) % (b) != 0 ? ((a) / (b) + 1) : ((a) / (b)))

#define INSIDE 2
#define OUTSIDE 1
#define UNDEFINED 0

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std::chrono;

// Constant pointer to the image data
__constant__ float *image_data;
__constant__ float *image_x_grad;
__constant__ float *image_y_grad;
__constant__ float *image_xy_grad;
__constant__ float *ixx;
__constant__ float *iyy;
__constant__ float *ixy;
__constant__ int *scan_keys;
__constant__ int *scan_keys_T;

__constant__ float gaussian_h[3] = {1.0f, 2.0f, 1.0f};
__constant__ float gaussian_v[3] = {1.0/16.0, 2.0/16.0, 1.0/16.0};
__constant__ float sobel_h[3] = {-1.0f, 0.0f, -1.0f};
__constant__ float sobel_v[3] = {1.0f, 2.0f, 1.0f};

__constant__ int apron_mapping[132][2] = 
    {
        {-1, -1}, {-1, 0}, {-1, 1}, {-1, 2}, {-1, 3}, {-1, 4}, {-1, 5}, {-1, 6}, {-1, 7}, {-1, 8}, {-1, 9}, {-1, 10}, 
        {-1, 11}, {-1, 12}, {-1, 13}, {-1, 14}, {-1, 15}, {-1, 16}, {-1, 17}, {-1, 18}, {-1, 19}, {-1, 20}, {-1, 21}, 
        {-1, 22}, {-1, 23}, {-1, 24}, {-1, 25}, {-1, 26}, {-1, 27}, {-1, 28}, {-1, 29}, {-1, 30}, {-1, 31}, {-1, 32}, 
        {0, -1}, {0, 32}, {1, -1}, {1, 32}, {2, -1}, {2, 32}, {3, -1}, {3, 32}, {4, -1}, {4, 32}, {5, -1}, {5, 32}, 
        {6, -1}, {6, 32}, {7, -1}, {7, 32}, {8, -1}, {8, 32}, {9, -1}, {9, 32}, {10, -1}, {10, 32}, {11, -1}, {11, 32}, 
        {12, -1}, {12, 32}, {13, -1}, {13, 32}, {14, -1}, {14, 32}, {15, -1}, {15, 32}, {16, -1}, {16, 32}, {17, -1}, {17, 32}, 
        {18, -1}, {18, 32}, {19, -1}, {19, 32}, {20, -1}, {20, 32}, {21, -1}, {21, 32}, {22, -1}, {22, 32}, {23, -1}, {23, 32}, 
        {24, -1}, {24, 32}, {25, -1}, {25, 32}, {26, -1}, {26, 32}, {27, -1}, {27, 32}, {28, -1}, {28, 32}, {29, -1}, {29, 32}, 
        {30, -1}, {30, 32}, {31, -1}, {31, 32}, {32, -1}, {32, 0}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5}, {32, 6}, 
        {32, 7}, {32, 8}, {32, 9}, {32, 10}, {32, 11}, {32, 12}, {32, 13}, {32, 14}, {32, 15}, {32, 16}, {32, 17}, {32, 18}, 
        {32, 19}, {32, 20}, {32, 21}, {32, 22}, {32, 23}, {32, 24}, {32, 25}, {32, 26}, {32, 27}, {32, 28}, {32, 29}, {32, 30},
        {32, 31}, {32, 32}
    };

typedef struct point {
    short row;
    short col;
} point_t;

float *c_image_data;
float *c_image_x_grad;
float *c_image_y_grad;
float *c_image_xy_grad;
float *c_ixx;
float *c_iyy;
float *c_ixy;
int *c_scan_keys;
int *c_scan_keys_T;
int *c_output;
point_t *c_compressed_output;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

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
        float x_grad = 0.125f * (image_data[(pY + dyu) * w + pX + dxr] - image_data[(pY + dyu) * w + pX + dxl] + 
                                 image_data[(pY + dyd) * w + pX + dxr] - image_data[(pY + dyd) * w + pX + dxl] ) + 
                        0.25f * (image_data[pY * w + pX + dxr] - image_data[pY * w + pX + dxl]);
        float y_grad = 0.125f * (image_data[(pY + dyd) * w + pX + dxl] - image_data[(pY + dyu) * w + pX + dxl] + 
                                 image_data[(pY + dyd) * w + pX + dxr] - image_data[(pY + dyu) * w + pX + dxr]) + 
                        0.25f * (image_data[(pY + dyd) * w + pX] - image_data[(pY + dyu) * w + pX]);

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
        float x_grad = -0.125f * (image_data[(ppY - 1) * w + ppX - 1] - image_data[(ppY - 1) * w + ppX + 1] + 
                                 image_data[(ppY + 1) * w + ppX - 1] - image_data[(ppY + 1) * w + ppX + 1]) +
                        -0.25f * (image_data[ppY * w + ppX - 1] - image_data[ppY * w + ppX + 1]);
        float y_grad = -0.125f * image_data[(ppY - 1) * w + ppX - 1] + -0.25f * image_data[(ppY - 1) * w + ppX] + -0.125f * image_data[(ppY - 1) * w + ppX + 1] + 
                    0.125f * image_data[(ppY + 1) * w + ppX - 1] + 0.25f * image_data[(ppY + 1) * w + ppX] + 0.125f * image_data[(ppY + 1) * w + ppX + 1];

        image_x_grad[pY * width + pX] = x_grad;
        image_y_grad[pY * width + pX] = y_grad;
    }
}

// expects padding
__global__
void sobel_kernel_shared(int height, int width) {
    __shared__ float image_shared[THREADS_PER_BLOCK];
    const uint pY = blockIdx.y * (TILE_WIDTH) + threadIdx.y;
    const uint pX = blockIdx.x * (TILE_WIDTH) + threadIdx.x;
    const uint ty = threadIdx.y;
    const uint tx = threadIdx.x;

    if(pX <= width + 1 && pY <= height + 1) {
        image_shared[threadIdx.y * BLOCK_WIDTH + threadIdx.x] = image_data[pY * (width + 2) + pX];
    }
    __syncthreads();
    if(pX >= 1 && pX <= width && pY >= 1 && pY <= height) {
        if(tx > 0 && tx < BLOCK_WIDTH - 1 && ty > 0 && ty < BLOCK_WIDTH - 1) {
            float x_grad = -0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx - 1] + -0.25f * image_shared[ty * BLOCK_WIDTH + tx - 1] + -0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx - 1] +
                        0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx + 1] + 0.25f * image_shared[ty * BLOCK_WIDTH + tx + 1] + 0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx + 1];
            float y_grad = -0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx - 1] + -0.25f * image_shared[(ty - 1) * BLOCK_WIDTH + tx] + -0.125f * image_shared[(ty - 1) * BLOCK_WIDTH + tx + 1] + 
                        0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx - 1] + 0.25f * image_shared[(ty + 1) * BLOCK_WIDTH + tx] + 0.125f * image_shared[(ty + 1) * BLOCK_WIDTH + tx + 1];

            image_x_grad[(pY - 1) * width + pX - 1] = x_grad;
            image_y_grad[(pY - 1) * width + pX - 1] = y_grad;
        }
    }
}

__global__
void sobel_kernel_shared_no_apron(int height, int width) {
    __shared__ float image_shared[THREADS_PER_BLOCK + 4 * (BLOCK_WIDTH + 1)];
    const int pY = blockIdx.y * blockDim.y + threadIdx.y;
    const int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const int ppY = pY + 1;
    const int ppX = pX + 1;
    const int ty = threadIdx.y + 1;
    const int tx = threadIdx.x + 1;
    const int li = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
    const int S_WIDTH = BLOCK_WIDTH + 2;

    image_shared[ty * S_WIDTH + tx] = image_data[ppY * (width + 2) + ppX];
    if(li < 132) {
        const int dy = apron_mapping[li][0];
        const int dx = apron_mapping[li][1];
        image_shared[(dy + 1) * S_WIDTH + dx + 1] = image_data[(ppY - threadIdx.y + dy) * (width + 2) + ppX - threadIdx.x + dx];
    }
    __syncthreads();

    if(pX < width && pY < height) {
        float x_grad = -0.125f * image_shared[(ty - 1) * S_WIDTH + tx - 1] + -0.25f * image_shared[ty * S_WIDTH + tx - 1] + -0.125f * image_shared[(ty + 1) * S_WIDTH + tx - 1] +
                    0.125f * image_shared[(ty - 1) * S_WIDTH + tx + 1] + 0.25f * image_shared[ty * S_WIDTH + tx + 1] + 0.125f * image_shared[(ty + 1) * S_WIDTH + tx + 1];
        float y_grad = -0.125f * image_shared[(ty - 1) * S_WIDTH + tx - 1] + -0.25f * image_shared[(ty - 1) * S_WIDTH + tx] + -0.125f * image_shared[(ty - 1) * S_WIDTH + tx + 1] + 
                    0.125f * image_shared[(ty + 1) * S_WIDTH + tx - 1] + 0.25f * image_shared[(ty + 1) * S_WIDTH + tx] + 0.125f * image_shared[(ty + 1) * S_WIDTH + tx + 1];

        image_x_grad[pY * width + pX] = x_grad;
        image_y_grad[pY * width + pX] = y_grad;
    }
}

// BROKEN
__global__
void sobel_kernel_shared_no_apron_stream(int height, int width, int row) {
    __shared__ float image_shared[THREADS_PER_BLOCK + 4 * (BLOCK_WIDTH + 1)];
    const int pY = row + threadIdx.y;
    const int pX = blockIdx.x * blockDim.x + threadIdx.x;
    const int ppY = pY + 1;
    const int ppX = pX + 1;
    const int ty = threadIdx.y + 1;
    const int tx = threadIdx.x + 1;
    const int li = threadIdx.y * BLOCK_WIDTH + threadIdx.x;
    const int S_WIDTH = BLOCK_WIDTH + 2;

    image_shared[ty * S_WIDTH + tx] = image_data[ppY * (width + 2) + ppX];
    if(li < 132) {
        const int dy = apron_mapping[li][0];
        const int dx = apron_mapping[li][1];
        image_shared[(dy + 1) * S_WIDTH + dx + 1] = image_data[(ppY - threadIdx.y + dy) * (width + 2) + ppX - threadIdx.x + dx];
    }
    __syncthreads();

    if(pX < width && pY < height) {
        float x_grad = -0.125f * image_shared[(ty - 1) * S_WIDTH + tx - 1] + -0.25f * image_shared[ty * S_WIDTH + tx - 1] + -0.125f * image_shared[(ty + 1) * S_WIDTH + tx - 1] +
                    0.125f * image_shared[(ty - 1) * S_WIDTH + tx + 1] + 0.25f * image_shared[ty * S_WIDTH + tx + 1] + 0.125f * image_shared[(ty + 1) * S_WIDTH + tx + 1];
        float y_grad = -0.125f * image_shared[(ty - 1) * S_WIDTH + tx - 1] + -0.25f * image_shared[(ty - 1) * S_WIDTH + tx] + -0.125f * image_shared[(ty - 1) * S_WIDTH + tx + 1] + 
                    0.125f * image_shared[(ty + 1) * S_WIDTH + tx - 1] + 0.25f * image_shared[(ty + 1) * S_WIDTH + tx] + 0.125f * image_shared[(ty + 1) * S_WIDTH + tx + 1];

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
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image_data[(ppixelY + i) * (2 * pwidth + width) + ppixelX - 1];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
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
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image_data[(ppixelY - 1) * (2 * pwidth + width) + ppixelX + i];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
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

// convolves a single row for streaming
__global__
void sobel_kernel_seperable_horizontal(float *image, float *output, int height, int width, int row) {
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppX = pX + 1;

    if(ppX < width - 1) {
        float *image_row = &image[row * (width + 1)];
        float value = -1.0f * image_row[ppX - 1] + 0.0f * image_row[ppX] + 1.0f * image_row[ppX + 1];
        output[row * width + pX] = value;
    }
}

__global__
void blur_kernel_seperable_horizontal(float *image, float *output, int height, int width, int row,
                                      float h1, float h2, float h3) {
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppX = pX + 1;

    if(ppX <= width) {
        float *image_row = &image[row * (width + 2)];
        float value = image_row[ppX + 1] - image_row[ppX - 1]; //h2 * image_row[ppX] + h3 * image_row[ppX + 1];
        output[row * width + pX] = value;
    }
}

__global__ 
void blur_kernel_seperable_verticle(float *horz, float *output, int height, int width, int row,
                                    float v1, float v2, float v3) {
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    if(pX < width) {
        float value = v1 * horz[(row - 1) * width + pX] + v2 * horz[row * width + pX] + v3 * horz[(row + 1) * width + pX];
        output[(row - 1) * width + pX] = value;
    }
}

__global__
void transpose_kernel(float *input, float *output, int height, int width) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    if(pY < height && pX < width) {
        output[pX * height + pY] = input[pY * width + pX];
    }
}

__global__
void build_scan_keys_kernel(int height, int width) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    if(pX < width && pY < height) {
        scan_keys[pY * width + pX] = pY;
        scan_keys_T[pX * height + pY] = pX;
    }
}

__global__
void mult(float *input1, float *input2, float *output, int height, int width) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    if(pY < height && pX < width) {
        output[pY * width + pX] = input1[pY * width + pX] * input2[pY * width + pX];
    }
}

void compute_integral_image(int height, int width) {
    // build scanKeys
    dim3 threadBlock(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    build_scan_keys_kernel<<<grid, threadBlock>>>(height, width);

    mult<<<grid, threadBlock>>>(c_image_x_grad, c_image_y_grad, c_image_xy_grad, height, width);
    mult<<<grid, threadBlock>>>(c_image_x_grad, c_image_x_grad, c_image_x_grad, height, width);
    mult<<<grid, threadBlock>>>(c_image_y_grad, c_image_y_grad, c_image_y_grad, height, width);

    thrust::device_ptr<float> tixx = thrust::device_pointer_cast(c_ixx);
    thrust::device_ptr<float> tiyy = thrust::device_pointer_cast(c_iyy);
    thrust::device_ptr<float> tixy = thrust::device_pointer_cast(c_ixy);
    thrust::device_ptr<float> tdxx = thrust::device_pointer_cast(c_image_x_grad);
    thrust::device_ptr<float> tdyy = thrust::device_pointer_cast(c_image_y_grad);
    thrust::device_ptr<float> tdxy = thrust::device_pointer_cast(c_image_xy_grad);
    thrust::device_ptr<int> keys = thrust::device_pointer_cast(c_scan_keys);
    thrust::device_ptr<int> tkeys = thrust::device_pointer_cast(c_scan_keys_T);
    // do first scan
    thrust::exclusive_scan_by_key(keys, keys + height * width, tdxx, tixx);
    thrust::exclusive_scan_by_key(keys, keys + height * width, tdyy, tiyy);
    thrust::exclusive_scan_by_key(keys, keys + height * width, tdxy, tixy);

    // transpose
    transpose_kernel<<<grid, threadBlock>>>(c_ixx, c_image_x_grad, height, width);
    transpose_kernel<<<grid, threadBlock>>>(c_iyy, c_image_y_grad, height, width);
    transpose_kernel<<<grid, threadBlock>>>(c_ixy, c_image_xy_grad, height, width);

    // second scan, in place
    thrust::exclusive_scan_by_key(tkeys, tkeys + height * width, tdxx, tdxx);
    thrust::exclusive_scan_by_key(tkeys, tkeys + height * width, tdyy, tdyy);
    thrust::exclusive_scan_by_key(tkeys, tkeys + height * width, tdxy, tdxy);

    // tranpose 
    dim3 gridT(dceil(height, BLOCK_HEIGHT), dceil(width, BLOCK_WIDTH));
    transpose_kernel<<<gridT, threadBlock>>>(c_image_x_grad, c_ixx, width, height);
    transpose_kernel<<<gridT, threadBlock>>>(c_image_y_grad, c_iyy, width, height);
    transpose_kernel<<<gridT, threadBlock>>>(c_image_xy_grad, c_ixy, width, height);
    cudaDeviceSynchronize();
}

__global__
void nms_naive(float *cornerness, unsigned char *output, int height, int width) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;

    float value = cornerness[pY * width + pX];
    int filtered = (value > 0.05f) ? 1 : 0;
    if(pX >= 1 && pX < (width - 1) && pY >= 1 && pY < (height - 1)) {
        for(int i = -1; i < 1; i++) {
            int row = pY + i;
            for(int j = -1; j < 1; j++) {
                int col = pX + j;

                if(cornerness[row * width + col] > value) {
                    filtered = 0;
                    break;
                }
            }
        }
        output[pY * width + pX] = filtered;
    }
    else {
        output[pY * width + pX] = 0;
    }
}

__global__
void compress_kernel(int *psum, point_t *output, int height, int width) {
    const uint pY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint li = pY * width + pX;

    if(pX < width && pY < height) {
        if(li > 0) {
            if(psum[li] > psum[li - 1]) {
                const uint idx = psum[li] - 1;
                output[idx].row = pY;
                output[idx].col = pX;
            }
        }
        else {
            if(psum[li] >= 1) {
                output[0].row = pY;
                output[0].col = pX;
            }
        }
    }
}

int compress_output(int *input, point_t *output, int height, int width) {
    thrust::device_ptr<int> ti = thrust::device_pointer_cast(input);
    thrust::inclusive_scan(thrust::device, ti, ti + (height * width), ti);
    int len;
    cudaMemcpy(&len, &input[height * width - 1], sizeof(int), cudaMemcpyDeviceToHost);

    dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
    compress_kernel<<<grid, threadBlock>>>(input, output, height, width);
    cudaDeviceSynchronize();
    return len;
}

/* input is a grayscale image of size height by width */
std::tuple<long int, long int, long int> harrisCornerDetectorStaged(float *pinput, int *output, int height, int width) {
    auto start_time = high_resolution_clock::now();
    const size_t padding = TOTAL_PADDING_SIZE;

    const int input_image_size = sizeof(float) * (height + 2 * padding) * (width + 2 * padding);
    const int grad_image_width = width + 2 * (padding - 1);
    const int grad_image_height = height + 2 * (padding - 1); 
    const int output_image_size = sizeof(unsigned char) * height * width;

    // Copy input arrays to the GPU
    auto mem_start_time1 = high_resolution_clock::now();
    cudaMemcpy(c_image_data, pinput, input_image_size, cudaMemcpyHostToDevice);
    auto mem_end_time1 = high_resolution_clock::now();

    const dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    const dim3 grid_grad (dceil(grad_image_width, BLOCK_WIDTH), dceil(grad_image_height, BLOCK_HEIGHT));
    const dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
    auto kernel_start_time = high_resolution_clock::now();
    auto sobel_start_time = high_resolution_clock::now();
    sobel_kernel_shared_no_apron<<<grid_grad, threadBlock>>>(grad_image_height, grad_image_width);
    //cudaDeviceSynchronize();
    auto sobel_end_time = high_resolution_clock::now();
    cornerness_kernel<<<grid, threadBlock>>>(height, width);
    //cudaDeviceSynchronize();
    nms_naive<<<grid, threadBlock>>>(c_image_data, (unsigned char *)c_output, height, width);
    
    //int len = compress_output(c_output, c_compressed_output, height, width);
    cudaDeviceSynchronize();
    auto kernel_end_time = high_resolution_clock::now();

    //printf("%i\n", len);

    // Copy result to CPU
    auto mem_start_time2 = high_resolution_clock::now();
    cudaMemcpy(output, c_output, output_image_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(output, c_compressed_output, len * sizeof(point_t), cudaMemcpyDeviceToHost);
    auto mem_end_time2 = high_resolution_clock::now();
    auto end_time = high_resolution_clock::now();

    return std::make_tuple(duration_cast<microseconds>(kernel_end_time - kernel_start_time).count(), 
                                    duration_cast<microseconds>(mem_end_time1 - mem_start_time1).count(), 
                                    duration_cast<microseconds>(mem_end_time2 - mem_start_time2).count());
}

void benchSobel(float *pinput, float *output_x, float *output_y, int height, int width, int mode) {
    const int iter = 10;
    switch(mode) {
        case 0: { // sobel padding not shared 
            // 1 padding
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            auto mem_start_time = high_resolution_clock::now();
            cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            auto mem_end_time = high_resolution_clock::now();
            long int mem_dur = duration_cast<microseconds>(mem_end_time - mem_start_time).count();
            dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                sobel_x_kernel<<<grid, threadBlock>>>(height, width, 1, 1);
                sobel_y_kernel<<<grid, threadBlock>>>(height, width, 1, 1);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_x_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            printf("Memory copy took %lu us\n", mem_dur);
            printf("Total time: %lu us\n", (count / iter) + mem_dur);
            break;
        }
        case 1: { // sobel padding shared
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, TILE_WIDTH), dceil(height, TILE_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                sobel_kernel_shared<<<grid, threadBlock>>>(height, width);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_x_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            break;
        }
        case 2: { // sobel padding shared no apron
            // 1 padding
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                sobel_kernel_shared_no_apron<<<grid, threadBlock>>>(height, width);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_x_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            break;
        }
        case 3: { // sobel padding shared no apron stream, BROKEN
            // 1 padding
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            //cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, BLOCK_WIDTH), 1);
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
            const unsigned int nStreams = dceil(image_height - 2, BLOCK_HEIGHT);
            printf("nStreams: %i\n", nStreams);

            cudaStream_t stream[nStreams];

            for(int i = 0; i < nStreams; i++) {
                cudaStreamCreate(&stream[i]);
            }

            cudaMemcpy(c_image_data, pinput, sizeof(float) * image_width * 2, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                for(int j = 0; j < nStreams; j++) {
                    int nRows = MIN(BLOCK_HEIGHT, image_height - 2 - j * BLOCK_HEIGHT);
                    //printf("%i\n", nRows);
                    float *ptrd = &c_image_data[(j * BLOCK_HEIGHT + 2) * image_width];
                    float *ptrs = &pinput[(j * BLOCK_HEIGHT + 2) * image_width];
                    cudaMemcpyAsync(ptrd, ptrs, nRows * image_width * sizeof(float), cudaMemcpyHostToDevice, stream[j]);
                    sobel_kernel_shared_no_apron_stream<<<grid, threadBlock, 0, stream[j]>>>(height, width, j * BLOCK_HEIGHT + 1);
                }
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_data, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            break;
        }
        case 4: { // sobel no padding no shared
            const int image_height = height;
            const int image_width = width;
            cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                sobel_kernel<<<grid, threadBlock>>>(height, width);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_x_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            break;
        }
        case 5: { // streaming approach with separable kernels 
            // 1 padding
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            //cudaMemcpy(c_image_data, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, 1024));
            dim3 threadBlock (1024);

            cudaStream_t stream[image_height];

            for(int i = 0; i < image_height; i++) {
                cudaStreamCreate(&stream[i]);
            }

            long int count = 0;
            for(int i = 0; i < 1; i++) {
                auto start_time = high_resolution_clock::now();
                for(int row = 0; row < image_height; row++) {
                    float *ptrd = &c_image_data[row * image_width];
                    float *ptrs = &pinput[row * image_width];
                    cudaMemcpyAsync(ptrd, ptrs, image_width * sizeof(float), cudaMemcpyHostToDevice, stream[row]);
                    blur_kernel_seperable_horizontal<<<grid, threadBlock, 0, stream[row]>>>(c_image_data, c_image_x_grad, height, width, row, -1.0f, 0.0f, 1.0f);
                    if(row >= 3) {
                        blur_kernel_seperable_verticle<<<grid, threadBlock, 0, stream[row - 2]>>>(c_image_x_grad, c_image_data, height, width, row - 2, 1.0/8.0f, 2.0/8.0f, 1.0/8.0f);
                    }
                }
                //blur_kernel_seperable_verticle<<<grid, threadBlock>>>(c_image_x_grad, c_image_data, height, width, height, 1.0/8.0f, 2.0/8.0f, 1.0/8.0f);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            cudaMemcpy(output_x, c_image_data, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_y, c_image_y_grad, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

            printf("Sobel Kernel took %ld us\n", count / iter);
            break;
        }
        default: {
            printf("Unknown mode\n");
        }
    }
}

void benchCorner(float *pinput, float *output_c, float *output_ix, float *output_iy, float *output_ixy, int height, int width, int mode) {
    const int iter = 10;

    switch(mode) {
        case 0: { // naive approach to computing the cornerness response
            // 1 padding
            const int image_height = (2 + height);
            const int image_width = (2 + width);
            cudaMemcpy(c_image_x_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            cudaMemcpy(c_image_y_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            //cudaMemcpy(c_image_xy_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

            long int count = 0;
            for(int i = 0; i < iter; i++) {
                auto start_time = high_resolution_clock::now();
                cornerness_kernel<<<grid, threadBlock>>>(height, width);
                cudaDeviceSynchronize();
                auto end_time = high_resolution_clock::now();
                count += duration_cast<microseconds>(end_time - start_time).count();
            }

            if(output_c != NULL) {
                cudaMemcpy(output_c, c_image_data, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            }

            printf("Corner Kernel took %ld us\n", count / iter);
            break;
        }
        case 1: { // integral image
           // 1 padding
           const int image_height = (2 + height);
           const int image_width = (2 + width);
           cudaMemcpy(c_image_x_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
           cudaMemcpy(c_image_y_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
           cudaMemcpy(c_image_xy_grad, pinput, sizeof(float) * image_height * image_width, cudaMemcpyHostToDevice);
           dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
           dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);

           long int count = 0;
           for(int i = 0; i < iter; i++) {
               auto start_time = high_resolution_clock::now();
               compute_integral_image(image_height, image_width);
               cudaDeviceSynchronize();
               auto end_time = high_resolution_clock::now();
               count += duration_cast<microseconds>(end_time - start_time).count();
           }

           if(output_c != NULL) {
               cudaMemcpy(output_c, c_image_data, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
           }
           if(output_ix != NULL) {
                cudaMemcpy(output_ix, c_ixx, sizeof(float) * image_height * image_width, cudaMemcpyDeviceToHost);
           }
           if(output_iy != NULL) {
            cudaMemcpy(output_iy, c_iyy, sizeof(float) * image_height * image_width, cudaMemcpyDeviceToHost);
           }
           if(output_ixy != NULL) {
            cudaMemcpy(output_ixy, c_ixy, sizeof(float) * image_height * image_width, cudaMemcpyDeviceToHost);
           }

           printf("Corner Kernel took %ld us\n", count / iter);
           break; 
        }
    }
}

void benchNMS(float *input, int *output, int height, int width, int mode) {
    const int iter = 10;

    switch(mode) {
        case 0: { // naive implementation 
            cudaMemcpy(c_image_data, input, sizeof(float) * width * height, cudaMemcpyHostToDevice);
            dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
            dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
            long int count = 0;
            for(int i = 0; i < iter; i++) {
               auto start_time = high_resolution_clock::now();
               nms_naive<<<grid, threadBlock>>>(c_image_data, (unsigned char *)c_ixx, height, width);
               cudaDeviceSynchronize();
               auto end_time = high_resolution_clock::now();
               count += duration_cast<microseconds>(end_time - start_time).count();
            }

            if(output != NULL) {
                cudaMemcpy(output, c_ixx, sizeof(float) * height * width, cudaMemcpyDeviceToHost);
            }

            printf("NMS took %ld us\n", count / iter);
        }
    }
}

void init_cuda() {
    cudaSetDevice(0);
    cudaFree(0);

    // Allocate space for our input images and intermediate results
    cudaMalloc(&c_image_data, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_image_x_grad, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_image_y_grad, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_image_xy_grad, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_ixx, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_iyy, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_ixy, sizeof(float) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_scan_keys, sizeof(int) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_scan_keys_T, sizeof(int) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_output, sizeof(int) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMalloc(&c_compressed_output, sizeof(point_t) * MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH);
    cudaMemcpyToSymbol(image_data, &c_image_data, sizeof(float *));
    cudaMemcpyToSymbol(image_x_grad, &c_image_x_grad, sizeof(float *));
    cudaMemcpyToSymbol(image_y_grad, &c_image_y_grad, sizeof(float *));
    cudaMemcpyToSymbol(image_xy_grad, &c_image_xy_grad, sizeof(float *));
    cudaMemcpyToSymbol(ixx, &c_ixx, sizeof(float *));
    cudaMemcpyToSymbol(iyy, &c_iyy, sizeof(float *));
    cudaMemcpyToSymbol(ixy, &c_ixy, sizeof(float *));
    cudaMemcpyToSymbol(scan_keys, &c_scan_keys, sizeof(float *));
    cudaMemcpyToSymbol(scan_keys_T, &c_scan_keys_T, sizeof(int *));
}

void free_cuda() {
    cudaFree(c_image_data);
    cudaFree(c_image_x_grad);
    cudaFree(c_image_y_grad);
    cudaFree(c_image_xy_grad);
    cudaFree(c_ixx);
    cudaFree(c_ixy);
    cudaFree(c_iyy);
    cudaFree(c_scan_keys);
    cudaFree(c_scan_keys_T);
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

void deviceQuery ()
{
  cudaDeviceProp prop;
  int nDevices=0, i;
  cudaError_t ierr;

  ierr = cudaGetDeviceCount(&nDevices);
  if (ierr != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierr)); }



  for( i = 0; i < nDevices; ++i )
  {
     ierr = cudaGetDeviceProperties(&prop, i);
     printf("Device number: %d\n", i);
     printf("  Device name: %s\n", prop.name);
     printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);
     
     printf("  Clock Rate: %d kHz\n", prop.clockRate);
     printf("  Total SMs: %d \n", prop.multiProcessorCount);
     printf("  Shared Memory Per SM: %lu bytes\n", prop.sharedMemPerMultiprocessor);
     printf("  Registers Per SM: %d 32-bit\n", prop.regsPerMultiprocessor);
     printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
     printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
     printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
     printf("  Memory Clock Rate: %d kHz\n\n", prop.memoryClockRate);
     
     
     printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
     printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
     printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
     printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);

     printf("  Max blocks in X-dimension of grid: %d\n", prop.maxGridSize[0]);
     printf("  Max blocks in Y-dimension of grid: %d\n", prop.maxGridSize[1]);
     printf("  Max blocks in Z-dimension of grid: %d\n\n", prop.maxGridSize[2]);     
     
     printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
     printf("  Registers Per Block: %d 32-bit\n", prop.regsPerBlock);
     printf("  Warp size: %d\n\n", prop.warpSize);

  }
}