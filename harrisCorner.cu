#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "config.h"

#include <chrono>

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define THREADS_PER_BLOCK (BLOCK_WIDTH * BLOCK_HEIGHT)

#define SHARED_PADDING (WINDOW_SIZE / 2) + (GRAD_SIZE / 2) 

#define dceil(a, b) ((a) % (b) != 0 ? ((a) / (b) + 1) : ((a) / (b)))

#define INSIDE (1.0f)
#define OUTSIDE (0.0f)
#define UNDEFINED (-1.0f)

using namespace std::chrono;

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
void sobel_x_kernel(float *image, float *output, int height, int width, int pheight, int pwidth) {
    const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppixelY = pixelY + pheight;
    const uint ppixelX = pixelX + pwidth;

    if(pixelX < width && pixelY < height) {
        float value = 0.0f; 
        // left horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
            value += weight * image[(ppixelY + i) * (2 * pwidth + width) + ppixelX - 1];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image[(ppixelY + i) * (2 * pwidth + width) + ppixelX + 1];
        }
        output[pixelY * width + pixelX] = value;
    }
}

__global__
void sobel_y_kernel(float *image, float *output, int height, int width, int pheight, int pwidth) {
    const uint pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ppixelY = pixelY + pheight;
    const uint ppixelX = pixelX + pwidth;

    if(pixelX < width && pixelY < height) {
        float value = 0.0f; 
        // left horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? 0.125f : 0.25f;
            value += weight * image[(ppixelY - 1) * (2 * pwidth + width) + ppixelX + i];
        }

        // right horizontal sweep
        for(int i = -1; i <= 1; i++) {
            const float weight = i != 0 ? -0.125f : -0.25f;
            value += weight * image[(ppixelY + 1) * (2 * pwidth + width) + ppixelX + i];
        }
        output[pixelY * width + pixelX] = value;
    }
}

__global__
void cornerness_kernel(float *x_grad, float *y_grad, float *output, int height, int width) {
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
                gxx += x_grad[pos] * x_grad[pos];
                gyy += y_grad[pos] * y_grad[pos];
                gxy += x_grad[pos] * y_grad[pos];
            }
        }
        const float det = gxx * gyy - gxy * gxy;
        const float trace = gxx + gyy;
        output[pixelY * width + pixelX] = det - K * trace * trace;
    }
}

/* algorithm borrowed from: http://www.bmva.org/bmvc/2008/papers/45.pdf */
__global__
void non_maximum_suppression_kernel(float *cornerness, float *input, float *output, int height, int width, bool *done) {
    
}

/* input is a grayscale image of size height by width */
void harrisCornerDetectorStaged(float *pinput, float *output, int height, int width) {
    auto start_time = high_resolution_clock::now();
    const size_t padding = TOTAL_PADDING_SIZE;
    // Create space for the image on the GPU
    float *device_x_grad;
    float *device_y_grad;
    float *device_input;
    float *device_output;

    const int input_image_size = sizeof(float) * (height + 2 * padding) * (width + 2 * padding);
    const int grad_image_width = width + 2 * (padding - GRAD_PADDING_SIZE);
    const int grad_image_height = height + 2 * (padding - GRAD_PADDING_SIZE);
    const int grad_image_size = sizeof(float) * grad_image_height * grad_image_width;
    const int output_image_size = sizeof(float) * height * width;
    auto mem_start_time1 = high_resolution_clock::now();
    cudaMalloc(&device_input, input_image_size);
    auto mem_end_time1 = high_resolution_clock::now();

    cudaMalloc(&device_input, input_image_size);
    cudaMalloc(&device_x_grad, grad_image_size);
    cudaMalloc(&device_y_grad, grad_image_size);
    cudaMalloc(&device_output, output_image_size);

    // Copy input arrays to the GPU
    auto mem_start_time2 = high_resolution_clock::now();
    cudaMemcpy(device_input, pinput, input_image_size, cudaMemcpyHostToDevice);
    auto mem_end_time2 = high_resolution_clock::now();

    const dim3 grid (dceil(width, BLOCK_WIDTH), dceil(height, BLOCK_HEIGHT));
    const dim3 threadBlock (BLOCK_WIDTH, BLOCK_HEIGHT);
    auto kernel_start_time = high_resolution_clock::now();
    sobel_x_kernel<<<grid, threadBlock>>>(device_input, device_x_grad, grad_image_height, grad_image_width, GRAD_PADDING_SIZE, GRAD_PADDING_SIZE);
    sobel_y_kernel<<<grid, threadBlock>>>(device_input, device_y_grad, grad_image_height, grad_image_width, GRAD_PADDING_SIZE, GRAD_PADDING_SIZE);
    cudaDeviceSynchronize();
    cornerness_kernel<<<grid, threadBlock>>>(device_x_grad, device_y_grad, device_output, height, width);
    cudaDeviceSynchronize();
    auto kernel_end_time = high_resolution_clock::now();

    // Copy result to CPU
    auto mem_start_time3 = high_resolution_clock::now();
    cudaMemcpy(output, device_output, output_image_size, cudaMemcpyDeviceToHost);
    auto mem_end_time3 = high_resolution_clock::now();
    auto mem_start_time4 = high_resolution_clock::now();
    cudaFree(device_x_grad);
    cudaFree(device_y_grad);
    cudaFree(device_input);
    cudaFree(device_output);
    auto mem_end_time4 = high_resolution_clock::now();
    auto end_time = high_resolution_clock::now();

    printf("Kernel: %ld us\n", duration_cast<microseconds>(kernel_end_time - kernel_start_time).count());
    printf("Memory 1: %ld us\n", duration_cast<microseconds>(mem_end_time1 - mem_start_time1).count());
    printf("Memory 2: %ld us\n", duration_cast<microseconds>(mem_end_time2 - mem_start_time2).count());
    printf("Memory 3: %ld us\n", duration_cast<microseconds>(mem_end_time3 - mem_start_time3).count());
    printf("Memory 4: %ld us\n", duration_cast<microseconds>(mem_end_time4 - mem_start_time4).count());
    printf("Total Time: %ld us\n", duration_cast<microseconds>(end_time - start_time).count());
}

void init_cuda() {
    cudaSetDevice(0);
    cudaFree(0);
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