#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

/* note that input and output must be in shared memory. This operation can be done in place */
__device__ 
void convertToGrayScale(int *sInput, int *sOutput, size_t height, size_t width) {

}

