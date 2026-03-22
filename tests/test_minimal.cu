// Minimal CUDA test
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    fprintf(stderr, "Starting...\n");
    fflush(stderr);

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    fprintf(stderr, "cudaGetDeviceCount: %d devices, err=%d (%s)\n",
            count, (int)err, cudaGetErrorString(err));
    fflush(stderr);

    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        fprintf(stderr, "GPU: %s SM %d.%d\n", prop.name, prop.major, prop.minor);
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
