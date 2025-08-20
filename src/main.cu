#include <stdio.h>

// CUDA kernel: runs on GPU
__global__ void square(int *a, int N)
{
    int idx = threadIdx.x; // Each thread gets its unique index
    if (idx < N) // Check bounds
        a[idx] = a[idx] * a[idx]; // Square the element
}

int main()
{
    int N = 10;
    int a[N];

    // Initialize array on CPU
    for (int i = 0; i < N; i++)
        a[i] = i;

    int *d_a;
    // Allocate memory on GPU
    cudaMalloc(&d_a, N * sizeof(int));

    // Copy data from CPU (host) to GPU (device)
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block, N threads
    square<<<1, N>>>(d_a, N);

    // Copy results back from GPU to CPU
    cudaMemcpy(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);

    // Print results
    for (int i = 0; i < N; i++)
        printf("%d ", a[i]);
    printf("\n");

    return 0;
}
