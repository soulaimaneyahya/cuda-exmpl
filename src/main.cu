#include <stdio.h>

// CUDA kernel: squares array and prints thread/block IDs
__global__ void square(int *a, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // global thread index
    if (idx < N)
    {
        a[idx] = a[idx] * a[idx];
        printf("Thread %d in Block %d processed element %d\n", threadIdx.x, blockIdx.x, idx);
    }
}

int main()
{
    int N = 10;
    int a[N];

    // Initialize array on CPU
    for (int i = 0; i < N; i++)
        a[i] = i;

    int *d_a;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 2 blocks, 5 threads each (just example)
    square<<<2, 5>>>(d_a, N);
    cudaDeviceSynchronize(); // Ensure printf works before copying data back

    cudaMemcpy(a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);

    // Print results
    for (int i = 0; i < N; i++)
        printf("%d ", a[i]);
    printf("\n");

    return 0;
}
