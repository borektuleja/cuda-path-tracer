#include <stdlib.h>
#include <string.h>

#include "util/memory.h"

void *buffer_create_host(size_t bytes)
{
    void *address = NULL;
    address = malloc(bytes);
    memset(address, 0, bytes);
    return address;
}

void *buffer_create_device(size_t bytes)
{
    void *address = NULL;
    CUDA_CHECK(cudaMalloc(&address, bytes));
    CUDA_CHECK(cudaMemset(address, 0, bytes));
    return address;
}

void buffer_delete_host(void *host)
{
    free(host);
}

void buffer_delete_device(void *device)
{
    CUDA_CHECK(cudaFree(device));
}

void buffer_copy_host_to_device(const void *host, void *device, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
}

void buffer_copy_device_to_host(const void *device, void *host, size_t bytes)
{
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}
