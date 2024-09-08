#ifndef _MEMORY_H_
#define _MEMORY_H_

#include "util/cuda_defs.h"

void *buffer_create_host(size_t bytes);
void *buffer_create_device(size_t bytes);

void buffer_delete_host(void *host);
void buffer_delete_device(void *device);

void buffer_copy_host_to_device(const void *host, void *device, size_t bytes);
void buffer_copy_device_to_host(const void *device, void *host, size_t bytes);

#endif
