cls
del shader.optixir
nvcc -optix-ir --use_fast_math --gpu-architecture=compute_50 --machine=64 --relocatable-device-code=true --generate-line-info -I ../vendor/optix/include -I ../src shader.cu -o shader.optixir
