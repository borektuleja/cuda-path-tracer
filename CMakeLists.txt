cmake_minimum_required(VERSION 3.27.0)

project(bssrdf LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(vendor/pugixml)
add_subdirectory(src)
