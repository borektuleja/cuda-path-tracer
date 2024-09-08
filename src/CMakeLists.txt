set(INC
    ${CMAKE_SOURCE_DIR}/vendor/glfw/include
    ${CMAKE_SOURCE_DIR}/vendor/optix/include
    ${CMAKE_SOURCE_DIR}/vendor/pugixml/include
    ${CMAKE_SOURCE_DIR}/vendor/tinyobj/include
    ${CMAKE_SOURCE_DIR}/src
)

set(DIR
    ${CMAKE_SOURCE_DIR}/vendor/glfw/lib
)

set(LIB
    cudart
    glfw3
    opengl32
    pugixml
)

set(SRC
    core/io.cpp
    core/renderer.cpp
    core/texture.cpp
    core/window.cpp

    optix/optix_bvh.cpp
    optix/optix_facade.cpp
    optix/optix_globals.cpp
    optix/optix_sbt.cpp
    
    scene/camera.cpp
    scene/mesh.cpp
    scene/scene_builder.cpp
    scene/scene_facade.cpp
    
    util/memory.cpp
    
    main.cpp
)

set(SRC_HEADERS
    core/io.h
    core/renderer.h
    core/texture.h
    core/window.h

    math/math_defs.h
    math/math_float3_fun.h
    math/math_float3_lin.h
    math/math_float3_std.h
    math/math_float3.h
    
    optix/optix_bvh.h
    optix/optix_defs.h
    optix/optix_facade.h
    optix/optix_globals.h
    optix/optix_params.h
    optix/optix_sbt.h
    
    scene/camera.h
    scene/mesh.h
    scene/scene_builder.h
    scene/scene_defs.h
    scene/scene_facade.h
    
    util/algo.h
    util/cuda_defs.h
    util/memory.h
    util/random.h
    util/shapes.h
    util/util_rt_frsnl.h
    util/util_rt_phase.h
    util/util_rt_sampl.h
)

add_executable(bssrdf ${SRC} ${SRC_HEADERS})

target_include_directories(bssrdf PRIVATE ${INC})
target_link_directories(bssrdf PRIVATE ${DIR})
target_link_libraries(bssrdf PRIVATE ${LIB})
