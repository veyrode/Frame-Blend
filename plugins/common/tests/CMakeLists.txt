cmake_minimum_required(VERSION 3.16)
project(frame_blend_gpu_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static C/C++ runtime: hosts load their own, possibly much older, msvcp140.dll.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

if(NOT DEFINED FRAME_BLEND_CORE_LIB OR NOT EXISTS "${FRAME_BLEND_CORE_LIB}")
    message(FATAL_ERROR "Set FRAME_BLEND_CORE_LIB to target/release/frame_blend.lib.")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../gpu.cmake)
add_executable(gpu_parity gpu_parity.cpp)
frame_blend_add_gpu(gpu_parity)
target_include_directories(gpu_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(gpu_parity PRIVATE "${FRAME_BLEND_CORE_LIB}" kernel32 ntdll userenv ws2_32 dbghelp)

add_executable(media_reader_test media_reader_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../adobe/fb_media_reader.cpp)
target_include_directories(media_reader_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../adobe ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
frame_blend_add_gpu(media_reader_test)
frame_blend_add_ffmpeg(media_reader_test)
target_link_libraries(media_reader_test PRIVATE "${FRAME_BLEND_CORE_LIB}" kernel32 ntdll userenv ws2_32 dbghelp)

add_executable(nv12_parity nv12_parity.cpp)
frame_blend_add_gpu(nv12_parity)
target_include_directories(nv12_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(nv12_parity PRIVATE "${FRAME_BLEND_CORE_LIB}" kernel32 ntdll userenv ws2_32 dbghelp)

add_executable(upload_bench upload_bench.cpp)
frame_blend_add_gpu(upload_bench)
target_include_directories(upload_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_link_libraries(upload_bench PRIVATE "${FRAME_BLEND_CORE_LIB}" kernel32 ntdll userenv ws2_32 dbghelp)
