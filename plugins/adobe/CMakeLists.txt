cmake_minimum_required(VERSION 3.16)
project(frame_blend_adobe CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static C/C++ runtime: hosts load their own, possibly much older, msvcp140.dll.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

if(NOT DEFINED AE_SDK_PATH OR NOT EXISTS "${AE_SDK_PATH}/Headers/AE_Effect.h")
    message(FATAL_ERROR "Set AE_SDK_PATH to the After Effects SDK Examples directory (containing Headers/AE_Effect.h).")
endif()
if(NOT DEFINED PR_SDK_PATH OR NOT EXISTS "${PR_SDK_PATH}/Headers/PrSDKGPUFilter.h")
    message(FATAL_ERROR "Set PR_SDK_PATH to the Premiere Pro SDK directory containing Headers/PrSDKGPUFilter.h.")
endif()
if(NOT DEFINED FRAME_BLEND_CORE_LIB OR NOT EXISTS "${FRAME_BLEND_CORE_LIB}")
    message(FATAL_ERROR "Set FRAME_BLEND_CORE_LIB to target/release/frame_blend.lib.")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../common/gpu.cmake)

add_library(frame_blend_adobe MODULE frame_blend_ae.cpp frame_blend_gpu.cpp fb_media_reader.cpp)
frame_blend_add_gpu(frame_blend_adobe)
frame_blend_add_ffmpeg(frame_blend_adobe)
target_compile_definitions(frame_blend_adobe PRIVATE PRWIN_ENV)
set_target_properties(frame_blend_adobe PROPERTIES PREFIX "" OUTPUT_NAME "FrameBlend" SUFFIX ".aex")
target_include_directories(frame_blend_adobe PRIVATE
    ${AE_SDK_PATH}/Headers
    ${AE_SDK_PATH}/Headers/SP
    ${AE_SDK_PATH}/Util
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)
# Premiere GPU filter headers, after the After Effects ones.
target_include_directories(frame_blend_adobe PRIVATE ${PR_SDK_PATH}/Headers)

set(PIPL_R ${CMAKE_CURRENT_SOURCE_DIR}/FrameBlendPiPL.r)
set(PIPL_RR ${CMAKE_CURRENT_BINARY_DIR}/FrameBlendPiPL.rr)
set(PIPL_RRC ${CMAKE_CURRENT_BINARY_DIR}/FrameBlendPiPL.rrc)
set(PIPL_RC ${CMAKE_CURRENT_BINARY_DIR}/FrameBlendPiPL.rc)
set(PIPL_TOOL ${AE_SDK_PATH}/Resources/PiPLtool.exe)
if(NOT EXISTS "${PIPL_TOOL}")
    message(FATAL_ERROR "AE_SDK_PATH must contain Resources/PiPLtool.exe.")
endif()
add_custom_command(
    OUTPUT ${PIPL_RC}
    COMMAND "${CMAKE_CXX_COMPILER}" /I "${AE_SDK_PATH}/Headers" /EP "${PIPL_R}" > "${PIPL_RR}"
    COMMAND "${PIPL_TOOL}" "${PIPL_RR}" "${PIPL_RRC}"
    COMMAND "${CMAKE_CXX_COMPILER}" /D "MSWindows" /EP "${PIPL_RRC}" > "${PIPL_RC}"
    DEPENDS ${PIPL_R}
    COMMENT "Generating PiPL resource"
    VERBATIM
)
target_sources(frame_blend_adobe PRIVATE ${PIPL_RC})

target_link_libraries(frame_blend_adobe PRIVATE "${FRAME_BLEND_CORE_LIB}" kernel32 ntdll userenv ws2_32 dbghelp
)
