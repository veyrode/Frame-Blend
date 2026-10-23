# Shared GPU sources for both adapters: fb_blend.cu is compiled to PTX with
# nvcc (loaded at run time through the CUDA driver API) and fb_blend.cl is
# embedded as source for run-time OpenCL compilation.
set(FB_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR})
find_program(NVCC_EXECUTABLE nvcc HINTS "$ENV{CUDA_PATH}/bin" REQUIRED)
get_filename_component(FB_CUDA_INCLUDE_DIR "${NVCC_EXECUTABLE}/../../include" ABSOLUTE)

# FFmpeg (Premiere file reader): headers and import libraries are tracked; the
# DLLs are delay-loaded from the plugin's folder (see fb_media_reader.cpp).
set(FB_FFMPEG_DIR "${FB_COMMON_DIR}/../../dependence/sdks/FFmpeg_8.1_lgpl_win64")
set(FB_FFMPEG_DLLS avutil-60.dll swresample-6.dll avcodec-62.dll avformat-62.dll)

function(frame_blend_add_ffmpeg target)
    target_include_directories(${target} PRIVATE "${FB_FFMPEG_DIR}/include" "${FB_CUDA_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE "${FB_FFMPEG_DIR}/lib/avformat.lib" "${FB_FFMPEG_DIR}/lib/avcodec.lib"
                          "${FB_FFMPEG_DIR}/lib/avutil.lib" delayimp)
    target_link_options(${target} PRIVATE /DELAYLOAD:avformat-62.dll /DELAYLOAD:avcodec-62.dll /DELAYLOAD:avutil-60.dll)
    foreach(dll ${FB_FFMPEG_DLLS})
        if(NOT EXISTS "${FB_FFMPEG_DIR}/bin/${dll}")
            message(FATAL_ERROR "Missing ${FB_FFMPEG_DIR}/bin/${dll}: download the FFmpeg build named in "
                                "dependence/sdks/FFmpeg_8.1_lgpl_win64/README.md and put its four DLLs in that bin folder.")
        endif()
    endforeach()
    foreach(dll ${FB_FFMPEG_DLLS})
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FB_FFMPEG_DIR}/bin/${dll}" "$<TARGET_FILE_DIR:${target}>")
    endforeach()
endfunction()

function(frame_blend_add_gpu target)
    set(gen ${CMAKE_CURRENT_BINARY_DIR}/generated)
    file(MAKE_DIRECTORY ${gen})
    get_filename_component(host_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    # CUDA 12.0 predates MSVC 14.5; the kernel uses no host STL, so the version
    # checks are overridden. compute_50 PTX is JIT-compiled for any newer GPU.
    add_custom_command(
        OUTPUT ${gen}/fb_blend.ptx
        COMMAND ${NVCC_EXECUTABLE} -ptx -arch=compute_50 -allow-unsupported-compiler
                -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ccbin "${host_compiler_dir}"
                -o ${gen}/fb_blend.ptx ${FB_COMMON_DIR}/gpu/fb_blend.cu
        DEPENDS ${FB_COMMON_DIR}/gpu/fb_blend.cu
        COMMENT "Compiling fb_blend.cu to PTX"
        VERBATIM)
    add_custom_command(
        OUTPUT ${gen}/fb_blend_ptx.h
        COMMAND ${CMAKE_COMMAND} -DINPUT=${gen}/fb_blend.ptx -DOUTPUT=${gen}/fb_blend_ptx.h -DNAME=kFbBlendPtx
                -P ${FB_COMMON_DIR}/embed.cmake
        DEPENDS ${gen}/fb_blend.ptx ${FB_COMMON_DIR}/embed.cmake
        VERBATIM)
    add_custom_command(
        OUTPUT ${gen}/fb_blend_cl.h
        COMMAND ${CMAKE_COMMAND} -DINPUT=${FB_COMMON_DIR}/gpu/fb_blend.cl -DOUTPUT=${gen}/fb_blend_cl.h -DNAME=kFbBlendCl
                -P ${FB_COMMON_DIR}/embed.cmake
        DEPENDS ${FB_COMMON_DIR}/gpu/fb_blend.cl ${FB_COMMON_DIR}/embed.cmake
        VERBATIM)
    target_sources(${target} PRIVATE
        ${FB_COMMON_DIR}/fb_gpu.cpp
        ${FB_COMMON_DIR}/fb_debug.cpp
        ${gen}/fb_blend_ptx.h
        ${gen}/fb_blend_cl.h)
    target_include_directories(${target} PRIVATE ${FB_COMMON_DIR} ${gen})
endfunction()
