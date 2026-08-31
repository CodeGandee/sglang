function(sgl_configure_shadowkv_sources output_variable)
    set(shadowkv_sources)
    if(SGL_KERNEL_ENABLE_SHADOWKV)
        if(NOT SGL_KERNEL_CUDA_ARCH STREQUAL "80" AND NOT SGL_KERNEL_CUDA_ARCH STREQUAL "100a")
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires SGL_KERNEL_CUDA_ARCH=80 or 100a; got '${SGL_KERNEL_CUDA_ARCH}'.")
        endif()
        if(SGL_KERNEL_BUILD_SM90_VARIANT OR NOT SGL_KERNEL_BUILD_SM100_VARIANT)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires only the precise SM100 common_ops variant.")
        endif()
        if(NOT SGL_KERNEL_ENABLE_BF16)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires BF16 support.")
        endif()
        list(APPEND shadowkv_sources
            "csrc/shadowkv/packed_gqa.cu"
            "csrc/shadowkv/plan_reuse.cu"
            "csrc/shadowkv/reconstruct_rope.cu"
        )
    endif()
    set(${output_variable} "${shadowkv_sources}" PARENT_SCOPE)
endfunction()
