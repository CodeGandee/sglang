function(sgl_configure_shadowkv_sources cuda_output binding_output unused_specialized_output)
    set(cuda_sources)
    set(binding_sources)
    if(SGL_KERNEL_ENABLE_SHADOWKV)
        if(NOT SGL_KERNEL_CUDA_ARCH MATCHES "^(80|100a)$")
            message(FATAL_ERROR
                "ShadowKV kernels support only SGL_KERNEL_CUDA_ARCH=80 or 100a.")
        endif()
        if(SGL_KERNEL_BUILD_SM90_VARIANT OR NOT SGL_KERNEL_BUILD_SM100_VARIANT)
            message(FATAL_ERROR
                "SGL_KERNEL_ENABLE_SHADOWKV requires only the precise common_ops variant.")
        endif()
        if(NOT SGL_KERNEL_ENABLE_BF16)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires BF16 support.")
        endif()
        list(APPEND cuda_sources
            csrc/shadowkv/generic/packed_gqa.cu
            csrc/shadowkv/generic/plan_reuse.cu
            csrc/shadowkv/generic/reconstruct_rope.cu
        )
        list(APPEND binding_sources
            csrc/shadowkv/bindings/shadowkv_extension.cc
        )
    endif()
    set(${cuda_output} "${cuda_sources}" PARENT_SCOPE)
    set(${binding_output} "${binding_sources}" PARENT_SCOPE)
    set(${unused_specialized_output} "" PARENT_SCOPE)
endfunction()
