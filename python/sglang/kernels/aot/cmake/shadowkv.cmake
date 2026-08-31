if(NOT DEFINED SGL_KERNEL_SHADOWKV_GENERIC_SUPPORTED_ARCHS)
    set(SGL_KERNEL_SHADOWKV_GENERIC_SUPPORTED_ARCHS "80;100a" CACHE STRING
        "Exact CUDA architectures supported by the generic ShadowKV source")
endif()
if(NOT DEFINED SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH)
    set(SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH "" CACHE STRING
        "Exact architecture for an optional ShadowKV specialized source set")
endif()
if(NOT DEFINED SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES)
    set(SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES "" CACHE STRING
        "Semicolon-separated architecture-specialized ShadowKV sources")
endif()

function(sgl_configure_shadowkv_sources generic_output)
    set(generic_sources)
    set(binding_sources)
    set(specialized_sources)
    if(SGL_KERNEL_ENABLE_SHADOWKV)
        list(FIND SGL_KERNEL_SHADOWKV_GENERIC_SUPPORTED_ARCHS "${SGL_KERNEL_CUDA_ARCH}" generic_arch_index)
        if(generic_arch_index EQUAL -1)
            message(FATAL_ERROR
                "SGL_KERNEL_ENABLE_SHADOWKV requires SGL_KERNEL_CUDA_ARCH in the generic candidate set "
                "${SGL_KERNEL_SHADOWKV_GENERIC_SUPPORTED_ARCHS}; got '${SGL_KERNEL_CUDA_ARCH}'.")
        endif()
        if(SGL_KERNEL_BUILD_SM90_VARIANT OR NOT SGL_KERNEL_BUILD_SM100_VARIANT)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires only the precise SM100 common_ops variant.")
        endif()
        if(NOT SGL_KERNEL_ENABLE_BF16)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires BF16 support.")
        endif()
        if(SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH AND NOT SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES)
            message(FATAL_ERROR "ShadowKV specialization architecture requires a non-empty specialized source set.")
        endif()
        if(SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES AND NOT SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH)
            message(FATAL_ERROR "ShadowKV specialized sources require an exact specialization architecture.")
        endif()
        if(SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH AND
           NOT SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH STREQUAL SGL_KERNEL_CUDA_ARCH)
            message(FATAL_ERROR
                "ShadowKV specialization '${SGL_KERNEL_SHADOWKV_SPECIALIZATION_ARCH}' contradicts build target "
                "'${SGL_KERNEL_CUDA_ARCH}'.")
        endif()
        list(APPEND generic_sources
            "csrc/shadowkv/generic/packed_gqa.cu"
            "csrc/shadowkv/generic/plan_reuse.cu"
            "csrc/shadowkv/generic/reconstruct_rope.cu"
        )
        list(APPEND binding_sources
            "csrc/shadowkv/bindings/shadowkv_extension.cc"
        )
        list(APPEND specialized_sources ${SGL_KERNEL_SHADOWKV_SPECIALIZATION_SOURCES})
    endif()
    set(${generic_output} "${generic_sources}" PARENT_SCOPE)
    if(ARGC GREATER 1)
        set(${ARGV1} "${binding_sources}" PARENT_SCOPE)
    endif()
    if(ARGC GREATER 2)
        set(${ARGV2} "${specialized_sources}" PARENT_SCOPE)
    endif()
endfunction()
