set(SGL_KERNEL_SHADOWKV_TARGET_ARCHITECTURE "" CACHE STRING
    "Exact manifest-derived architecture for the ShadowKV bundle closure")
set(SGL_KERNEL_SHADOWKV_BUILD_INPUT_SHA256 "" CACHE STRING
    "Canonical digest of the manifest-derived ShadowKV build input")
set(SGL_KERNEL_SHADOWKV_BUNDLE_IDS "" CACHE STRING
    "Semicolon-separated effective ShadowKV bundle identities")
set(SGL_KERNEL_SHADOWKV_NATIVE_SOURCES "" CACHE STRING
    "Semicolon-separated manifest-derived ShadowKV native compile units")
set(SGL_KERNEL_SHADOWKV_EXPECTED_SYMBOLS "" CACHE STRING
    "Semicolon-separated internal symbols owned by the effective closure")

function(sgl_configure_shadowkv_sources cuda_output binding_output unused_specialized_output)
    set(cuda_sources)
    set(binding_sources)
    set(specialized_sources)
    if(SGL_KERNEL_ENABLE_SHADOWKV)
        if(NOT SGL_KERNEL_SHADOWKV_TARGET_ARCHITECTURE STREQUAL "sm${SGL_KERNEL_CUDA_ARCH}")
            message(FATAL_ERROR
                "ShadowKV bundle target '${SGL_KERNEL_SHADOWKV_TARGET_ARCHITECTURE}' contradicts "
                "SGL_KERNEL_CUDA_ARCH='${SGL_KERNEL_CUDA_ARCH}'.")
        endif()
        string(LENGTH "${SGL_KERNEL_SHADOWKV_BUILD_INPUT_SHA256}" shadowkv_digest_length)
        if(NOT shadowkv_digest_length EQUAL 64 OR
           NOT SGL_KERNEL_SHADOWKV_BUILD_INPUT_SHA256 MATCHES "^[0-9a-f]+$")
            message(FATAL_ERROR "ShadowKV requires one canonical build-input SHA-256 digest.")
        endif()
        if(NOT SGL_KERNEL_SHADOWKV_BUNDLE_IDS)
            message(FATAL_ERROR "ShadowKV requires at least one effective bundle identity.")
        endif()
        if(NOT SGL_KERNEL_SHADOWKV_NATIVE_SOURCES)
            message(FATAL_ERROR "ShadowKV requires manifest-derived native compile units.")
        endif()
        if(NOT SGL_KERNEL_SHADOWKV_EXPECTED_SYMBOLS)
            message(FATAL_ERROR "ShadowKV requires manifest-derived internal symbols.")
        endif()
        if(SGL_KERNEL_BUILD_SM90_VARIANT OR NOT SGL_KERNEL_BUILD_SM100_VARIANT)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires only the precise SM100 common_ops variant.")
        endif()
        if(NOT SGL_KERNEL_ENABLE_BF16)
            message(FATAL_ERROR "SGL_KERNEL_ENABLE_SHADOWKV requires BF16 support.")
        endif()

        set(unique_sources ${SGL_KERNEL_SHADOWKV_NATIVE_SOURCES})
        list(LENGTH unique_sources source_count)
        list(REMOVE_DUPLICATES unique_sources)
        list(LENGTH unique_sources unique_source_count)
        if(NOT source_count EQUAL unique_source_count)
            message(FATAL_ERROR "ShadowKV manifest-derived native sources contain duplicates.")
        endif()
        foreach(source IN LISTS unique_sources)
            if(IS_ABSOLUTE "${source}" OR source MATCHES "(^|/)\\.\\.(/|$)")
                message(FATAL_ERROR "Unsafe ShadowKV native source path '${source}'.")
            endif()
            if(NOT source MATCHES "^csrc/shadowkv/")
                message(FATAL_ERROR
                    "ShadowKV native source '${source}' is outside csrc/shadowkv.")
            endif()
            if(NOT EXISTS "${PROJECT_SOURCE_DIR}/${source}")
                message(FATAL_ERROR "Declared ShadowKV native source '${source}' does not exist.")
            endif()
            if(source MATCHES "\\.cu$")
                list(APPEND cuda_sources "${source}")
            elseif(source MATCHES "\\.(cc|cpp|cxx)$")
                list(APPEND binding_sources "${source}")
            else()
                message(FATAL_ERROR
                    "ShadowKV native compile unit '${source}' has an unsupported suffix.")
            endif()
        endforeach()
        if(NOT cuda_sources OR NOT binding_sources)
            message(FATAL_ERROR
                "ShadowKV effective closure requires CUDA and binding compile units.")
        endif()
    else()
        foreach(name IN ITEMS
                SGL_KERNEL_SHADOWKV_TARGET_ARCHITECTURE
                SGL_KERNEL_SHADOWKV_BUILD_INPUT_SHA256
                SGL_KERNEL_SHADOWKV_BUNDLE_IDS
                SGL_KERNEL_SHADOWKV_NATIVE_SOURCES
                SGL_KERNEL_SHADOWKV_EXPECTED_SYMBOLS)
            if(NOT "${${name}}" STREQUAL "")
                message(FATAL_ERROR
                    "Disabled ShadowKV build contains stale bundle input ${name}.")
            endif()
        endforeach()
    endif()
    set(${cuda_output} "${cuda_sources}" PARENT_SCOPE)
    set(${binding_output} "${binding_sources}" PARENT_SCOPE)
    set(${unused_specialized_output} "${specialized_sources}" PARENT_SCOPE)
endfunction()
