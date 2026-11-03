cmake_minimum_required(VERSION 3.10)

project(rknn3_llm_demos)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif()

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

# ---------------------------------------------------------------
# Locate rknn3-api and tokenizer libs. Two layouts are supported:
#   A. this repo:    <repo>/rknn3-api, <repo>/examples/3rdparty
#   B. model-zoo:    <mz>/3rdparty (add_subdirectory provides
#                    LIBRKNNAPI / LIBTOKENIZER / *_INCLUDES)
# ---------------------------------------------------------------

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../../rknn3-api)
    set(RKNN3_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../rknn3-api)

    if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(LIB_ARCH x86_64)
        set(RKNN3_API_LIB_EXT ".dll")
    elseif (CMAKE_C_COMPILER MATCHES "aarch64")
        set(LIB_ARCH aarch64)
        set(RKNN3_API_LIB_EXT ".so")
    elseif (TARGET_SOC STREQUAL "x86")
        set(LIB_ARCH x86_64)
        set(RKNN3_API_LIB_EXT ".so")
    else()
        set(LIB_ARCH arm64-v8a)
        set(RKNN3_API_LIB_EXT ".so")
    endif()

    set(LIBRKNNAPI ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api${RKNN3_API_LIB_EXT})
    set(LIBRKNNAPI_RKCP ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api_rkcp${RKNN3_API_LIB_EXT})
    set(LIBRKNNAPI_NATIVE ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api_native${RKNN3_API_LIB_EXT})
    set(LIBRKNNAPI_INCLUDES ${RKNN3_API_PATH}/include)

    set(TOKENIZER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/tokenizer)
    set(LIBTOKENIZER ${TOKENIZER_PATH}/lib/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/libtokenizer.a)
    set(LIBTOKENIZER_INCLUDES ${TOKENIZER_PATH}/include)
else()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
endif()

message(STATUS "LIBRKNNAPI = ${LIBRKNNAPI}")
message(STATUS "LIBRKNNAPI_INCLUDES = ${LIBRKNNAPI_INCLUDES}")
message(STATUS "LIBTOKENIZER = ${LIBTOKENIZER}")
message(STATUS "TARGET_LIB_ARCH = ${TARGET_LIB_ARCH}")

# ---------------------------------------------------------------
# JSON: this repo ships a single-file amalgamated header at
# 3rdparty/json/json.hpp while the code includes "nlohmann/json.hpp".
# If the canonical <...>/nlohmann/json.hpp is absent, generate a shim
# in the build dir so "nlohmann/json.hpp" resolves to the flat header.
# ---------------------------------------------------------------
set(LLM_JSON_INCLUDE_DIR)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/json/nlohmann/json.hpp)
    set(LLM_JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty)
elseif (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/json/json.hpp)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nlohmann)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nlohmann/json.hpp
         "#pragma once\n#include \"../../../../3rdparty/json/json.hpp\"\n")
    set(LLM_JSON_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

# ---------------------------------------------------------------
# Shared sources used by both targets
# ---------------------------------------------------------------

set(LLM_COMMON_SOURCES
    engine.cc
    args.cc
    rope.cc
    lora.cc
)

# Install the platform rknn3 api library (rkcp / native) when present
set(RKNN3_API_PLATFORM_LIB)
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if (EXISTS ${LIBRKNNAPI_RKCP})
        set(RKNN3_API_PLATFORM_LIB ${LIBRKNNAPI_RKCP})
    endif()
elseif (TARGET_SOC STREQUAL "rk3572" OR TARGET_SOC STREQUAL "RK3572")
    if (EXISTS ${LIBRKNNAPI_NATIVE})
        set(RKNN3_API_PLATFORM_LIB ${LIBRKNNAPI_NATIVE})
    endif()
elseif (EXISTS ${LIBRKNNAPI_RKCP})
    set(RKNN3_API_PLATFORM_LIB ${LIBRKNNAPI_RKCP})
endif()

if (RKNN3_API_PLATFORM_LIB)
    install(PROGRAMS ${RKNN3_API_PLATFORM_LIB} DESTINATION lib)
endif()
install(PROGRAMS ${LIBRKNNAPI} DESTINATION lib)

# ---------------------------------------------------------------
# rknn3_llm_cli : interactive command-line chat client
# ---------------------------------------------------------------

add_executable(rknn3_llm_cli
    ${LLM_COMMON_SOURCES}
    cli_main.cc
)

target_link_libraries(rknn3_llm_cli
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

# ---------------------------------------------------------------
# rknn3_llm_step_decode : per-token (external) decode demo
# ---------------------------------------------------------------

add_executable(rknn3_llm_step_decode
    ${LLM_COMMON_SOURCES}
    step_main.cc
)

target_link_libraries(rknn3_llm_step_decode
    ${LIBRKNNAPI}
    ${LIBTOKENIZER}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    foreach(tgt rknn3_llm_cli rknn3_llm_step_decode)
        target_link_libraries(${tgt} log)
    endforeach()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(tgt rknn3_llm_cli rknn3_llm_step_decode)
        target_link_libraries(${tgt} pthread)
    endforeach()
endif()

foreach(tgt rknn3_llm_cli rknn3_llm_step_decode)
    target_include_directories(${tgt} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBRKNNAPI_INCLUDES}
        ${LIBTOKENIZER_INCLUDES}
        ${LLM_JSON_INCLUDE_DIR}
    )
    install(TARGETS ${tgt} DESTINATION .)
endforeach()
