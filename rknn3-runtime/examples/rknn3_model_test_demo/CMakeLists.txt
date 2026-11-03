cmake_minimum_required(VERSION 3.6)

project(rknn3_model_test_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Map CMake system name to SDK directory naming (Darwin -> macos)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CUR_SYSTEM_NAME "macos")
else()
  set(CUR_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
endif()

# skip 3rd-party lib dependencies
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
endif()

# rknn3 api
set(RKNN3_API_PATH ${CMAKE_SOURCE_DIR}/../../rknn3-api)

# Determine architecture / library directory
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LIB_ARCH x86_64)
  set(RKNN3_API_LIB_EXT ".dll")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(LIB_ARCH arm64)
  set(RKNN3_API_LIB_EXT ".dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(LIB_ARCH arm64-v8a)
  set(RKNN3_API_LIB_EXT ".so")
elseif(CMAKE_C_COMPILER MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(LIB_ARCH aarch64)
  set(RKNN3_API_LIB_EXT ".so")
elseif(CMAKE_C_COMPILER MATCHES "x86_64|amd64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|x64")
  set(LIB_ARCH x86_64)
  set(RKNN3_API_LIB_EXT ".so")
else()
  set(LIB_ARCH arm64-v8a)
  set(RKNN3_API_LIB_EXT ".so")
endif()

set(RKNN_API_LIB_DIR ${RKNN3_API_PATH}/${CUR_SYSTEM_NAME}/${LIB_ARCH})
set(RKNN3_API_LIB ${RKNN_API_LIB_DIR}/librknn3_api${RKNN3_API_LIB_EXT})
set(RKNN3_API_NATIVE_LIB ${RKNN_API_LIB_DIR}/librknn3_api_native${RKNN3_API_LIB_EXT})
set(RKNN3_API_RKCP_LIB ${RKNN_API_LIB_DIR}/librknn3_api_rkcp${RKNN3_API_LIB_EXT})

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND LIB_ARCH STREQUAL "x86_64")
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

include_directories(${RKNN3_API_PATH}/include)

set(CMAKE_INSTALL_RPATH "lib")

# rknn3_model_test
add_executable(rknn3_model_test
  src/rknn3_model_test.cpp
  src/cnpy.cpp
  src/rknn3_test_utils.cpp
)

target_link_libraries(rknn3_model_test
  ${RKNN3_API_LIB}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND LIB_ARCH STREQUAL "x86_64")
  target_link_libraries(rknn3_model_test Threads::Threads)
endif()

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn3_model_test_${CUR_SYSTEM_NAME})
install(TARGETS rknn3_model_test DESTINATION ./)

install(PROGRAMS ${RKNN3_API_LIB} DESTINATION lib)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # On Windows, install all DLLs from the rknn3-api Windows directory
  file(GLOB RKNN3_API_DLLS "${RKNN_API_LIB_DIR}/*.dll")
  foreach(dll ${RKNN3_API_DLLS})
    install(PROGRAMS ${dll} DESTINATION lib)
  endforeach()
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # On macOS (USB connection), install all dylibs from rknn3-api/macos/arm64
  file(GLOB RKNN3_API_DYLIBS "${RKNN_API_LIB_DIR}/*.dylib")
  foreach(dylib ${RKNN3_API_DYLIBS})
    install(PROGRAMS ${dylib} DESTINATION lib)
  endforeach()
elseif (LIB_ARCH STREQUAL "x86_64")
  # PC direct connection to RK182X: install coprocessor library
  if (EXISTS ${RKNN3_API_RKCP_LIB})
    install(PROGRAMS ${RKNN3_API_RKCP_LIB} DESTINATION lib)
  endif()
elseif (TARGET_SOC STREQUAL "RK3572")
  if (EXISTS ${RKNN3_API_NATIVE_LIB})
    install(PROGRAMS ${RKNN3_API_NATIVE_LIB} DESTINATION lib)
  endif()
elseif (TARGET_SOC STREQUAL "RK3576" OR TARGET_SOC STREQUAL "RK3588")
  if (EXISTS ${RKNN3_API_RKCP_LIB})
    install(PROGRAMS ${RKNN3_API_RKCP_LIB} DESTINATION lib)
  endif()
endif()
