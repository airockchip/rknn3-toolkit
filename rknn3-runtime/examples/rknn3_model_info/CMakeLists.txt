cmake_minimum_required(VERSION 3.6)

project(rknn3_model_info)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn3 api
set(RKNN3_API_PATH ${CMAKE_SOURCE_DIR}/../../rknn3-api)

# Determine architecture based on system and compiler
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(LIB_ARCH x86_64)
  set(RKNN3_API_LIB_EXT ".dll")
elseif(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
  set(RKNN3_API_LIB_EXT ".so")
else()
  set(LIB_ARCH arm64-v8a)
  set(RKNN3_API_LIB_EXT ".so")
endif()

set(RKNN3_API_LIB ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api${RKNN3_API_LIB_EXT})
set(RKNN3_API_NATIVE_LIB ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api_native${RKNN3_API_LIB_EXT})
set(RKNN3_API_RKCP_LIB ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api_rkcp${RKNN3_API_LIB_EXT})

include_directories(${RKNN3_API_PATH}/include)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn3_model_info
  src/rknn3_model_info.cpp
)

target_link_libraries(rknn3_model_info
  ${RKNN3_API_LIB}
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn3_model_info_${CMAKE_SYSTEM_NAME})
install(TARGETS rknn3_model_info DESTINATION ./)

install(PROGRAMS ${RKNN3_API_LIB} DESTINATION lib)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  file(GLOB RKNN3_API_DLLS "${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/*.dll")
  foreach(dll ${RKNN3_API_DLLS})
    install(PROGRAMS ${dll} DESTINATION lib)
  endforeach()
elseif (TARGET_SOC STREQUAL "RK3572")
  if (EXISTS ${RKNN3_API_NATIVE_LIB})
    install(PROGRAMS ${RKNN3_API_NATIVE_LIB} DESTINATION lib)
  endif()
elseif (TARGET_SOC STREQUAL "RK3576" OR TARGET_SOC STREQUAL "RK3588")
  if (EXISTS ${RKNN3_API_RKCP_LIB})
    install(PROGRAMS ${RKNN3_API_RKCP_LIB} DESTINATION lib)
  endif()
endif()
