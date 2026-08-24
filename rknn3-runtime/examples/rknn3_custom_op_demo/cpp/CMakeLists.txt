cmake_minimum_required(VERSION 3.6)

project(rknn3_custom_op_demo CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# rknn3 api
set(RKNN3_API_PATH ${CMAKE_SOURCE_DIR}/../../../rknn3-api)

# Determine architecture based on system and compiler
# (Android uses clang without "aarch64" in its path, so check Android first)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(LIB_ARCH arm64-v8a)
elseif(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
elseif(CMAKE_C_COMPILER MATCHES "x86_64|amd64")
  set(LIB_ARCH x86_64)
else()
  set(LIB_ARCH aarch64)
endif()

set(RKNN3_API_LIB ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api.so)
set(RKNN3_API_RKCP_LIB ${RKNN3_API_PATH}/${CMAKE_SYSTEM_NAME}/${LIB_ARCH}/librknn3_api_rkcp.so)

include_directories(${RKNN3_API_PATH}/include)

set(CMAKE_INSTALL_RPATH "lib")

# demo executable
add_executable(rknn3_custom_op_demo main.cc)

if(DEFINED TARGET_SOC AND TARGET_SOC STREQUAL "RK3572")
  target_compile_definitions(rknn3_custom_op_demo PRIVATE INTERFACE_LINUX_OR_ANDROID=1)
endif()

target_link_libraries(rknn3_custom_op_demo ${RKNN3_API_LIB} dl m)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../install)
install(TARGETS rknn3_custom_op_demo DESTINATION ./)
install(PROGRAMS ${RKNN3_API_LIB} DESTINATION lib)

if (TARGET_SOC STREQUAL "RK3576" OR TARGET_SOC STREQUAL "RK3588")
  if (EXISTS ${RKNN3_API_RKCP_LIB})
    install(PROGRAMS ${RKNN3_API_RKCP_LIB} DESTINATION lib)
  endif()
endif()

# also install the plugin .so if it exists
set(PLUGIN_SO ${CMAKE_SOURCE_DIR}/plugin/librknn_custom_op.so)
if (EXISTS ${PLUGIN_SO})
  install(PROGRAMS ${PLUGIN_SO} DESTINATION lib)
endif()
