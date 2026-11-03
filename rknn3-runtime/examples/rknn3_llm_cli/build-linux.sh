#!/bin/bash

set -e

echo "$0 $@"
while getopts ":t:a:b" opt; do
  case $opt in
    t)
      TARGET_SOC=$OPTARG
      ;;
    a)
      TARGET_ARCH=$OPTARG
      ;;
    b)
      BUILD_TYPE=$OPTARG
      ;;
    :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
    ?)
      echo "Invalid option: -$OPTARG index:$OPTIND"
      ;;
  esac
done

if [ -z ${TARGET_SOC} ] || [ -z ${TARGET_ARCH} ]; then
  echo "$0 -t <target> -a <arch> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3588/rk3576/rk3572/x86)"
  echo "    -a : arch (aarch64/x86_64)"
  echo "    -b : build_type(Debug/Release)"
  echo "such as: $0 -t rk3588 -a aarch64 -b Release"
  echo ""
  exit -1
fi

# Debug / Release
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

# 当目标架构与宿主机架构相同时使用系统编译器，否则使用交叉编译器
HOST_ARCH=$(uname -m)
if [ "${HOST_ARCH}" = "${TARGET_ARCH}" ]; then
    export CC=gcc
    export CXX=g++
else
    if [[ -z ${GCC_COMPILER} ]];then
        echo "Please set GCC_COMPILER for $TARGET_SOC"
        echo "such as export GCC_COMPILER=~/opt/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu"
        exit
    fi
    export CC=${GCC_COMPILER}-gcc
    export CXX=${GCC_COMPILER}-g++
    if ! command -v ${CC} >/dev/null 2>&1; then
        echo "${CC} is not available"
        echo "Please set GCC_COMPILER for $TARGET_SOC"
        exit
    fi
fi

case ${TARGET_SOC} in
    rk3588)
        ;;
    rk3576)
        ;;
    rk3572)
        ;;
    x86)
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3588/rk3576/rk3572/x86"
        exit -1
        ;;
esac

TARGET_PLATFORM=${TARGET_SOC}_linux
if [[ -n ${TARGET_ARCH} ]];then
    TARGET_PLATFORM=${TARGET_PLATFORM}_${TARGET_ARCH}
fi

if [ "${TARGET_SOC}" = "x86" ]; then
    TARGET_PLATFORM="x86_64"
fi

ROOT_PWD=$( cd "$( dirname $0 )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}
INSTALL_DIR=${ROOT_PWD}/install/rknn3_llm_cli_${TARGET_PLATFORM}

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
if [ "${TARGET_SOC}" = "x86" ]; then
    cmake ${ROOT_PWD}/cpp \
        -DTARGET_SOC=x86 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
else
    cmake ${ROOT_PWD}/cpp \
        -DTARGET_SOC=${TARGET_SOC} \
        -DCMAKE_SYSTEM_NAME=Linux \
        -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_C_COMPILER=${CC} \
        -DCMAKE_CXX_COMPILER=${CXX} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
fi

make -j4
make install

echo ""
echo "Build done. Install at: ${INSTALL_DIR}"
