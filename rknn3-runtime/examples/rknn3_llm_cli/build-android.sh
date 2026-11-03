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
  echo "    -t : target (rk3588/rk3576/rk3572)"
  echo "    -a : arch (arm64-v8a)"
  echo "    -b : build_type(Debug/Release)"
  echo "such as: $0 -t rk3588 -a arm64-v8a -b Release"
  echo ""
  exit -1
fi

if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

if [[ -z ${ANDROID_NDK_PATH} ]];then
    echo "Please set ANDROID_NDK_PATH"
    echo "such as export ANDROID_NDK_PATH=~/opts/ndk/android-ndk-r21e"
    exit
fi

case ${TARGET_SOC} in
    rk3588)
        ;;
    rk3576)
        ;;
    rk3572)
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3588/rk3576/rk3572"
        exit -1
        ;;
esac

case ${TARGET_ARCH} in
    arm64-v8a)
        ;;
    armeabi-v7a)
        ;;
    *)
        echo "Invalid arch: ${TARGET_ARCH}"
        echo "Valid arch: arm64-v8a/armeabi-v7a"
        exit -1
        ;;
esac

TARGET_PLATFORM=${TARGET_SOC}_android_${TARGET_ARCH}

ROOT_PWD=$( cd "$( dirname $0 )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}
INSTALL_DIR=${ROOT_PWD}/install/rknn3_llm_cli_${TARGET_PLATFORM}

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "INSTALL_DIR=${INSTALL_DIR}"
echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

if [[ -d "${INSTALL_DIR}" ]]; then
  rm -rf ${INSTALL_DIR}
fi

cd ${BUILD_DIR}
cmake ${ROOT_PWD}/cpp \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_PATH}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${TARGET_ARCH} \
    -DANDROID_PLATFORM=android-24 \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}

make -j4
make install

echo ""
echo "Build done. Install at: ${INSTALL_DIR}"
