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

if [ -z ${TARGET_SOC} ];then
  echo "$0 -t <target> -a <arch> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3588 / rk3576 / rk3572)"
  echo "    -a : arch (arm64-v8a)"
  echo "    -b : build_type(Debug/Release)"
  echo "such as: $0 -t rk3588 -a arm64-v8a -b Release"
  echo ""
  exit -1
fi

# Validate target
case ${TARGET_SOC} in
    rk3588)
        TARGET_SOC="RK3588"
    ;;
    rk3576)
        TARGET_SOC="RK3576"
        ;;
    rk3572)
        TARGET_SOC="RK3572"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3588 / rk3576 / rk3572"
        exit -1
        ;;
esac

# Validate architecture
case ${TARGET_ARCH} in
    arm64-v8a)
        TARGET_ARCH="arm64-v8a"
        ;;
    *)
        echo "Invalid arch: ${TARGET_ARCH}"
        echo "Valid arch: arm64-v8a"
        exit -1
        ;;
esac

# Debug / Release
if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

if [ -z ${ANDROID_NDK_PATH} ]
then
  ANDROID_NDK_PATH=~/opt/android-ndk-r17c
fi

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

BUILD_DIR=${ROOT_PWD}/build/build_android_v8a

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "TARGET_ARCH=${TARGET_ARCH}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "ANDROID_NDK_PATH=${ANDROID_NDK_PATH}"
echo "==================================="

cd ${BUILD_DIR}
cmake ../.. \
        -DANDROID_TOOLCHAIN=clang \
        -DTARGET_SOC=${TARGET_SOC} \
       	-DCMAKE_SYSTEM_NAME=Android \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="arm64-v8a" \
        -DANDROID_STL=c++_static \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
make -j4
make install
cd ..

