#!/bin/bash

set -e

echo "$0 $@"
while getopts ":t:b" opt; do
  case $opt in
    t)
      TARGET_SOC=$OPTARG
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
  echo "$0 -t <target> [-b <build_type>]"
  echo ""
  echo "    -t : target (rk3588/rk3576)"
  echo "    -b : build_type(Debug/Release)"
  echo "such as: $0 -t rk3588 -b Release"
  echo ""
  exit -1
fi

if [[ -z ${BUILD_TYPE} ]];then
    BUILD_TYPE=Release
fi

case ${TARGET_SOC} in
    rk3588)
        TARGET_SOC="RK3588"
        ;;
    rk3576)
        TARGET_SOC="RK3576"
        ;;
    *)
        echo "Invalid target: ${TARGET_SOC}"
        echo "Valid target: rk3588 / rk3576"
        exit -1
        ;;
esac

TARGET_PLATFORM=${TARGET_SOC}_windows
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )
BUILD_DIR=${ROOT_PWD}/build/build_${TARGET_PLATFORM}_${BUILD_TYPE}

echo "==================================="
echo "TARGET_SOC=${TARGET_SOC}"
echo "BUILD_TYPE=${BUILD_TYPE}"
echo "BUILD_DIR=${BUILD_DIR}"
echo "==================================="

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_SYSTEM_NAME=Windows \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
make -j4
make install
