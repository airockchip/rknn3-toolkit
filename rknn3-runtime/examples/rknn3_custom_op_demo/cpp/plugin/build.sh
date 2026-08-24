#!/bin/bash

set -e

# ============================================================
# Usage: ./build.sh <target>
#   target: android | riscv
# ============================================================

# Project root (parent of cpp/)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INSTALL_DIR="${PROJECT_ROOT}/install"

# rknn3_api.h location: relative to this script
#   plugin/ -> cpp/ -> rknn3_custom_op_demo/ -> examples/ -> rknn3-runtime/ -> rknn3-api/include
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RKNN3_INCLUDE="${SCRIPT_DIR}/../../../../rknn3-api/include"

if [ ! -f "${RKNN3_INCLUDE}/rknn3_api.h" ]; then
    echo "ERROR: Cannot find rknn3_api.h at ${RKNN3_INCLUDE}"
    exit 1
fi

TARGET_LIB="librknn_custom_op.so"
TARGET="${1}"

if [ "${TARGET}" != "android" ] && [ "${TARGET}" != "riscv" ]; then
    echo "Usage: $0 <android|riscv>"
    exit 1
fi

# ============================================================
# Android (arm64-v8a)
# ============================================================
if [ "${TARGET}" == "android" ]; then
    API="21"
    ARCH="aarch64"
    NDK="${ANDROID_NDK_PATH:-~/opt/android-ndk-r23b}"
    NDK=$(eval echo "${NDK}")  # expand ~ and vars

    if [ ! -d "${NDK}" ]; then
        echo "ERROR: Android NDK not found at '${NDK}'"
        echo "Please set ANDROID_NDK_PATH, e.g.:"
        echo "  export ANDROID_NDK_PATH=~/opt/android-ndk-r23b"
        exit 1
    fi

    TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/linux-x86_64"
    CC="${TOOLCHAIN}/bin/${ARCH}-linux-android${API}-clang"

    if [ ! -x "${CC}" ]; then
        echo "ERROR: Android clang not found: ${CC}"
        echo "NDK='${NDK}' (API=${API}, ARCH=${ARCH})"
        exit 1
    fi

    CFLAGS="-fPIC -O2 -g2 -D_POSIX_C_SOURCE=1 -I${RKNN3_INCLUDE} -march=armv8-a"
    LDFLAGS="-fPIC -shared -Wl,--hash-style=both -lm"
fi

# ============================================================
# RISC-V (c908v)
# ============================================================
if [ "${TARGET}" == "riscv" ]; then
    CC="riscv64-unknown-elf-gcc"
    STRIP="riscv64-unknown-elf-strip"
    CC_PATH=$(which ${CC} 2>/dev/null)

    if [ x"${CC_PATH}" == x"" ]; then
        echo "Can not find riscv64-unknown-elf-gcc, trying toolkit3..."
        if command -v bash &>/dev/null; then
            CC_PATH=$(bash -lic 'toolkit3 && which riscv64-unknown-elf-gcc' 2>/dev/null)
        fi
        if [ -z "${CC_PATH}" ]; then
            echo "ERROR: riscv64-unknown-elf-gcc not found. Run 'toolkit3' first."
            exit 1
        fi
    fi

    CFLAGS="-mcpu=c908v -mrvv-v0p10-compatible -march=rv64gcv -mabi=lp64d -O2 -g2 -mcmodel=medany -fpic -fno-plt -D_POSIX_C_SOURCE=1 -I${RKNN3_INCLUDE}"
    LDFLAGS="-mcpu=c908v -O2 -g2 -Wl,-r,-z,max-page-size=1024 -fpic -nostartfiles -nostdlib -static-libgcc -e 0 -lm"
fi

echo "============================================"
echo "Building ${TARGET_LIB} for ${TARGET}"
echo "CC:      ${CC}"
echo "INCLUDE: ${RKNN3_INCLUDE}"
echo "============================================"

# --- clean ---
rm -f *.o *.so

# --- compile ---
${CC} -o rknn_custom_op_impl.o     -c rknn_custom_op_impl.c     ${CFLAGS} -I.
${CC} -o rknn3_custom_op.o  -c rknn3_custom_op.c  ${CFLAGS} -I.

# --- link ---
${CC} -o "${TARGET_LIB}" ${LDFLAGS} *.o

# --- strip ---
if [ "${TARGET}" == "android" ]; then
    echo "Skip strip for android"
else
    ${STRIP} --strip-unneeded -R .hash "${TARGET_LIB}"
fi

# --- clean .o ---
rm -f *.o

# --- install to install/ directory ---
mkdir -p "${INSTALL_DIR}/lib"
cp "${TARGET_LIB}" "${INSTALL_DIR}/lib"

echo "Build ${TARGET_LIB} for ${TARGET} done"
echo "Installed to: ${INSTALL_DIR}/${TARGET_LIB}"
