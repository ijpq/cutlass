#!/bin/bash -e

cd $(dirname $0)

mkdir -p build_cmake
cd build_cmake
[ -d gtest ] || git clone git@git-core.megvii-inc.com:third-party/gtest.git

cmake_command="cmake ../.. \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUTLASS_NVCC_ARCHS=75 \
    -DGOOGLETEST_DIR=$(pwd)/gtest/googletest"
echo "run cmake: $cmake_command"
bash -c "$cmake_command"
exec make -j$(nproc)
