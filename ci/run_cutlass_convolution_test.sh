#!/bin/bash -e

# get device information and exclude unsupported test cases
nvcc ./ci/get_exclude_info.cu -o ./ci/get_exclude_info
EXCLUDE_TERMS=$(./ci/get_exclude_info)
echo Excluded Terms: "${EXCLUDE_TERMS}"

# init gtest-parallel
git submodule sync
git submodule update --init third-party/gtest-parallel

GTEST_PARALLEL=$(readlink -f $(dirname $0)/../third-party/gtest-parallel/gtest-parallel)

cd $(dirname $0)

WORKERS=$(cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l)

CUTLASS_CONVOLUTION_TEST=./build_cmake/test/unit/convolution/device/cutlass_test_unit_convolution_device
# do not run performance test
${GTEST_PARALLEL} ${CUTLASS_CONVOLUTION_TEST} --gtest_filter="-*perf*:${EXCLUDE_TERMS}" --workers=${WORKERS}
