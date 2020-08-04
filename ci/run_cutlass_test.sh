#!/bin/bash -e

cd $(dirname $0)

cutlass_tests=$(find ./build_cmake/test/unit -name "cutlass_test_unit*" | \
    grep -v "dir")
for cutlass_test in $cutlass_tests
do
    echo "run $cutlass_test"
    $cutlass_test
done
