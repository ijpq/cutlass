#!/bin/bash -e

cd $(dirname $0)

exclude_name1="complex"
exclude_name2="Complex"

cutlass_tests=$(find ./build_cmake/test/unit -name "cutlass_test_unit*" | \
  grep -v "dir" | grep -v "convolution")
for cutlass_test in $cutlass_tests
do
  if [[ "$cutlass_test" == *"$exclude_name1"* || "$cutlass_test" == *"$exclude_name2"* ]]; then
   echo "exclude test $cutlass_test"
   continue
  else
    echo "run $cutlass_test"
    $cutlass_test
  fi

done
