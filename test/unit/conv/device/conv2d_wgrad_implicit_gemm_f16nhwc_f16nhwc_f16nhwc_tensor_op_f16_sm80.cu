/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/cutlass_unit_test.h"
#include "cutlass/cutlass.h"

#include "cutlass/conv/kernel/default_conv2d_wgrad.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)

////////////////////////////////////////////////////////////////////////////////

TEST(SM80_Device_Conv2d_Wgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16,
     128x128_64x3_64x64x64) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;
    using ElementCompute = cutlass::half_t;

    /// Device-level Conv2d instance
    using Conv2dWgradKernel =
            typename cutlass::conv::kernel::DefaultConv2dWgrad<
                    ElementA, cutlass::layout::TensorNHWC, ElementB,
                    cutlass::layout::TensorNHWC, ElementC,
                    cutlass::layout::TensorNHWC, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<128, 128, 64>,
                    cutlass::gemm::GemmShape<64, 64, 64>,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::epilogue::thread::LinearCombination<
                            ElementC,
                            128 / cutlass::sizeof_bits<ElementC>::value,
                            ElementAccumulator, ElementCompute>,
                    cutlass::gemm::threadblock::
                            GemmIdentityThreadblockSwizzle<>,
                    3, cutlass::arch::OpMultiplyAdd,
                    cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

    using Conv2dWgrad =
            cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dWgrad>());
}

TEST(SM80_Device_Conv2d_Wgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16,
     128x128_64x3_64x64x64) {
    /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;
    using ElementCompute = cutlass::half_t;

    /// Device-level Conv2d instance
    using Conv2dWgradKernel =
            typename cutlass::conv::kernel::DefaultConv2dWgrad<
                    ElementA, cutlass::layout::TensorNHWC, ElementB,
                    cutlass::layout::TensorNHWC, ElementC,
                    cutlass::layout::TensorNHWC, ElementAccumulator,
                    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<128, 128, 64>,
                    cutlass::gemm::GemmShape<64, 64, 64>,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::epilogue::thread::LinearCombination<
                            ElementC,
                            128 / cutlass::sizeof_bits<ElementC>::value,
                            ElementAccumulator, ElementCompute>,
                    cutlass::gemm::threadblock::
                            GemmIdentityThreadblockSwizzle<>,
                    3, cutlass::arch::OpMultiplyAdd,
                    cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

    using Conv2dWgrad =
            cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

    /// Run all unit test sizes with device-level Conv2d instance
    EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dWgrad>());
}

////////////////////////////////////////////////////////////////////////////////
#endif  // CUTLASS_ARCH_MMA_SM80_SUPPORTED
