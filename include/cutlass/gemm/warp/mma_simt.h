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
    \brief Templates implementing warp-level matrix multiply-accumulate
   operations.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/thread/mma.h"

#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"
#include "cutlass/gemm/warp/mma_simt_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Data type of A elements
        typename ElementA_,
        /// Layout of A matrix (concept: MatrixLayout)
        typename LayoutA_,
        /// Data type of B elements
        typename ElementB_,
        /// Layout of B matrix (concept: MatrixLayout)
        typename LayoutB_,
        /// Element type of C matrix
        typename ElementC_,
        /// Layout of C matrix (concept: MatrixLayout)
        typename LayoutC_,
        /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
        typename Policy_,
        /// Number of partitions along K dimension
        int PartitionsK = 1,
        /// Complex transformation on operand A
        ComplexTransform TransformA = ComplexTransform::kNone,
        /// Complex transformation on operand B
        ComplexTransform TransformB = ComplexTransform::kNone,
        /// Used for partial specialization
        typename Enable = bool>
class MmaSimt {
public:
    /// Shape of warp-level matrix operation (concept: GemmShape)
    using Shape = Shape_;

    /// Data type of multiplicand A
    using ElementA = ElementA_;

    /// Layout of multiplicand A
    using LayoutA = LayoutA_;

    /// Data type of multiplicand B
    using ElementB = ElementB_;

    /// Layout of multiplicand B
    using LayoutB = LayoutB_;

    /// Data type of accumulator matrix C
    using ElementC = ElementC_;

    /// Layout of accumulator matrix C
    using LayoutC = LayoutC_;

    /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
    using Policy = Policy_;

    /// Indicates class of matrix operator
    using OperatorClass = arch::OpClassSimt;

    /// Hard-coded for now
    using ArchTag = arch::Sm50;

    /// Complex transform on A operand
    static ComplexTransform const kTransformA = TransformA;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB = TransformB;

    /// Layout of threads
    using ThreadLayoutA = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutA>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutA>::value,
                    layout::RowMajor, LayoutA>::type>::type;

    using ThreadLayoutB = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutB>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutB>::value,
                    layout::RowMajor, LayoutB>::type>::type;

    static constexpr bool use_dp4a =
            (platform::is_same<layout::ColumnMajorInterleaved<4>,
                               LayoutA>::value ||
             platform::is_same<layout::RowMajorInterleaved<4>,
                               LayoutA>::value) &&
            platform::is_same<ElementA, int8_t>::value &&
            platform::is_same<ElementB, int8_t>::value;

    using dp4a_type =
            typename platform::conditional<use_dp4a, int8_t, bool>::type;

    /// Thread-level matrix multiply accumulate operator
    using ThreadMma =
            thread::Mma<GemmShape<Shape::kM / Policy::WarpShape::kRow,
                                  Shape::kN / Policy::WarpShape::kColumn,
                                  Policy::LaneMmaShape::kK>,
                        ElementA, ThreadLayoutA, ElementB, ThreadLayoutB,
                        ElementC, LayoutC, arch::OpMultiplyAdd, dp4a_type>;

    /// Underlying matrix multiply operator (concept: arch::Mma)
    using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

    /// Indicates math operator
    using MathOperator = typename ArchMmaOperator::Operator;

    /// Shape of the underlying instruction
    using InstructionShape = GemmShape<1, 1, use_dp4a ? 4 : 1>;

public:
    /// Iterates over the A operand in memory
    using IteratorA = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;

    /// Storage for A tile
    using FragmentA = typename IteratorA::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentA = FragmentA;

    /// Iterates over the B operand in memory
    using IteratorB = MmaSimtTileIterator<
            MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB,
            ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;

    /// Storage for B tile
    using FragmentB = typename IteratorB::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentB = FragmentB;

    /// Iterates over the C operand in memory
    using IteratorC =
            MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC,
                                ElementC, LayoutC, Policy>;

    /// Storage for C tile
    using FragmentC = typename ThreadMma::FragmentC;

public:
    //
    // Methods
    //

    /// Ctor
    CUTLASS_DEVICE
    MmaSimt() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    CUTLASS_DEVICE
    void operator()(FragmentC& d, FragmentA a, FragmentB b, FragmentC const& c,
                    int group_idx = 0) const {
        ThreadMma mma;

        if (kTransformA == ComplexTransform::kConjugate) {
            conjugate<FragmentA> conj_a;
            a = conj_a(a);
        }

        if (kTransformB == ComplexTransform::kConjugate) {
            conjugate<FragmentB> conj_b;
            b = conj_b(b);
        }

        mma(d, a, b, c);
    }

    /// Transform the mma operands to the required types
    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
        // TODO: Implement this
        dst_A = A;
        dst_B = B;
    }
};

template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Data type of A elements
        typename ElementA_,
        /// Layout of A matrix (concept: MatrixLayout)
        typename LayoutA_,
        /// Data type of B elements
        typename ElementB_,
        /// Layout of B matrix (concept: MatrixLayout)
        typename LayoutB_,
        /// Data type of mask first elements
        typename ElementMaskFirst_,
        /// Layout of mask first matrix (concept: MatrixLayout)
        typename LayoutMaskFirst_,
        /// Data type of mask second elements
        typename ElementMaskSecond_,
        /// Layout of mask second matrix (concept: MatrixLayout)
        typename LayoutMaskSecond_,
        /// Element type of C matrix
        typename ElementC_,
        /// Layout of C matrix (concept: MatrixLayout)
        typename LayoutC_,
        /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
        typename Policy_,
        /// Number of partitions along K dimension
        int PartitionsK = 1,
        /// Complex transformation on operand A
        ComplexTransform TransformA = ComplexTransform::kNone,
        /// Complex transformation on operand B
        ComplexTransform TransformB = ComplexTransform::kNone,
        /// Used for partial specialization
        typename Enable = bool,
        /// for conv, if B is not src, exchange the define of mask first and
        /// mask second
        bool BisSrc = true>
class RegionRestrictedMmaSimt {
public:
    /// Shape of warp-level matrix operation (concept: GemmShape)
    using Shape = Shape_;

    /// Data type of multiplicand A
    using ElementA = ElementA_;

    /// Layout of multiplicand A
    using LayoutA = LayoutA_;

    /// Data type of multiplicand B
    using ElementB = ElementB_;

    /// Layout of multiplicand B
    using LayoutB = LayoutB_;

    /// Data type of mask first
    using ElementMaskFirst = ElementMaskFirst_;

    /// Layout of mask first
    using LayoutMaskFirst = LayoutMaskFirst_;

    /// Data type of mask second
    using ElementMaskSecond = ElementMaskSecond_;

    /// Layout of mask second
    using LayoutMaskSecond = LayoutMaskSecond_;

    /// Data type of accumulator matrix C
    using ElementC = ElementC_;

    /// Layout of accumulator matrix C
    using LayoutC = LayoutC_;

    /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
    using Policy = Policy_;

    /// Indicates class of matrix operator
    using OperatorClass = arch::OpClassSimt;

    /// Hard-coded for now
    using ArchTag = arch::Sm50;

    /// Complex transform on A operand
    static ComplexTransform const kTransformA = TransformA;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB = TransformB;

    /// Layout of threads
    using ThreadLayoutA = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutA>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutA>::value,
                    layout::RowMajor, LayoutA>::type>::type;

    using ThreadLayoutB = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutB>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutB>::value,
                    layout::RowMajor, LayoutB>::type>::type;

    static constexpr bool use_dp4a =
            (platform::is_same<layout::ColumnMajorInterleaved<4>,
                               LayoutA>::value ||
             platform::is_same<layout::RowMajorInterleaved<4>,
                               LayoutA>::value) &&
            platform::is_same<ElementA, int8_t>::value &&
            platform::is_same<ElementB, int8_t>::value;

    using dp4a_type =
            typename platform::conditional<use_dp4a, int8_t, bool>::type;

    /// Thread-level matrix multiply accumulate operator
    using ThreadMma = thread::RegionRestrictedMma<
            GemmShape<Shape::kM / Policy::WarpShape::kRow,
                      Shape::kN / Policy::WarpShape::kColumn,
                      Policy::LaneMmaShape::kK>,
            ElementA, ThreadLayoutA, ElementB, ThreadLayoutB, ElementMaskFirst,
            LayoutMaskFirst, ElementMaskSecond, LayoutMaskSecond, ElementC,
            LayoutC, arch::OpMultiplyAdd, dp4a_type>;

    /// Underlying matrix multiply operator (concept: arch::Mma)
    using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

    /// Indicates math operator
    using MathOperator = typename ArchMmaOperator::Operator;

    /// Shape of the underlying instruction
    using InstructionShape = GemmShape<1, 1, use_dp4a ? 4 : 1>;

public:
    /// Iterates over the A operand in memory
    using IteratorA = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;

    /// Storage for A tile
    using FragmentA = typename IteratorA::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentA = FragmentA;

    /// Iterates over the B operand in memory
    using IteratorB = MmaSimtTileIterator<
            MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB,
            ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;

    /// Storage for B tile
    using FragmentB = typename IteratorB::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentB = FragmentB;

    /// Iterates over the mask first in memory
    using IteratorMaskFirst = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementMaskFirst, LayoutMaskFirst, Policy, PartitionsK, Shape::kK>;

    /// Storage for mask first tile
    using FragmentMaskFirst = typename IteratorMaskFirst::Fragment;

    /// Iterates over the mask second in memory
    using IteratorMaskSecond = MmaSimtTileIterator<
            MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB,
            ElementMaskSecond, LayoutMaskSecond, Policy, PartitionsK,
            Shape::kK>;

    /// Storage for mask second tile
    using FragmentMaskSecond = typename IteratorMaskSecond ::Fragment;

    /// Iterates over the C operand in memory
    using IteratorC =
            MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC,
                                ElementC, LayoutC, Policy>;

    /// Storage for C tile
    using FragmentC = typename ThreadMma::FragmentC;

public:
    //
    // Methods
    //

    /// Ctor
    CUTLASS_DEVICE
    RegionRestrictedMmaSimt() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    CUTLASS_DEVICE
    void operator()(FragmentC& d, FragmentA a, FragmentB b,
                    FragmentMaskFirst const& maskFirst,
                    FragmentMaskSecond const& maskSecond, FragmentC const& c,
                    int group_idx = 0) const {
        ThreadMma mma;

        if (kTransformA == ComplexTransform::kConjugate) {
            conjugate<FragmentA> conj_a;
            a = conj_a(a);
        }

        if (kTransformB == ComplexTransform::kConjugate) {
            conjugate<FragmentB> conj_b;
            b = conj_b(b);
        }

        mma(d, a, b, maskFirst, maskSecond, c);
    }

    /// Transform the mma operands to the required types
    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
        // TODO: Implement this
        dst_A = A;
        dst_B = B;
    }
};

template <
        /// Size of the Gemm problem - concept: gemm::GemmShape<>
        typename Shape_,
        /// Data type of A elements
        typename ElementA_,
        /// Layout of A matrix (concept: MatrixLayout)
        typename LayoutA_,
        /// Data type of B elements
        typename ElementB_,
        /// Layout of B matrix (concept: MatrixLayout)
        typename LayoutB_,
        /// Data type of mask first elements
        typename ElementMaskInput_,
        /// Layout of mask first matrix (concept: MatrixLayout)
        typename LayoutMaskInput_,
        /// Data type of mask second elements
        typename ElementMaskOutput_,
        /// Layout of mask second matrix (concept: MatrixLayout)
        typename LayoutMaskOutput_,
        /// Element type of C matrix
        typename ElementC_,
        /// Layout of C matrix (concept: MatrixLayout)
        typename LayoutC_,
        /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
        typename Policy_,
        /// Number of partitions along K dimension
        int PartitionsK = 1,
        /// Complex transformation on operand A
        ComplexTransform TransformA = ComplexTransform::kNone,
        /// Complex transformation on operand B
        ComplexTransform TransformB = ComplexTransform::kNone,
        /// Used for partial specialization
        typename Enable = bool,
        /// for conv, if B is not src, exchange the define of mask first and
        /// mask second
        bool BisSrc = true>
class RegionRestrictedMmaSimtFprop {
public:
    /// Shape of warp-level matrix operation (concept: GemmShape)
    using Shape = Shape_;

    /// Data type of multiplicand A
    using ElementA = ElementA_;

    /// Layout of multiplicand A
    using LayoutA = LayoutA_;

    /// Data type of multiplicand B
    using ElementB = ElementB_;

    /// Layout of multiplicand B
    using LayoutB = LayoutB_;

    /// Data type of mask first
    using ElementMaskInput = ElementMaskInput_;

    /// Layout of mask first
    using LayoutMaskInput = LayoutMaskInput_;

    /// Data type of mask second
    using ElementMaskOutput = ElementMaskOutput_;

    /// Layout of mask second
    using LayoutMaskOutput = LayoutMaskOutput_;

    /// Data type of accumulator matrix C
    using ElementC = ElementC_;

    /// Layout of accumulator matrix C
    using LayoutC = LayoutC_;

    /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
    using Policy = Policy_;

    /// Indicates class of matrix operator
    using OperatorClass = arch::OpClassSimt;

    /// Hard-coded for now
    using ArchTag = arch::Sm50;

    /// Complex transform on A operand
    static ComplexTransform const kTransformA = TransformA;

    /// Complex transform on B operand
    static ComplexTransform const kTransformB = TransformB;

    /// Layout of threads
    using ThreadLayoutA = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutA>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutA>::value,
                    layout::RowMajor, LayoutA>::type>::type;

    using ThreadLayoutB = typename platform::conditional<
            platform::is_same<layout::ColumnMajorInterleaved<4>,
                              LayoutB>::value,
            layout::ColumnMajor,
            typename platform::conditional<
                    platform::is_same<layout::RowMajorInterleaved<4>,
                                      LayoutB>::value,
                    layout::RowMajor, LayoutB>::type>::type;

    static constexpr bool use_dp4a =
            (platform::is_same<layout::ColumnMajorInterleaved<4>,
                               LayoutA>::value ||
             platform::is_same<layout::RowMajorInterleaved<4>,
                               LayoutA>::value) &&
            platform::is_same<ElementA, int8_t>::value &&
            platform::is_same<ElementB, int8_t>::value;

    using dp4a_type =
            typename platform::conditional<use_dp4a, int8_t, bool>::type;

    /// Thread-level matrix multiply accumulate operator
    using ThreadMma = thread::RegionRestrictedMmaFprop<
            GemmShape<Shape::kM / Policy::WarpShape::kRow,  // warp shape in
                                                            // units of elements
                                                            // / warp shape in
                                                            // units of threads
                      Shape::kN / Policy::WarpShape::kColumn,
                      Policy::LaneMmaShape::kK>,
            ElementA, ThreadLayoutA, ElementB, ThreadLayoutB, ElementMaskInput,
            LayoutMaskInput, ElementMaskOutput, LayoutMaskOutput, ElementC,
            LayoutC, arch::OpMultiplyAdd, dp4a_type>;

    /// Underlying matrix multiply operator (concept: arch::Mma)
    using ArchMmaOperator = typename ThreadMma::ArchMmaOperator;

    /// Indicates math operator
    using MathOperator = typename ArchMmaOperator::Operator;

    /// Shape of the underlying instruction
    using InstructionShape = GemmShape<1, 1, use_dp4a ? 4 : 1>;

public:
    /// Iterates over the A operand in memory
    using IteratorA = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementA, LayoutA, Policy, PartitionsK, Shape::kK>;

    /// Storage for A tile
    using FragmentA = typename IteratorA::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentA = FragmentA;

    /// Iterates over the B operand in memory
    using IteratorB = MmaSimtTileIterator<
            MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>, Operand::kB,
            ElementB, LayoutB, Policy, PartitionsK, Shape::kK>;

    /// Storage for B tile
    using FragmentB = typename IteratorB::Fragment;

    /// Storage for transformed A tile
    using TransformedFragmentB = FragmentB;

    /// Iterates over the B in memory
    using IteratorMaskInput = MmaSimtTileIterator<
            MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>, Operand::kA,
            ElementMaskInput, LayoutMaskInput, Policy, PartitionsK, Shape::kK>;

    /// Storage for mask first tile
    using FragmentMaskInput = typename IteratorMaskInput::Fragment;

    using IteratorMaskOutput =
            MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC,
                                ElementMaskOutput, LayoutMaskOutput, Policy>;

    /// Storage for mask second tile
    using FragmentMaskOutput = typename IteratorMaskOutput::Fragment;

    /// Iterates over the C operand in memory
    using IteratorC =
            MmaSimtTileIterator<MatrixShape<Shape::kM, Shape::kN>, Operand::kC,
                                ElementC, LayoutC, Policy>;

    /// Storage for C tile
    using FragmentC = typename ThreadMma::FragmentC;

public:
    //
    // Methods
    //

    /// Ctor
    CUTLASS_DEVICE
    RegionRestrictedMmaSimtFprop() {}

    /// Performs a warp-level matrix multiply-accumulate operation
    CUTLASS_DEVICE
    void operator()(FragmentC& d, FragmentA a, FragmentB b,
                    FragmentMaskInput const& maskInput,
                    FragmentMaskOutput const& maskOutput, FragmentC const& c,
                    int group_idx = 0) const {
        ThreadMma mma;

        if (kTransformA == ComplexTransform::kConjugate) {
            conjugate<FragmentA> conj_a;
            a = conj_a(a);
        }

        if (kTransformB == ComplexTransform::kConjugate) {
            conjugate<FragmentB> conj_b;
            b = conj_b(b);
        }

        mma(d, a, b, maskInput, maskOutput, c);
    }

    /// Transform the mma operands to the required types
    CUTLASS_DEVICE
    void transform(TransformedFragmentA& dst_A, TransformedFragmentB& dst_B,
                   FragmentA const& A, FragmentB const& B) const {
        // TODO: Implement this
        dst_A = A;
        dst_B = B;
    }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass
