/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
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
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Templates implementing loading of tiles from pitch-linear rank=2
   tensors.

    This iterator uses masks to guard out-of-bounds accesses and visits the last
   "residue" tile first, with the objective of minimizing predicate mask updates
   during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

/**
 * \file include/cutlass/convolution/threadblock/conv2d_tile_params.h
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {

enum class ImplicitGemmMode { GEMM_NT, GEMM_TN };
namespace threadblock {

enum class TileMapType {
    kRow2C_Col2N,    ///< Row map to channel, Column map to batch
    kRow2CHW_Col2N,  ///< Row map to CHW(channel x height x width), Column
                     ///< map to batch
    kRow2C_Col2NHW,  ///< Row map to channel, column map to NHW (batch x
                     ///< height x width)
    kRow2NHW_Col2C   ///< Row map to NHW(batch x height x width), Column
                     ///< map to channel
};

template <typename Layout, TileMapType tile_map_type_>
struct TileMap;

template <int Interleave>
struct TileMap<layout::TensorCxRSKx<Interleave>, TileMapType::kRow2C_Col2N> {
    using Layout = layout::TensorCxRSKx<Interleave>;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 2;
    CUTLASS_HOST_DEVICE
    TileMap() {}
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        return TensorCoord{coord.column(), 0, 0, coord.row()};
    }
};

template <int Interleave>
struct TileMap<layout::TensorCxRSKx<Interleave>, TileMapType::kRow2CHW_Col2N> {
    using Layout = layout::TensorCxRSKx<Interleave>;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 0;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = coord.column(), h = 0, w = 0, c = 0;
        int tmp = 0;
        fast_divmod(c, tmp, coord.row(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

template <int Interleave>
struct TileMap<layout::TensorKxRSCx<Interleave>, TileMapType::kRow2NHW_Col2C> {
    using Layout = layout::TensorKxRSCx<Interleave>;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 0;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = 0, h = 0, w = 0, c = coord.column();
        int tmp = 0;
        fast_divmod(n, tmp, coord.row(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

template <typename Layout_>
struct TileMap<Layout_, TileMapType::kRow2C_Col2NHW> {
    using Layout = Layout_;
    using Index = typename Layout::Index;
    using TensorCoord = typename Layout::TensorCoord;
    static const int kStrideAxis = 1;
    Index hw_, w_;
    unsigned int hw_mul_, hw_shr_, w_mul_, w_shr_;
    CUTLASS_HOST_DEVICE
    TileMap() : hw_(0), w_(0), hw_mul_(0), hw_shr_(0), w_mul_(0), w_shr_(0) {}
    CUTLASS_HOST_DEVICE
    TileMap(Index hw, Index w) : hw_(hw), w_(w) {
        find_divisor(hw_mul_, hw_shr_, hw_);
        find_divisor(w_mul_, w_shr_, w_);
    }
    CUTLASS_HOST_DEVICE
    TensorCoord operator()(MatrixCoord const& coord) const {
        int n = 0, h = 0, w = 0, c = coord.row();
        int tmp = 0;
        fast_divmod(n, tmp, coord.column(), hw_, hw_mul_, hw_shr_);
        fast_divmod(h, w, tmp, w_, w_mul_, w_shr_);
        return TensorCoord{n, h, w, c};
    }
};

struct ExtraParamZeroPoint {
    uint8_t src_zero_point;

    CUTLASS_HOST_DEVICE
    ExtraParamZeroPoint() : src_zero_point(0) {}

    CUTLASS_HOST_DEVICE
    ExtraParamZeroPoint(uint8_t src_zero_point_)
            : src_zero_point(src_zero_point_) {}
};

namespace detail {
template <typename Element, typename ExtraParam>
CUTLASS_HOST_DEVICE uint32_t prepare_pack_pad(const ExtraParam& params) {
    return 0;
}

template <>
CUTLASS_HOST_DEVICE uint32_t prepare_pack_pad<uint4b_t, ExtraParamZeroPoint>(
        const ExtraParamZeroPoint& params) {
    uint32_t ret = 0;
    for (size_t i = 0; i < 8; i++) {
        ret |= params.src_zero_point << (4 * i);
    }
    return ret;
}
}  // namespace detail

}  // namespace threadblock
}  // namespace conv
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
