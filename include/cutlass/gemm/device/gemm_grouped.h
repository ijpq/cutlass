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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions,
  batched strided, and batched array variants.
*/

#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GEMM Grouped
template <typename GemmKernel_>
class GemmGrouped {
public:
    using GemmKernel = GemmKernel_;
    using ThreadblockShape = typename GemmKernel::Mma::Shape;

    using ElementA = typename GemmKernel::ElementA;
    using LayoutA = typename GemmKernel::LayoutA;
    using TensorRefA = TensorRef<ElementA const, LayoutA>;
    static ComplexTransform const kTransformA = GemmKernel::kTransformA;

    using ElementB = typename GemmKernel::ElementB;
    using LayoutB = typename GemmKernel::LayoutB;
    using TensorRefB = TensorRef<ElementB const, LayoutB>;
    static ComplexTransform const kTransformB = GemmKernel::kTransformB;

    using ElementC = typename GemmKernel::ElementC;
    using LayoutC = typename GemmKernel::LayoutC;
    using TensorRefC = TensorRef<ElementC const, LayoutC>;
    using TensorRefD = TensorRef<ElementC, LayoutC>;

    using ElementAccumulator =
            typename GemmKernel::Mma::Policy::Operator::ElementC;

    using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
    using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
    using Operator = typename GemmKernel::Operator;

    /// Argument structure
    using Arguments = typename GemmKernel::Arguments;

protected:
    /// Kernel parameters object
    typename GemmKernel::Params params_;

public:
    /// Constructs the GEMM.
    GemmGrouped() {}

    /// Determines whether the GEMM can execute the given problem.
    static Status can_implement(Arguments const& args) {
        // Determine grid shape
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int gemm_k_size = 0;

        get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

        ThreadblockSwizzle threadblock_swizzle;
        dim3 grid = threadblock_swizzle.get_grid_shape(grid_tiled_shape);

        if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
              grid.z <= std::numeric_limits<uint16_t>::max())) {
            return Status::kErrorInvalidProblem;
        }

        return GemmKernel::can_implement(args);
    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args) {
        // This kerenl does not utilize a workspace
        return size_t();
    }

    /// Computes the grid shape
    static dim3 get_grid_shape(Arguments const& args) {
        return dim3(args.threadblock_count, 1, 1);
    }

    /// Computes the maximum number of active blocks per multiprocessor
    static int maximum_active_blocks(int smem_capacity = -1) {
        CUTLASS_TRACE_HOST("GemmUniversalBase::maximum_active_blocks()");

        int max_active_blocks = -1;
        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

        if (smem_size <= (48 << 10)) {
            cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_active_blocks, Kernel<GemmKernel>,
                    GemmKernel::kThreadCount, smem_size);

            if (result == cudaSuccess) {
                CUTLASS_TRACE_HOST(
                        "  max_active_blocks: " << max_active_blocks);
                return max_active_blocks;
            }
        } else {
            // Query assuming zero shared memory then compute occupancy limit
            // based on SMEM
            cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_active_blocks, Kernel<GemmKernel>,
                    GemmKernel::kThreadCount, 0);

            if (result != cudaSuccess) {
                CUTLASS_TRACE_HOST(
                        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() "
                        "returned error "
                        << cudaGetErrorString(result));

                return -1;
            }

            if (smem_capacity < 0) {
                int device_idx = 0;
                result = cudaGetDevice(&device_idx);

                if (result != cudaSuccess) {
                    return -1;
                }

                cudaDeviceProp properties;
                result = cudaGetDeviceProperties(&properties, device_idx);

                if (result != cudaSuccess) {
                    return -1;
                }

                smem_capacity =
                        static_cast<int>(properties.sharedMemPerMultiprocessor);
            }

            int occupancy =
                    std::min(max_active_blocks, smem_capacity / smem_size);

            CUTLASS_TRACE_HOST("  occupancy: " << occupancy);

            return occupancy;
        }

        CUTLASS_TRACE_HOST("  returning internal error");

        return -1;
    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        CUTLASS_TRACE_HOST("GemmUniversalBase::initialize() - workspace "
                           << workspace
                           << ", stream: " << (stream ? "non-null" : "null"));

        // Workspace
        size_t workspace_bytes = get_workspace_size(args);

        if (workspace_bytes && !workspace) {
            return Status::kErrorWorkspaceNull;
        }

        // Initialize the Params structure
        params_ = typename GemmKernel::Params(args, workspace);

        // Specify shared memory capacity for kernel.
        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            cudaError_t result = cudaFuncSetAttribute(
                    Kernel<GemmKernel>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return Status::kErrorInternal;
            }
        }

        return Status::kSuccess;
    }

    /// Lightweight update given a subset of arguments
    Status update(Arguments const& args, void* workspace = nullptr) {
        size_t workspace_bytes = get_workspace_size(args);

        if (workspace_bytes && !workspace) {
            return Status::kErrorWorkspaceNull;
        }

        params_.update(args, workspace);

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr) {
        //
        // Configure grid and block dimensions
        //

        if (!params_.problem_visitor.problem_count) {
            return Status::kSuccess;
        }

        dim3 grid(params_.threadblock_count, 1, 1);
        dim3 block(GemmKernel::kThreadCount, 1, 1);

        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        //
        // Launch kernel
        //

        // Launch
        cutlass::Kernel<GemmKernel>
                <<<grid, block, smem_size, stream>>>(params_);

        //
        // Query for errors
        //
        cudaError_t result = cudaGetLastError();

        if (result != cudaSuccess) {
            CUTLASS_TRACE_HOST("  grid launch failed with error "
                               << cudaGetErrorString(result));
            return Status::kErrorInternal;
        }

        return Status::kSuccess;
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr,
                      cudaStream_t stream = nullptr) {
        Status status = initialize(args, workspace, stream);

        if (status == Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
