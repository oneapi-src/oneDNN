/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
#define GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"
#include "oneapi/dnnl/dnnl_types.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/serialization.hpp"
#include "common/utils.hpp"
#include "gemmstone/microkernel_selector.hpp"
#include "gpu/intel/kernel_cache.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"
#include "gpu/intel/post_ops.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

#include <array>
#include <memory>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

// Which axis of the problem is partitioned by the grouped offsets array.
//   m_axis: src [total_M, K] and dst [total_M, N] grouped, wei dense 3D
//           [G, K, N]. The token count varies per expert (MoE forward).
//           Called 2Dx3D by the common layer.
//   k_axis: src [M, total_K] and wei [total_K, N] grouped with the same
//           partition, dst dense 3D [G, M, N]. The contraction dim varies
//           per group (MoE backward).
enum class grouped_axis_t { undefined = -1, m_axis, k_axis, n_axis };

// OpenCL kernel parameters
struct grouped_ocl_params_t : trivially_serializable_t<grouped_ocl_params_t> {
    grouped_axis_t axis = grouped_axis_t::undefined;
    data_type_t src_dt = data_type::undef;
    data_type_t wei_dt = data_type::undef;
    data_type_t dst_dt = data_type::undef;
    data_type_t bia_dt = data_type::undef;
    data_type_t src_scale_dt = data_type::undef;
    data_type_t src_zp_dt = data_type::undef;
    data_type_t wei_scale_dt = data_type::undef;
    data_type_t wei_zp_dt = data_type::undef;

    int src_group_k = 0;
    int wei_group_k = 0, wei_group_n = 0;

    // Classification of each post-op entry
    std::array<po_kind_t, 3> po_chain
            = {{po_kind_t::none, po_kind_t::none, po_kind_t::none}};

    bool transc = false;
    bool is_gemv = false;
    bool with_src_scale = false, with_src_zp = false;
    bool with_wei_scale = false, with_wei_zp = false;
    bool with_bias = false;
    bool with_post_op = false;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(grouped_ocl_params_t);

// Kernel cache key: the OpenCL kernel parameters and the microkernel selected
// by the pd.
struct grouped_micro_params_t : grouped_ocl_params_t {
    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> m_axis_kernel_names
                = {"grouped_micro_gemm_m_axis"};
        static const std::vector<const char *> k_axis_kernel_names
                = {"grouped_micro_gemm_k_axis"};
        return axis == grouped_axis_t::k_axis ? k_axis_kernel_names
                                              : m_axis_kernel_names;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const;

    status_t get_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
            const compute::device_info_t &dev_info) const;

    serialization_stream_t serialize() const {
        const grouped_ocl_params_t &ocl = *this;
        return serialization_stream_t(ocl, post_ops, ukernel);
    }

    gpu_post_ops_t post_ops;

    // Microkernel selected by the pd. Its options and host payload follow from
    // the device and the problem, so they are not part of the key.
    gemmstone::microkernel::GEMMCandidate ukernel;

    // Microkernel the pd generated for ukernel. It provides the launch
    // configuration and creates the kernel on a kernel cache miss. It is
    // determined by ukernel, so it is not part of the key.
    std::shared_ptr<const gemmstone::microkernel::Package> ukernel_package;
};

struct grouped_micro_gemm_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;
        const char *kernel_name_ = "grouped_gemm:micro";
        DECLARE_COMMON_PD_T(kernel_name_, grouped_micro_gemm_t);

        status_t init(const impl::engine_t *engine);
        status_t init_m_axis(const impl::engine_t *engine);
        status_t init_k_axis(const impl::engine_t *engine);
        status_t init_conf();
        status_t init_microkernel(const impl::engine_t *engine);

        bool transc() const {
            if (dst_md()->format_kind != dnnl_format_kind_sparse)
                return gemm_desc_t::get_trans(*dst_md()) == transpose::trans;
            else
                return false;
        }

        // Number of groups partitioning the grouped axis
        dim_t ngroups() const {
            return memory_desc_wrapper(src_md())
                    .sparse_desc()
                    .grouped_desc.group_count;
        }

        // Group sizes of the src quantization along m and k
        std::array<int, 2> src_group_sizes() const;

        int sg_size_ = 0;
        quantization_t src_quant_;
        quantization_t wei_quant_;
        grouped_micro_params_t conf_;
    };
    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

private:
    // grouped_micro_gemm_k_axis.cl, dispatched for the k_axis workload
    status_t execute_k_axis(const exec_ctx_t &ctx) const;
    // grouped_micro_gemm_m_axis.cl, dispatched for the m_axis workload
    status_t execute_m_axis(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace matmul

// The microkernel problem and strategy serialize one way only, so the key
// cannot be validated with a serialization round trip.
template <>
struct key_validator_t<matmul::grouped_micro_params_t> {
    static bool is_valid(const matmul::grouped_micro_params_t &) {
        return true;
    }
};
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // GPU_INTEL_MATMUL_GROUPED_MICRO_GEMM_HPP
