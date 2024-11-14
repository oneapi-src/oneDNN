/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_REF_REDUCTION_HPP
#define GPU_GENERIC_SYCL_REF_REDUCTION_HPP

#include "common/primitive_desc_iterator.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"
#include "gpu/gpu_reduction_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct reduction_sizes_t {
    size_t input_size = 0;
    size_t reduce_size = 0;
    size_t output_size = 0;
};

struct range_t {
    int x, y, z;
};

struct ref_reduction_t : public gpu::generic::sycl::primitive_t {
    using gpu::generic::sycl::primitive_t::primitive_t;

    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("dpcpp:ref:any", ref_reduction_t);

        status_t init(impl::engine_t *engine) {
            using sm = primitive_attr_t::skip_mask_t;

            memory_desc_wrapper src_wrap(src_md());
            memory_desc_wrapper dst_wrap(dst_md());

            bool ok = set_default_params() == status::success
                    && attr()->has_default_values(sm::post_ops)
                    && sycl_post_ops_t::post_ops_ok(attr())
                    && attr_.set_default_formats(dst_md()) == status::success
                    && src_wrap.is_plain() && dst_wrap.is_plain()
                    && src_wrap.is_dense() && dst_wrap.is_dense()
                    && src_wrap.ndims() == dst_wrap.ndims()
                    && src_wrap.ndims() <= xpu::sycl::md_t::max_dims
                    && md_dims_in_range(src_md()) && md_dims_in_range(dst_md())
                    && check_work_amount(src_wrap);
            if (!ok) return status::unimplemented;

            return init_conf(engine);
        }

        std::vector<sycl_reduction_conf_t> confs_;
        std::vector<range_t> global_ranges_;
        std::vector<range_t> local_ranges_;
        std::vector<size_t> local_mem_sizes_;
        bool needs_atomic_reduction_;
        bool needs_reorder_;
        std::vector<int> squeezed_dims_;
        std::vector<int> squeezed_axes_;
        std::vector<int> out_size_vec_;
        size_t num_reductions_ = 0;
        bool multi_reduction_ = false;
        memory_desc_t scratch_md_1_, scratch_md_2_;
        memory_desc_t reorder_src_md_;
        std::shared_ptr<primitive_desc_t> reorder_pd_;

        int max_wg_size_;
        int max_sg_size_;

    private:
        bool check_work_amount(const memory_desc_wrapper &src_mdw) {
            // Arbitrary threshold
            // XXX: Refactor kernel to support larger problems
            return src_mdw.nelems() < 9000000;
        }

        reduction_sizes_t get_reduction_sizes(
                const sycl_reduction_conf_t &conf);
        void squeeze_dims_and_axes(const memory_desc_wrapper &src_wrap,
                const std::vector<bool> &axes_mask,
                std::vector<int> &squeezed_dims,
                std::vector<int> &squeezed_axis);
        std::vector<int> get_first_two_out_sizes(
                const std::vector<int> &dims, const std::vector<int> &axes);

        size_t compute_workspace_size(const std::vector<int> &dims,
                const std::vector<int> &axes, int reduce_size);
        status_t init_scratchpad();
        status_t init_out_scratchpad();
        status_t init_reorder(impl::engine_t *engine);
        status_t init_conf(impl::engine_t *engine);
    };

    status_t init(impl::engine_t *engine) override;
    status_t launch_kernel(const exec_ctx_t &ctx, sycl_reduction_conf_t &conf,
            size_t red_iter, bool &needs_reorder,
            bool &needs_atomic_reduction) const;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    kernel_t kernel_;
    kernel_t init_kernel_;
    kernel_t finalize_kernel_;
    std::shared_ptr<impl::primitive_t> reorder_p_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
