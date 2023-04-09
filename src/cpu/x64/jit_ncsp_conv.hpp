/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef CPU_X64_JIT_NCSP_CONV_HPP
#define CPU_X64_JIT_NCSP_CONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct ncsp_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , with_sum_(attr->post_ops_.find(primitive_kind::sum) != -1) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ncsp_convolution_fwd_t);

        status_t init(engine_t *engine);

        std::shared_ptr<primitive_desc_t> nspc_conv_pd_;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_pre_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_post_reorder_pd_;
        memory_desc_t nspc_src_md_;
        memory_desc_t nspc_dst_md_;

    private:
        const bool with_sum_;
        std::string name_ = "ncsp:any+";
        void init_name() { name_.append(nspc_conv_pd_->name()); }
        void init_scratchpad();
    };

    ncsp_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {};

    ~ncsp_convolution_fwd_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> prim, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> nspc_conv_p_;
    std::shared_ptr<primitive_t> src_reorder_p_;
    std::shared_ptr<primitive_t> dst_pre_reorder_p_;
    std::shared_ptr<primitive_t> dst_post_reorder_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
