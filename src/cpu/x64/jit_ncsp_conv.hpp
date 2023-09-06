/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

struct ncsp_matmul_reduction_helper {
    convolution_pd_t *pd;
    ncsp_matmul_reduction_helper(convolution_pd_t *pd_) : pd(pd_) {}
    status_t reshape_activations(memory_desc_t *o_md, const memory_desc_t *i_md,
            bool to_matmul, bool is_dst) {
        dims_t reduce {};
        const dim_t ndims_out = to_matmul ? 1 + pd->with_groups() + 2
                                          : pd->with_groups() + pd->ndims();
        // convert between activations for convolution and matmul
        // batch dimension is the same for convolution and matmul
        // channel dimension of convolution is split into group and channels
        // spatial dimensions of convolution are combined into one
        // eg. {n, c, d, h, w} <-> {n, g, c/g, sp}
        if (to_matmul) {
            // conv to matmul: add batch, remove spatial
            int d = 0;
            reduce[d++] = pd->MB(); // n
            if (pd->with_groups()) reduce[d++] = pd->G(); // g
            reduce[d++] = i_md->dims[1] / pd->G(); // c/g
            reduce[d++] = pd->ID() * pd->IH() * pd->IW(); // sp
        } else {
            // matmul to conv: restore original dimensions
            const memory_desc_t *a_md
                    = is_dst ? pd->invariant_dst_md() : pd->invariant_src_md();
            for (int d = 0; d < pd->ndims(); ++d)
                reduce[d] = a_md->dims[d]; // n, c, d, h, w
        }
        return memory_desc_reshape(*o_md, *i_md, ndims_out, reduce);
    }

    status_t reshape_bias(memory_desc_t *o_md, const memory_desc_t *i_md) {
        dims_t reduce {};
        const dim_t ndims_out = 1 + pd->with_groups() + 2;
        // reshape bias from convolution to matmul
        // for matmul, batch and spatial dimensions are always 1
        // eg. {o} <-> {1, g, o/g, 1}
        int d = 0;
        reduce[d++] = 1; // b
        if (pd->with_groups()) reduce[d++] = pd->G(); // g
        reduce[d++] = i_md->dims[0] / pd->G(); // o/g
        reduce[d++] = 1; // sp
        return memory_desc_reshape(*o_md, *i_md, ndims_out, reduce);
    }

    status_t reshape_weights(
            memory_desc_t *o_md, const memory_desc_t *i_md, bool to_matmul) {
        dims_t reduce {};

        // 1 (batch) + groups + 2 (c/g and sp) for matmul
        // groups + convolution dims for convolution
        const dim_t ndims_out = to_matmul ? 1 + pd->with_groups() + 2
                                          : pd->with_groups() + pd->ndims();
        const dim_t ndims_ch = 2 + pd->with_groups();
        // this will never be the case for convolution reduction to matmul but adding in
        // for compiler errors.
        if (ndims_out > DNNL_MAX_NDIMS) return status::invalid_arguments;
        // convert between weights for convolution and matmul
        // for matmul, batch dimension b is always 1
        // eg. {g, o, i, d, h, w} <-> {b, g, o, i}
        if (to_matmul) {
            // conv to matmul: add batch, remove spatial
            reduce[0] = 1; // b
            for (int d = 0; d < ndims_ch; ++d)
                reduce[d + 1] = i_md->dims[d]; // g, oc, ic
        } else {
            // matmul to conv: remove batch, restore spatial
            for (int d = 0; d < ndims_ch; ++d)
                reduce[d] = i_md->dims[d + 1]; // g, o, i
            for (int d = ndims_ch; d < ndims_out; ++d)
                reduce[d] = 1; // d, h, w
        }
        return memory_desc_reshape(*o_md, *i_md, ndims_out, reduce);
    }

    status_t transpose(
            memory_desc_t &transposed_md, memory_desc_t &to_be_tranposed_md) {
        const int ndims = to_be_tranposed_md.ndims;
        int *perm = new int[ndims];
        for (int dim = 0; dim < ndims; dim++) {
            if (dim == ndims - 2)
                perm[dim] = dim + 1;
            else if (dim == ndims - 1)
                perm[dim] = dim - 1;
            else
                perm[dim] = dim;
        }
        return memory_desc_permute_axes(
                transposed_md, to_be_tranposed_md, perm);
    }

    bool is_gemm() {
        // 1x1
        return utils::everyone_is(1, pd->KD(), pd->KH(), pd->KW())
                // no pre-padding
                && utils::everyone_is(0, pd->padFront(), pd->padT(), pd->padL())
                // no post-padding
                && utils::everyone_is(0, pd->padBack(), pd->padB(), pd->padR())
                // no strides
                && utils::everyone_is(1, pd->KSD(), pd->KSH(), pd->KSW());
    }
};

struct ncsp_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , bias_po_(with_groups())
            , with_sum_(attr->post_ops_.find(primitive_kind::sum) != -1)
            , reduce(this) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ncsp_convolution_fwd_t);

        status_t init(engine_t *engine);
        status_t init_convolution(engine_t *engine);
        status_t init_matmul(engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_pd_;
        std::shared_ptr<primitive_desc_t> nspc_conv_pd_;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_pre_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_post_reorder_pd_;
        memory_desc_t matmul_src_md_;
        memory_desc_t matmul_wei_md_;
        memory_desc_t matmul_bia_md_;
        memory_desc_t matmul_dst_md_;
        memory_desc_t nspc_src_md_;
        memory_desc_t nspc_dst_md_;

        const bool bias_po_; // matmul with bias or not (if not, uses postops)

    private:
        bool is_matmul_;
        const bool with_sum_;
        ncsp_matmul_reduction_helper reduce;
        std::string name_ = "ncsp:tbd";
        void init_name() {
            std::string suffix = is_matmul_ ? "matmul" : "conv";
            name_ = "ncsp:" + suffix + "+";
            name_.append(
                    is_matmul_ ? matmul_pd_->name() : nspc_conv_pd_->name());
            if (!is_matmul_) {
                name_.append("+src_reorder->");
                name_.append(src_reorder_pd_->name());
                if (with_sum_) {
                    name_.append("+dst_pre_reorder->");
                    name_.append(dst_pre_reorder_pd_->name());
                }
                name_.append("+dst_post_reorder->");
                name_.append(dst_post_reorder_pd_->name());
            }
        }
        void init_scratchpad();
    };

    ncsp_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {};

    ~ncsp_convolution_fwd_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_convolution(const exec_ctx_t &ctx) const;
    status_t execute_matmul(const exec_ctx_t &ctx) const;

private:
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> prim, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> matmul_p_;
    std::shared_ptr<primitive_t> nspc_conv_p_;
    std::shared_ptr<primitive_t> src_reorder_p_;
    std::shared_ptr<primitive_t> dst_pre_reorder_p_;
    std::shared_ptr<primitive_t> dst_post_reorder_p_;
};

struct ncsp_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {

        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(adesc, attr, hint_fwd_pd)
            , reduce(this)
            , name_("tbd") {}

        ~pd_t() = default;
        DECLARE_COMMON_PD_T(name_.c_str(), ncsp_convolution_bwd_weights_t);

        status_t init(engine_t *engine);
        status_t init_convolution(engine_t *engine);

        std::shared_ptr<primitive_desc_t> nspc_conv_pd_;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;
        memory_desc_t nspc_src_md_;
        memory_desc_t nspc_diff_dst_md_;

    private:
        ncsp_matmul_reduction_helper reduce;
        bool is_matmul_ = false;
        std::string name_;
        void init_scratchpad();
        void init_name() {
            name_ = "ncsp:conv->";
            name_.append(nspc_conv_pd_->name());
            name_.append("+src_reorder->");
            name_.append(src_reorder_pd_->name());
            name_.append("+dst_reorder->");
            name_.append(dst_reorder_pd_->name());
        }
    };
    ncsp_convolution_bwd_weights_t(const pd_t *cpd) : primitive_t(cpd) {};
    ~ncsp_convolution_bwd_weights_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_convolution(const exec_ctx_t &ctx) const;

private:
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> prim, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> nspc_conv_p_;
    std::shared_ptr<primitive_t> src_reorder_p_;
    std::shared_ptr<primitive_t> dst_reorder_p_;
};

struct ncsp_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {

        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd)
            , reduce(this)
            , name_("tbd") {}

        ~pd_t() = default;
        DECLARE_COMMON_PD_T(name_.c_str(), ncsp_convolution_bwd_data_t);

        status_t init(engine_t *engine);
        status_t init_convolution(engine_t *engine);
        status_t init_matmul(engine_t *engine);

        std::shared_ptr<primitive_desc_t> matmul_diff_src_pd_;
        std::shared_ptr<primitive_desc_t> nspc_conv_pd_;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;
        memory_desc_t nspc_diff_dst_md_;
        memory_desc_t nspc_diff_src_md_;
        memory_desc_t matmul_src_md_;
        memory_desc_t matmul_wei_md_;
        memory_desc_t matmul_dst_md_;

    private:
        ncsp_matmul_reduction_helper reduce;
        bool is_matmul_ = false;
        std::string name_;
        void init_scratchpad();
        void init_name() {
            std::string suffix = is_matmul_ ? "matmul" : "conv";
            name_ = "ncsp:" + suffix + "->";
            name_.append(is_matmul_ ? matmul_diff_src_pd_->name()
                                    : nspc_conv_pd_->name());
            if (!is_matmul_) {
                name_.append("+src_reorder->");
                name_.append(src_reorder_pd_->name());
                name_.append("+dst_reorder->");
                name_.append(dst_reorder_pd_->name());
            }
        }
    };
    ncsp_convolution_bwd_data_t(const pd_t *cpd) : primitive_t(cpd) {};
    ~ncsp_convolution_bwd_data_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;
    status_t execute_convolution(const exec_ctx_t &ctx) const;
    status_t execute_matmul(const exec_ctx_t &ctx) const;

private:
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> prim, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> matmul_diff_src_p_;
    std::shared_ptr<primitive_t> nspc_conv_p_;
    std::shared_ptr<primitive_t> src_reorder_p_;
    std::shared_ptr<primitive_t> dst_reorder_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
