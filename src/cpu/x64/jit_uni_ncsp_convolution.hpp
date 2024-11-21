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

#ifndef CPU_X64_JIT_UNI_NCSP_CONVOLUTION_HPP
#define CPU_X64_JIT_UNI_NCSP_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct reduction_helper_t {
    reduction_helper_t() : pd_(nullptr) {}
    reduction_helper_t(const convolution_pd_t *pd_) : pd_(pd_) {}
    status_t reshape_activations(
            memory_desc_t *o_md, const memory_desc_t *i_md, bool is_dst);

    status_t reshape_bias(memory_desc_t *o_md, const memory_desc_t *i_md);
    status_t reshape_weights(
            memory_desc_t *o_md, const memory_desc_t *i_md, bool to_matmul);
    status_t reshape_for_transpose(memory_desc_t &o_md, memory_desc_t &i_md);
    // If convolution is 1x1, no padding, and single strides then dispatch
    // to matmul kernel. This is done because matmul supports transposed
    // layout and is more efficient than 1x1 convolutions due to not having
    // to dispatch an additional reorder kernel for src and dst. Adding
    // transposed support inside convolution brgemm kernels is preferable but it
    // will take time to implement.
    bool is_gemm();

private:
    const convolution_pd_t *pd_;
};

struct jit_uni_ncsp_convolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(name_.c_str(), jit_uni_ncsp_convolution_fwd_t);

        status_t init(engine_t *engine);

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

    private:
        status_t init_convolution(engine_t *engine);
        status_t init_matmul(engine_t *engine);
        reduction_helper_t reduction_helper_;
        bool is_matmul_ = false;
        std::string name_ = "jit_uni_ncsp_convolution:";
        void init_name() {
            std::string suffix = is_matmul_ ? "matmul" : "conv";
            name_ += suffix + "+";
            name_.append(
                    is_matmul_ ? matmul_pd_->name() : nspc_conv_pd_->name());
        }
        void init_scratchpad();
    };

    jit_uni_ncsp_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {};

    ~jit_uni_ncsp_convolution_fwd_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_convolution(const exec_ctx_t &ctx) const;
    status_t execute_matmul(const exec_ctx_t &ctx) const;
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> &prim, engine_t *engine,
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

struct jit_uni_ncsp_convolution_bwd_weights_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        using cpu_convolution_bwd_weights_pd_t::
                cpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(
                name_.c_str(), jit_uni_ncsp_convolution_bwd_weights_t);

        status_t init(engine_t *engine);

        std::shared_ptr<primitive_desc_t> nspc_conv_pd_;
        std::shared_ptr<primitive_desc_t> src_reorder_pd_;
        std::shared_ptr<primitive_desc_t> dst_reorder_pd_;
        memory_desc_t nspc_src_md_;
        memory_desc_t nspc_diff_dst_md_;

    private:
        status_t init_convolution(engine_t *engine);
        std::string name_;
        void init_scratchpad();
        void init_name() {
            name_ = "jit_uni_ncsp_convolution:conv+";
            name_.append(nspc_conv_pd_->name());
        }
    };
    jit_uni_ncsp_convolution_bwd_weights_t(const pd_t *cpd)
        : primitive_t(cpd) {};
    ~jit_uni_ncsp_convolution_bwd_weights_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_convolution(const exec_ctx_t &ctx) const;
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> &prim, engine_t *engine,
            const memory_arg_t &in, const memory_arg_t &out) const;
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }
    std::shared_ptr<primitive_t> nspc_conv_p_;
    std::shared_ptr<primitive_t> src_reorder_p_;
    std::shared_ptr<primitive_t> dst_reorder_p_;
};

struct jit_uni_ncsp_convolution_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T(name_.c_str(), jit_uni_ncsp_convolution_bwd_data_t);

        status_t init(engine_t *engine);

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
        status_t init_convolution(engine_t *engine);
        status_t init_matmul(engine_t *engine);
        reduction_helper_t reduction_helper_;
        bool is_matmul_ = false;
        std::string name_;
        void init_scratchpad();
        void init_name() {
            std::string suffix = is_matmul_ ? "matmul" : "conv";
            name_ = "jit_uni_ncsp_convolution:" + suffix + "+";
            name_.append(is_matmul_ ? matmul_diff_src_pd_->name()
                                    : nspc_conv_pd_->name());
        }
    };
    jit_uni_ncsp_convolution_bwd_data_t(const pd_t *cpd) : primitive_t(cpd) {};
    ~jit_uni_ncsp_convolution_bwd_data_t() = default;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    status_t execute_convolution(const exec_ctx_t &ctx) const;
    status_t execute_matmul(const exec_ctx_t &ctx) const;
    status_t reorder_activations(const exec_ctx_t &ctx,
            const std::shared_ptr<primitive_t> &prim, engine_t *engine,
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
