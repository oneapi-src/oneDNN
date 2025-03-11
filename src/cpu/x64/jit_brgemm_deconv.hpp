/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_DECONV_HPP
#define CPU_X64_JIT_BRGEMM_DECONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"

#include "cpu/x64/jit_brgemm_1x1_conv.hpp"
#include "cpu/x64/jit_brgemm_conv.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_strided.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_deconvolution_fwd_t : public primitive_t {

    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , has_strides_(other.has_strides_)
            , name_(other.name_) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), brgemm_deconvolution_fwd_t);

        status_t init(engine_t *engine);

        bool post_ops_ok() const {
            return attr()->post_ops_.find(primitive_kind::convolution) == -1;
        }

        bool zero_points_ok() const {
            const auto &zp = attr()->zero_points_;

            using namespace data_type;
            bool ok = IMPLICATION(!utils::one_of(src_md()->data_type, s8, u8),
                    zp.has_default_values());
            if (!ok) return false;

            if (!zp.has_default_values(DNNL_ARG_SRC)) {
                int mask_src = zp.get_mask(DNNL_ARG_SRC);
                ok = utils::one_of(mask_src, 0, (1 << 1));
                if (!ok) return false;
            }
            if (!zp.has_default_values(DNNL_ARG_DST)) {
                int mask_dst = zp.get_mask(DNNL_ARG_DST);
                ok = utils::one_of(mask_dst, 0, (1 << 1));
                if (!ok) return false;
            }

            return zp.has_default_values(DNNL_ARG_WEIGHTS);
        }

        brgemm_broadcast_t get_zp_type(int arg) const {
            return attr()->zero_points_.has_default_values(arg)
                    ? brgemm_broadcast_t::none
                    : brgemm_broadcast_t::per_tensor;
        }

        std::shared_ptr<primitive_desc_t> conv_pd_;
        bool has_strides_ = false;

    private:
        std::string name_;

        void init_name() {
            name_ = JIT_IMPL_NAME_HELPER("brg_deconv:", isa, "");
            name_.append("+");
            name_.append(conv_pd_->name());
        }
    };

    brgemm_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {};

    ~brgemm_deconvolution_fwd_t() = default;

    status_t init(engine_t *engine) override;

    status_t execute(exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::shared_ptr<primitive_t> conv_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
