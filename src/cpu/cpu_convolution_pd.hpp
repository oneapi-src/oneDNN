/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef CPU_CONVOLUTION_PD_HPP
#define CPU_CONVOLUTION_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "convolution_pd.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_convolution_fwd_pd_t: public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

    bool has_padded_dst() const {
        memory_desc_wrapper dst_d(&dst_md_);
        if (!dst_d.is_blocking_desc()) return false;
        return OC() != dst_d.blocking_desc().padding_dims[1];
    }

    bool wants_padded_bias() const {
        if (!with_bias()) return false;
        return has_padded_dst();
    }

    bool wants_zero_pad_dst(bool jit_impl = true) const {
        if (!has_padded_dst()) return false;
        const auto &po = attr()->post_ops_;
        int idx;
        if ((idx = po.find(primitive_kind::eltwise)) == -1) return false;
        return !math::eltwise_fwd_preserves_zero(po.entry_[idx].eltwise.alg,
                jit_impl);
    }

protected:
    memory_format_t src_format() {
        using namespace memory_format;
        return utils::pick(desc()->src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    memory_format_t wei_format() {
        using namespace memory_format;
        return with_groups()
            ? utils::pick(desc()->src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(desc()->src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    status_t set_default_params() {
        using namespace memory_format;
        if (src_md_.format == any)
            CHECK(types::set_default_format(src_md_, src_format()));
        if (dst_md_.format == any)
            CHECK(types::set_default_format(dst_md_, src_md_.format));
        if (weights_md_.format == any)
            CHECK(types::set_default_format(weights_md_, wei_format()));
        if (bias_md_.format == any)
            CHECK(types::set_default_format(bias_md_, x));
        return status::success;
    }
};

struct cpu_convolution_bwd_data_pd_t: public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;

protected:
    memory_format_t src_format() {
        using namespace memory_format;
        return utils::pick(desc_.diff_src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    memory_format_t wei_format() {
        using namespace memory_format;
        return with_groups()
            ? utils::pick(desc_.diff_src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(desc_.diff_src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    status_t set_default_params() {
        using namespace memory_format;
        if (diff_src_md_.format == any)
            CHECK(types::set_default_format(diff_src_md_, src_format()));
        if (diff_dst_md_.format == any)
            CHECK(types::set_default_format(diff_dst_md_, diff_src_md_.format));
        if (weights_md_.format == any)
           CHECK(types::set_default_format(weights_md_, wei_format()));
        if (bias_md_.format == any)
            CHECK(types::set_default_format(bias_md_, x));
        return status::success;
    }
};

struct cpu_convolution_bwd_weights_pd_t: public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;

    bool wants_padded_bias() const {
        if (!with_bias()) return false;
        memory_desc_wrapper diff_dst_d(&diff_dst_md_);
        if (!diff_dst_d.is_blocking_desc()) return false;
        return OC() != diff_dst_d.blocking_desc().padding_dims[1];
    }

protected:
    memory_format_t src_format() {
        using namespace memory_format;
        return utils::pick(desc_.src_desc.ndims - 3, ncw, nchw, ncdhw);
    }
    memory_format_t wei_format() {
        using namespace memory_format;
        return with_groups()
            ? utils::pick(desc_.src_desc.ndims - 3, goiw, goihw, goidhw)
            : utils::pick(desc_.src_desc.ndims - 3, oiw, oihw, oidhw);
    }

    status_t set_default_params() {
        using namespace memory_format;
        if (src_md_.format == any)
            CHECK(types::set_default_format(src_md_, src_format()));
        if (diff_dst_md_.format == any)
            CHECK(types::set_default_format(diff_dst_md_, src_format()));
        if (diff_weights_md_.format == any)
            CHECK(types::set_default_format(diff_weights_md_, wei_format()));
        if (diff_bias_md_.format == any)
            CHECK(types::set_default_format(diff_bias_md_, x));
        return status::success;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
