/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_batch_normalization.hpp"
#include "type_helpers.hpp"

#include <cmath>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

template <impl::precision_t prec>
jit_avx2_batch_normalization<prec>::jit_avx2_batch_normalization(
        const batch_normalization_primitive_desc_t &bnpd,
        const primitive_at_t *inputs,
        const primitive *outputs[])
    : batch_normalization<
        jit_avx2_batch_normalization<prec>>(bnpd, inputs, outputs)
    , generator(new jit_avx2_batch_norm_generator_f32(bnpd, this->_is_training))
    {}

template <impl::precision_t prec>
status_t jit_avx2_batch_normalization<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    auto scaleshift = reinterpret_cast<data_t *>(
            this->input()[1].primitive->output()[
            this->input()[1].output_index]->memory());
    auto workspace = this->_is_training
        ? reinterpret_cast<data_t *>(
            this->input()[2].primitive->output()[
            this->input()[2].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t*>(this->output()[0]->memory());

    const memory_desc_wrapper
        src_d(this->_bnpd.src_primitive_desc),
        scaleshift_d(this->_bnpd.scaleshift_primitive_desc),
        dst_d(this->_bnpd.dst_primitive_desc),
        workspace_d(this->_bnpd.workspace_primitive_desc);

    const auto &jbnp = this->generator->jbnp;

    auto ker = [&](int c) {
        jit_batch_normalization_kernel_t par_bn = {};

        par_bn.src = &src[src_d.blk_off(0, c, 0, 0)];
        par_bn.dst = &dst[dst_d.blk_off(0, c, 0, 0)];
        par_bn.scaleshift = &scaleshift[scaleshift_d.off(0, jbnp.c_block*c)];
        par_bn.workspace = &workspace[jbnp.c_block*c];

        this->generator->jit_ker((void*)&par_bn);
    };

#   pragma omp parallel for schedule(static)
    for (int c = 0; c < jbnp.nb_c; ++c) {
        ker(c);
    }
    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_batch_normalization<prec>::set_default_parameters(
        batch_normalization_desc_t &batch_norm_d) {
    if (batch_norm_d.src_desc.format == any) {
        CHECK(types::set_default_format<prec>(batch_norm_d.src_desc, nChw8c));
    }
    if (batch_norm_d.scaleshift_desc.format == any) {
        CHECK(types::set_default_format<prec>(batch_norm_d.scaleshift_desc,
                    nc));
    }
    if (batch_norm_d.dst_desc.format == any) {
        CHECK(types::set_default_format<prec>(batch_norm_d.dst_desc, nChw8c));
    }
    return batch_normalization<jit_avx2_batch_normalization<prec>>::template
        set_default_parameters<void>(batch_norm_d);
}

template <impl::precision_t prec>
status_t jit_avx2_batch_normalization<prec>::constraint(
        const batch_normalization_desc_t &batch_norm_d) {
    bool args_ok = true
        && one_of(batch_norm_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && jit_avx2_batch_norm_generator_f32::is_applicable(batch_norm_d);
    return args_ok ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_batch_normalization<prec>::implementation = {
    jit_avx2_batch_normalization<prec>::create
};

template class jit_avx2_batch_normalization<precision::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
