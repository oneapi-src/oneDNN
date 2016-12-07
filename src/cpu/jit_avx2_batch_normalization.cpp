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

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

void jit_avx2_batch_normalization_fwd_t::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    /* FIXME: check this */
   data_t* mean = conf_.stats_is_src() ?
       const_cast<data_t*>(reinterpret_cast<const data_t*>(
               this->input_memory(1))) :
       reinterpret_cast<data_t*>(this->memory(1));

   data_t* variance = conf_.stats_is_src() ?
       const_cast<data_t*>(reinterpret_cast<const data_t*>(
               this->input_memory(2))) :
       reinterpret_cast<data_t*>(this->memory(2));

    auto idx_scaleshift = 1 + 2*conf_.stats_is_src();
    auto scaleshift =
        reinterpret_cast<const data_t *>(this->input_memory(idx_scaleshift));

    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper data_d(conf_.src_pd());
    const memory_desc_wrapper mean_d(conf_.mean_pd());
    const memory_desc_wrapper variance_d(conf_.variance_pd());
    const memory_desc_wrapper scaleshift_d(conf_.weights_pd());

    const auto &jbp = kernel_->jbp;
    /* FIXME: check this */
    const int b_c_mult = data_d.format() == memory_format::nChw8c
        ? 1 : data_d.dims()[2] * data_d.dims()[3];

    auto ker = [&](int b_c) {
        jit_bnrm_call_s arg = {};

        const int c = b_c * jbp.c_block;
        const size_t d_off = data_d.blk_off(0, b_c * b_c_mult, 0, 0);
        arg.src = &src[d_off];
        arg.dst = &dst[d_off];
        arg.scaleshift = &scaleshift[scaleshift_d.off(0, c)];
        if (conf_.stats_is_src() || conf_.is_training()) {
            arg.mean = &mean[mean_d.off(c)];
            arg.variance = &variance[variance_d.off(c)];
        }
        (*kernel_)(&arg);
    };

#   pragma omp parallel for schedule(static)
    for (int b_c = 0; b_c < jbp.nb_c; ++b_c) {
        ker(b_c);
    }
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
