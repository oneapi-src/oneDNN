/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);
    settings_t def;

    if (canonical || p.dir != def.dir[0]) s << "--dir=" << p.dir << " ";
    if (canonical || p.dt != def.dt[0]) s << "--dt=" << p.dt << " ";
    if (canonical || p.tag != def.tag[0]) s << "--tag=" << p.tag << " ";
    s << "--alg=" << p.alg << " ";
    s << "--alpha=" << p.alpha << " ";
    s << "--beta=" << p.beta << " ";
    if (canonical || p.inplace != def.inplace[0])
        s << "--inplace=" << bool2str(p.inplace) << " ";

    s << p.dims;

    return s;
}

bool prb_t::maybe_skip_nvidia() const {
    bool fwd_ok = IMPLICATION(this->dir % FLAG_FWD,
            (this->dt == dnnl_s8 || this->dt == dnnl_f16
                    || this->dt == dnnl_f32)
                    && (this->alg == attr_t::post_ops_t::RELU
                            || this->alg == attr_t::post_ops_t::BRELU
                            || this->alg == attr_t::post_ops_t::TANH
                            || this->alg == attr_t::post_ops_t::ELU
                            || this->alg == attr_t::post_ops_t::LOGISTIC));
    bool bwd_ok = IMPLICATION(this->dir == BWD_D,
            this->dt == dnnl_f32
                    && (this->alg == attr_t::post_ops_t::RELU
                            || this->alg == attr_t::post_ops_t::BRELU));
    bool relu_ok
            = IMPLICATION(this->alg == eltwise::alg_t::RELU, this->alpha == 0);
    auto tag_ok = cudnn_supported_tag_plain(convert_tag(this->tag, ndims));
    return !tag_ok || !fwd_ok || !bwd_ok || !relu_ok;
}

} // namespace eltwise
