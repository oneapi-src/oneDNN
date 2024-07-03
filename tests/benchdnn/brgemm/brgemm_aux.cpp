/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include <sstream>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oneapi/dnnl/dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "brgemm/brgemm.hpp"

namespace brgemm {

dnnl_data_type_t prb_t::get_dt(data_kind_t data_kind) const {
    switch (data_kind) {
        case SRC: return src_dt();
        case WEI: return wei_dt();
        case BIA: return bia_dt;
        case DST: return dst_dt();
        case ACC: return acc_dt();
        default: assert(!"unexpected"); return dnnl_data_type_undef;
    }
}

void prb_t::check_block_size() const {
    // Note: batch_size is incorporated into K dimension.
    // That's why each source batch has an offset of `k`.
    // Weights have more complicated case. Weights are in double-blocked format,
    // which becomes triple-blocked for bf16 and int8 to become VNNI-friendly.
    // Because of this and batch_size incorporation, offsets below DO NOT work
    // with K not divisible by K block size and batch_size > 1.
    // The problem is it can't be handled properly when batch size is fused,
    // but this allows enable s8s8 and zero-points compensation cases easier.
    int block_size = 0;
    switch (wei_dt()) {
        case dnnl_f32: block_size = 16; break;
        case dnnl_f16: block_size = 16; break;
        case dnnl_bf16: block_size = 32; break;
        case dnnl_u8:
        case dnnl_f8_e5m2:
        case dnnl_f8_e4m3:
        case dnnl_s8: block_size = 64; break;
        default: break;
    }
    (void)block_size;
    assert(block_size > 1);
    assert(IMPLICATION(batch_size > 1, k % block_size == 0));
}

std::string prb_t::set_repro_line() {
    std::stringstream s;
    dump_global_params(s);
    settings_t def;

    bool has_default_dts = true;
    for (const auto &i_dt : dt)
        has_default_dts = has_default_dts && i_dt == dnnl_f32;

    if (canonical || !has_default_dts) s << "--dt=" << dt << " ";
    if (canonical || stag != def.stag[0]) s << "--stag=" << stag << " ";
    if (canonical || wtag != def.wtag[0]) s << "--wtag=" << wtag << " ";
    if (canonical || dtag != def.dtag[0]) s << "--dtag=" << dtag << " ";
    if (canonical || ld != def.ld[0]) {
        s << "--ld=";
        if (ld[0] != 0) s << ld[0];
        s << ":";
        if (ld[1] != 0) s << ld[1];
        s << ":";
        if (ld[2] != 0) s << ld[2];
        s << " ";
    }

    if (canonical || bia_dt != def.bia_dt[0]) s << "--bia_dt=" << bia_dt << " ";

    if (canonical || alpha != def.alpha[0]) s << "--alpha=" << alpha << " ";
    if (canonical || beta != def.beta[0]) s << "--beta=" << beta << " ";
    if (canonical || batch_size != def.batch_size[0])
        s << "--bs=" << batch_size << " ";
    if (canonical || brgemm_attr != def.brgemm_attr[0])
        s << "--brgemm-attr=" << brgemm_attr << " ";
    if (canonical || batch_kind != def.batch_kind[0])
        s << "--batch-kind=" << batch_kind << " ";

    s << attr;
    s << static_cast<const prb_vdims_t &>(*this);

    return s.str();
}

} // namespace brgemm
