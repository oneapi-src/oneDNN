/*******************************************************************************
 * * Copyright 2019-2021 Intel Corporation
 * *
 * * Licensed under the Apache License, Version 2.0 (the "License");
 * * you may not use this file except in compliance with the License.
 * * You may obtain a copy of the License at
 * *
 * *     http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing, software
 * * distributed under the License is distributed on an "AS IS" BASIS,
 * * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * * See the License for the specific language governing permissions and
 * * limitations under the License.
 * *******************************************************************************/

#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_graph_buildin_ops.h"

#include "tests/test_thread.hpp"

#include "bnorm/bnorm.hpp"
#include "bnorm/graph_bnorm.hpp"
#include "dnnl_common.hpp"
#include "dnnl_graph_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

namespace benchdnnext {
namespace bnorm {

bnorm_graph_prb_t::spec_t::spec_t(const ::bnorm::prb_t *prb) {
    dims_t dims_0d = {prb->mb, prb->ic};
    dims_t dims_1d = {prb->mb, prb->ic, prb->iw};
    dims_t dims_2d = {prb->mb, prb->ic, prb->ih, prb->iw};
    dims_t dims_3d = {prb->mb, prb->ic, prb->id, prb->ih, prb->iw};
    dims = [&](int n) {
        switch (n) {
            case 5: return dims_3d;
            case 4: return dims_2d;
            case 3: return dims_1d;
            default: return dims_0d;
        }
    }(prb->ndims);

    s_dims = {prb->ic};
    bnorm_dt = convert_dt(prb->dt);
    epsilon = prb->eps;
    tag = prb->tag;
}

fill_status_t bnorm_graph_prb_t::handle_main_op_() {
    using op = dnnl::graph::op;

    const std::string SRC {"bnorm_src"};
    const std::string SCALE {"bnorm_scale"};
    const std::string SHIFT {"bnorm_shift"};
    const std::string MEAN {"bnorm_mean"};
    const std::string VAR {"bnorm_variance"};
    const std::string DST {"bnorm_dst"};

    tensor_descs_.emplace(SRC, spec_.bnorm_dt, spec_.dims, lt::strided);
    tensor_descs_.emplace(SCALE, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(SHIFT, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(MEAN, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(VAR, spec_.bnorm_dt, spec_.s_dims, lt::strided);
    tensor_descs_.emplace(DST, spec_.bnorm_dt, spec_.dims, lt::strided);

    const size_t new_op_id = ops_.size();
    op bnorm_op(new_op_id, dnnl::graph::op::kind::BatchNormInference,
            {tensor_descs_[SRC], tensor_descs_[SCALE], tensor_descs_[SHIFT],
                    tensor_descs_[MEAN], tensor_descs_[VAR]},
            {tensor_descs_[DST]}, "bnorm");

    bnorm_op.set_attr("epsilon", spec_.epsilon);
    bnorm_op.set_attr<std::string>("data_format", convert_tag(spec_.tag));

    ops_.emplace_back(bnorm_op);
    curr_out_map_ids_.assign({DST});

    return fill_status::DONE;
}

fill_status_t bnorm_graph_prb_t::handle_elt_(
        const attr_t::post_ops_t::entry_t &po_entry) {
    return po_handler.bnorm.eltw_handler(*this, po_entry);
}

static int prepare_fwd_with_stats(const ::bnorm::prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &scale, dnn_mem_t &shift,
        dnn_mem_t &ss) {
    dnnl::impl::parallel_nd(prb->ic, prb->mb, prb->id, prb->ih, prb->iw,
            [&](int64_t c, int64_t mb, int64_t d, int64_t h, int64_t w) {
                int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
                float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;
                const int64_t value = (l % 65) - 32;
                s[sp] = round_to_nearest_representable(prb->dt, value);

                ((float *)mean)[c] = 4 * ((c % 5) - 2);
                ((float *)var)[c] = ((c % 7) << 1);

                if (prb->flags & ::bnorm::USE_SCALESHIFT) {
                    ((float *)ss)[c] = (1 << (c % 7));
                    ((float *)scale)[c] = (1 << (c % 7));
                    ((float *)ss)[prb->ic + c]
                            = ((c % 3) - 1) * ((float *)ss)[c];
                    ((float *)shift)[c] = ((c % 3) - 1) * ((float *)scale)[c];
                } else {
                    ((float *)ss)[c] = 1;
                    ((float *)scale)[c] = 1;
                    ((float *)ss)[prb->ic + c] = 0;
                    ((float *)shift)[c] = 0;
                }
            });

    return OK;
}

static int prepare_fwd_no_stats(const ::bnorm::prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &scale, dnn_mem_t &shift,
        dnn_mem_t &ss) {
    /** Idea: choose src[] values so that both mean and variance are computed
     * exactly (independently of the order of the computations).
     *
     * The `exactness` is achieved via [a1]: src[i] + src[i+1] = 2 * mean.
     *
     * The variation in src is allowed in the last flex_bits bits.
     * If the sequence (L) is too big (flex_bits <= min_flex_bits), the mean
     * value is set to 0 and src is partially filled with zeros (according to
     * density so that at least want_flex_bits is reserved for src variation.
     * Once src is set, variance is computed.
     *
     * ALG_0: mean is set to 0
     * ALG_1: mean is set to 2^prb, where prb \in {-2, -1, ..., 4}
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically */
    const int64_t exact_bits = digits_dt(prb->dt);
    const int64_t L = prb->mb * prb->id * prb->ih * prb->iw;
    const int64_t logL = (int64_t)ceilf(log2f(L));

    assert(logL <= 0 || (1LL << (logL - 1)) < L);
    assert(L <= (1LL << logL));

    const int64_t min_flex_bits = 3;
    const int64_t want_flex_bits = MIN2(6, exact_bits / 2);

    ::bnorm::check_alg_t alg = prb->check_alg;
    if (alg == ::bnorm::ALG_AUTO) /* choose appropriate checking algorithm */
        alg = (exact_bits - logL) / 2 - 1 >= min_flex_bits ? ::bnorm::ALG_1
                                                           : ::bnorm::ALG_0;

    const int64_t flex_bits = alg == ::bnorm::ALG_0
            ? want_flex_bits /* BFloat16 has only 7 bits of mantissa */
            : MIN2(prb->dt == dnnl_bf16 ? 7 : exact_bits,
                    (exact_bits - logL) / 2 - 1);

    if (flex_bits < min_flex_bits) return FAIL;

    const int64_t flex_mask = (1 << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ::bnorm::ALG_0
            ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
            : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    BENCHDNN_PRINT(6, "check_alg: %s, density = %g, flex_bits = " IFMT "\n",
            check_alg2str(alg), density, flex_bits);

    dnnl::impl::parallel_nd(prb->ic, [&](int64_t c) {
        const float m = ((float *)mean)[c]
                = alg == ::bnorm::ALG_0 ? 0.f : 0.25f * (1 << (c % 7));
        float v = 0; /* current variance */

        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            int64_t l_base = mb * prb->id * prb->ih * prb->iw
                    + c * 239 * 2; // l[0] must be even
            float *s = (float *)src + data_off(prb, mb, c, 0, 0, 0);

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {

                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;

                if (alg == ::bnorm::ALG_0
                        && !flip_coin(l / 2 * 257ULL, density)) {
                    s[sp] = 0;
                    continue;
                }

                const int64_t gen = (l / 2 * 1637) & flex_mask;
                const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
                const float f = 1.f * sgn * gen / (1 << flex_bits);

                s[sp] = alg == ::bnorm::ALG_0 ? f : m * (1.f + f);
                if (L % 2 && (mb * prb->id * prb->ih * prb->iw + sp == L - 1)) {
                    s[sp] = m;
                }
                v += (s[sp] - m) * (s[sp] - m);
            }
        }

        ((float *)var)[c] = v / (prb->mb * prb->id * prb->ih * prb->iw);

        if (prb->flags & ::bnorm::USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 8 * (1 << (c % 7));
            ((float *)scale)[c] = 1.f / 8 * (1 << (c % 7));
            ((float *)ss)[prb->ic + c] = ((c % 3) - 1) * ((float *)ss)[c] / 64;
            ((float *)shift)[c] = ((c % 3) - 1) * ((float *)scale)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)scale)[c] = 1;
            ((float *)ss)[prb->ic + c] = 0;
            ((float *)shift)[c] = 0;
        }
    });

    return OK;
}

static int prepare_fwd(const ::bnorm::prb_t *prb, dnn_mem_t &src,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &scale, dnn_mem_t &shift,
        dnn_mem_t &ss) {
    if (prb->flags & ::bnorm::GLOB_STATS)
        return prepare_fwd_with_stats(prb, src, mean, var, scale, shift, ss);
    else
        return prepare_fwd_no_stats(prb, src, mean, var, scale, shift, ss);
}

void check_known_skipped_case(const ::bnorm::prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, prb->dir, res);
    if (res->state == SKIPPED) return;

    for (const auto &po : prb->attr.post_ops.entry) {
        if (po.kind == attr_t::post_ops_t::RELU) {
            continue;
        } else {
            res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
            return;
        }
    }
}

int doit(const ::bnorm::prb_t *prb, res_t *res) {
    using dt = dnnl::graph::logical_tensor::data_type;
    res->impl_name = "graph";

    if (bench_mode == LIST) return res->state = LISTED, OK;
    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    bnorm_graph_prb_t graph_prb(prb);
    if (graph_prb.ctor_status != fill_status::DONE
            && graph_prb.ctor_status != fill_status::UNHANDLED_CONFIG_OPTIONS) {
        return res->state = UNIMPLEMENTED, FAIL;
    }

    auto graph_h = graph_prb.to_graph();
    const auto partitions = graph_h.get_partitions();
    if (partitions.empty() || partitions.size() > 1)
        return res->state = FAILED, FAIL;

    const auto par = partitions[0];
    if (!par.is_supported()) return res->state = UNIMPLEMENTED, FAIL;

    const auto ins = par.get_in_ports();
    const auto outs = par.get_out_ports();

    const auto &e = benchdnnext::get_test_engine();
    auto cp = par.compile(ins, outs, e);

    dnnl_dim_t dims_ss[] = {2, prb->ic};
    dnnl_dim_t data_dims[] = {prb->mb, prb->ic, prb->ih, prb->iw};

    static const engine_t cpu_engine(dnnl_cpu);

    dnn_mem_t src_fp = make_dnn_mem(ins[0], dt::f32, tag::abx);
    dnn_mem_t scale_fp = make_dnn_mem(ins[1], dt::f32, tag::abx);
    dnn_mem_t shift_fp = make_dnn_mem(ins[2], dt::f32, tag::abx);
    dnn_mem_t mean_fp = make_dnn_mem(ins[3], dt::f32, tag::abx);
    dnn_mem_t var_fp = make_dnn_mem(ins[4], dt::f32, tag::abx);
    dnn_mem_t ss_fp(2, dims_ss, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t src_hat_fp(4, data_dims, dnnl_f32, tag::abx, cpu_engine);
    dnn_mem_t ws_fp(4, data_dims, dnnl_u8, tag::abx, cpu_engine);

    dnn_mem_t dest_dt = make_dnn_mem(outs[0], tag::abx);
    dnn_mem_t src_dt = make_dnn_mem(ins[0], tag::abx);
    dnn_mem_t scale_dt = make_dnn_mem(ins[1], tag::abx);
    dnn_mem_t shift_dt = make_dnn_mem(ins[2], tag::abx);
    dnn_mem_t mean_dt = make_dnn_mem(ins[3], tag::abx);
    dnn_mem_t var_dt = make_dnn_mem(ins[4], tag::abx);
    dnn_mem_t &dst_dt = prb->inplace ? src_dt : dest_dt;

    if (prepare_fwd(prb, src_fp, mean_fp, var_fp, scale_fp, shift_fp, ss_fp)
            != OK) {
        return res->state = MISTRUSTED, OK;
    }
    SAFE(src_dt.reorder(src_fp), WARN);
    SAFE(scale_dt.reorder(scale_fp), WARN);
    SAFE(shift_dt.reorder(shift_fp), WARN);
    SAFE(mean_dt.reorder(mean_fp), WARN);
    SAFE(var_dt.reorder(var_fp), WARN);

    std::vector<dnnl::graph::tensor> tensors_in, tensors_out;
    tensors_in.emplace_back(ins[0], static_cast<void *>(src_dt));
    tensors_in.emplace_back(ins[1], static_cast<void *>(scale_dt));
    tensors_in.emplace_back(ins[2], static_cast<void *>(shift_dt));
    tensors_in.emplace_back(ins[3], static_cast<void *>(mean_dt));
    tensors_in.emplace_back(ins[4], static_cast<void *>(var_dt));
    tensors_out.emplace_back(outs[0], static_cast<void *>(dst_dt));
    SAFE(execute_and_wait(cp, tensors_in, tensors_out), WARN);

    if (bench_mode & CORR) {
        static const engine_t cpu_engine(dnnl_cpu);
        ::bnorm::compute_ref_fwd(
                prb, src_fp, mean_fp, var_fp, ss_fp, ws_fp, dst_fp, src_hat_fp);
        if (prb->dir & FLAG_FWD) {
            if (!(prb->flags & ::bnorm::GLOB_STATS) && !(prb->dir & FLAG_INF)) {
                SAFE(::bnorm::compare(prb, MEAN, mean_fp, mean_dt, res), WARN);
                SAFE(::bnorm::compare(prb, VAR, var_fp, var_dt, res), WARN);
            }
            dnn_mem_t dst(dst_dt, dnnl_f32, tag::abx, cpu_engine);
            SAFE(::bnorm::compare(prb, DATA, dst_fp, dst, res, &ss_fp), WARN);
        }
    }
    return measure_perf(res->timer, cp, tensors_in, tensors_out);
}

} // namespace bnorm
} // namespace benchdnnext
