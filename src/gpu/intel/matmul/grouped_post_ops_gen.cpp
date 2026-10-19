/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/matmul/grouped_post_ops_gen.hpp"
#include "common/matmul_pd.hpp"
#include "common/verbose.hpp"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include <iomanip>
#include <string>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

bool is_eltwise_alg_supported(alg_kind_t alg) {
    using namespace alg_kind;
    return utils::one_of(alg, eltwise_relu, eltwise_linear, eltwise_soft_relu,
            eltwise_mish, eltwise_logistic, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_sqrt, eltwise_abs, eltwise_exp,
            eltwise_gelu_tanh, eltwise_swish, eltwise_log, eltwise_clip,
            eltwise_clip_v2, eltwise_pow, eltwise_gelu_erf, eltwise_round,
            eltwise_hardswish, eltwise_hardsigmoid);
}

int find_po_in_chain(const po_kind_t *po_chain, po_kind_t kind) {
    for (int i = 0; i < 3; ++i) {
        if (po_chain[i] == kind) return i;
    }
    return -1;
}

// An unspecified scale data type is read as f32
static bool is_scale_dt_supported(const memory_desc_wrapper &mdw) {
    return utils::one_of(mdw.data_type(), data_type::undef, data_type::f16,
            data_type::bf16, data_type::f32);
}

status_t check_post_op_chain(const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_desc, dim_t ngroups,
        po_kind_t *po_chain) {
    constexpr int max_po_len = 3;
    auto &po = attr.post_ops_;
    VDISPATCH_MATMUL_IC(po.len() <= max_po_len,
            VERBOSE_UNSUPPORTED_POSTOP ": chain length(%d) exceeds max(%d)",
            po.len(), max_po_len);
    for (int i = 0; i < po.len(); ++i) {
        auto &e = po.entry_[i];
        VDISPATCH_MATMUL_IC(e.is_eltwise() || e.is_binary(),
                VERBOSE_UNSUPPORTED_POSTOP
                ": entry %d kind(%s) only eltwise and binary are supported",
                i, dnnl_prim_kind2str(e.kind));
        if (e.is_eltwise()) {
            VDISPATCH_MATMUL_IC(is_eltwise_alg_supported(e.eltwise.alg),
                    VERBOSE_UNSUPPORTED_POSTOP ": entry %d eltwise alg(%s)", i,
                    dnnl_alg_kind2str(e.eltwise.alg));
            po_chain[i] = po_kind_t::eltwise;
        } else if (e.is_binary()) {
            VDISPATCH_MATMUL_IC(e.binary.alg == alg_kind::binary_mul,
                    VERBOSE_UNSUPPORTED_POSTOP
                    ": entry %d binary alg(%s) only mul is supported",
                    i, dnnl_alg_kind2str(e.binary.alg));

            const memory_desc_wrapper po_mdw(po.entry_[i].binary.src1_desc);
            if (!po_mdw.is_grouped_desc() && po_mdw.nelems() == ngroups
                    && !po_mdw.is_host_scalar_desc()) {
                // [G, 1] operand: one scale per group (expert), e.g. nvfp4
                // per-expert global f32 scale
                VDISPATCH_MATMUL_IC(po_mdw.data_type() == data_type::f32,
                        VERBOSE_UNSUPPORTED_POSTOP
                        ": entry %d per-group scale dt(%s) must be f32",
                        i, dnnl_dt2str(po_mdw.data_type()));
                po_chain[i] = po_kind_t::binary_nvfp4_scale;
            } else {
                if (po_mdw.is_grouped_desc()
                        && po_mdw.dims()[1] == dst_desc.dims()[1]) {
                    // [total_tokens, N] grouped - element-wise multiply
                    VDISPATCH_MATMUL_IC(find_po_in_chain(po_chain,
                                                po_kind_t::binary_grouped_scale)
                                    == -1,
                            VERBOSE_UNSUPPORTED_POSTOP
                            ": entry %d is a second grouped binary scale "
                            "only one is supported",
                            i);
                    po_chain[i] = po_kind_t::binary_grouped_scale;
                    VDISPATCH_MATMUL_IC(is_scale_dt_supported(po_mdw),
                            VERBOSE_UNSUPPORTED_POSTOP
                            ": entry %d grouped binary scale dt(%s), expected "
                            "f16, bf16 or f32",
                            i, dnnl_dt2str(po_mdw.data_type()));
                } else if (!po_mdw.format_any()) {
                    // Dense tensor - check dimensions
                    const auto &dims = po_mdw.dims();
                    const auto ndims = po_mdw.ndims();
                    // [total_tokens, 1] - dense scale with horizontal broadcast
                    if (ndims >= 2 && dims[ndims - 1] == 1
                            && dims[ndims - 2] > 1) {
                        VDISPATCH_MATMUL_IC(
                                find_po_in_chain(
                                        po_chain, po_kind_t::binary_dense_scale)
                                        == -1,
                                VERBOSE_UNSUPPORTED_POSTOP
                                ": entry %d is a second dense binary scale "
                                "only one is supported",
                                i);
                        po_chain[i] = po_kind_t::binary_dense_scale;
                        VDISPATCH_MATMUL_IC(is_scale_dt_supported(po_mdw),
                                VERBOSE_UNSUPPORTED_POSTOP
                                ": entry %d dense binary scale dt(%s), "
                                "expected f16, bf16 or f32",
                                i, dnnl_dt2str(po_mdw.data_type()));
                    }
                }
            }
            VDISPATCH_MATMUL_IC(po_chain[i] != po_kind_t::none,
                    VERBOSE_UNSUPPORTED_POSTOP
                    ": entry %d binary src1 dims(%s) tag(%s) is not a "
                    "supported scale layout; expected per-group [%d,1], "
                    "grouped [total_tokens,N] or dense [total_tokens,1]",
                    i, md2dim_str(&e.binary.src1_desc).c_str(),
                    md2fmt_tag_str(&e.binary.src1_desc).c_str(), (int)ngroups);
        }
    }

    return status::success;
}

data_type_t binary_scale_dt(const gpu_post_ops_t &post_ops,
        const po_kind_t *po_chain, po_kind_t kind) {
    const int i = find_po_in_chain(po_chain, kind);
    if (i == -1) return data_type::undef;
    const data_type_t dt = post_ops[i].as_binary().src1_desc.dt;
    return dt == data_type::undef ? data_type::f32 : dt;
}

static std::string to_ocl_float(float v) {
    std::ostringstream os;
    os << std::setprecision(9) << std::scientific << v << "f";
    return os.str();
}

std::string generate_post_ops_microgemm_header(
        const gpu_post_ops_t &post_ops, const po_kind_t *po_chain) {
    std::string s = R"(
inline void apply_post_ops_chain(c_tile_type_float *c_tile, long n, long m, long lddst,
    off_t sg_i0, off_t sg_j0, off_t src_offset, off_t batch,
    const global binary_scale_grouped_tile_data_t *grouped_scale,
    const global binary_scale_dense_tile_data_t *dense_scale,
    const global float *nvfp4_scale) {
)";
    for (int i = 0; i < into<int>(post_ops.len()); ++i) {
        const auto &entry = post_ops[i];
        if (entry.is_eltwise()) {
            const auto &e = entry.as_eltwise();
            s += utils::format(R"(
#define eltwise_apply_%d(v) fwd_eltwise_common(%d, v, %s, %s, %s)
    tile_elementwise((*c_tile), eltwise_apply_%d);
#undef eltwise_apply_%d
                     )",
                    i, static_cast<int>(e.alg), to_ocl_float(e.alpha).c_str(),
                    to_ocl_float(e.beta).c_str(), to_ocl_float(e.scale).c_str(),
                    i, i);
        } else if (entry.is_binary()) {
            const po_kind_t kind = po_chain[i];
            if (kind == po_kind_t::binary_grouped_scale) {
                s += R"(
    {
        const global binary_scale_grouped_tile_data_t *group_scale_ptr
                = grouped_scale + src_offset * lddst;
#define GRP_BC ugemm_grouped_c_type_block1
#define GRP_NBR ugemm_grouped_c_type_nblock0
#define GRP_NBC ugemm_grouped_c_type_nblock1
#pragma unroll
        for (int gcb = 0; gcb < GRP_NBC; gcb++) {
            binary_group_chunk_type binary_group_chunk;
#if BINARY_SCALE_GROUPED_DT_F32
            tile_load(&binary_group_chunk, group_scale_ptr, n, m, lddst, sg_i0,
                    sg_j0 + gcb * GRP_BC);
#else
            binary_group_chunk_in_type binary_group_chunk_in;
            tile_load(&binary_group_chunk_in, group_scale_ptr, n, m, lddst,
                    sg_i0, sg_j0 + gcb * GRP_BC);
            tile_convert(binary_group_chunk_in, binary_group_chunk,
                    BINARY_SCALE_GROUPED_TO_FLOAT);
#endif
#pragma unroll
            for (int gbb = 0; gbb < GRP_NBR; gbb++)
                (*c_tile).x[GRP_NBR * gcb + gbb] *= binary_group_chunk.x[gbb];
        }
#undef GRP_BC
#undef GRP_NBR
#undef GRP_NBC
    }
                         )";
            } else if (kind == po_kind_t::binary_dense_scale) {
                s += utils::format(
                        R"(
#define CONCAT_I(var) var##_%d
    const global binary_scale_dense_tile_data_t *CONCAT_I(dense_scale_ptr) = dense_scale + src_offset;
    binary_dense_tile_type CONCAT_I(dense_scale_tile);
#if BINARY_SCALE_DENSE_DT_F32
    tile_load(&CONCAT_I(dense_scale_tile), CONCAT_I(dense_scale_ptr), m, 1, 0, sg_j0, 0);
#else
    binary_dense_in_tile_type CONCAT_I(dense_in_tile);
    tile_load(&CONCAT_I(dense_in_tile), CONCAT_I(dense_scale_ptr), m, 1, 0, sg_j0, 0);
    tile_convert(CONCAT_I(dense_in_tile), CONCAT_I(dense_scale_tile), BINARY_SCALE_DENSE_TO_FLOAT);
#endif
    tile_hbroadcast_mul(c_tile, CONCAT_I(dense_scale_tile));
#undef CONCAT_I
)",
                        i);
            } else if (kind == po_kind_t::binary_nvfp4_scale) {
                s += utils::format(
                        R"(
    float gs_%d = nvfp4_scale[batch];
#define binary_mul_%d(v) ((v) * gs_%d)
    tile_elementwise((*c_tile), binary_mul_%d);
#undef binary_mul_%d
)",
                        i, i, i, i, i);
            }
        }
    }

    s += "}\n";
    return s;
}

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
