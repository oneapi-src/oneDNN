/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#include "../visitor.hpp"
#include "transform.hpp"
#include <compiler/ir/graph/pass/pass.hpp>
#include <ops/convolution.hpp>
#include <ops/templates/conv_rl.hpp>
#include <ops/templates/utils.hpp>
#include <runtime/config.hpp>

SC_MODULE(rl_conv_weight_transform)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static inline int get_full_lowering_threshold(int tile_col) {
    int threshold = static_cast<int>(tile_col * 0.75);
    return threshold;
}

static bool use_rl(const context_ptr &ctx, const sc_data_type_t &data_dtype,
        const sc_dims &data_dims, const sc_dims &weight_dims,
        const sc_dims &pads_begin, const sc_dims &pads_end) {
    auto ndims = data_dims.size();
    assert(ndims == 4 && weight_dims.size() == ndims);
    if (!ops::is_amx_dtype(ctx, data_dtype)) { return false; }
    bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, data_dtype);
    int vnni_blk = is_vnni_low_fp ? 2 : 4;
    int tile_col = is_vnni_low_fp ? 32 : 64;
    int threshold = get_full_lowering_threshold(tile_col);
    bool is_1x1 = std::all_of(weight_dims.begin() + 2, weight_dims.end(),
            [](int x) { return x == 1; });
    bool is_small_padding = std::all_of(pads_begin.begin(), pads_begin.end(),
                                    [weight_dims, ndims](int x) {
                                        return x <= weight_dims[ndims - 1];
                                    })
            && std::all_of(pads_end.begin(), pads_end.end(),
                    [weight_dims, ndims](
                            int x) { return x <= weight_dims[ndims - 1]; });
    auto ic = weight_dims[1];
    auto kw = weight_dims[ndims - 1];
    return (!is_1x1 && ndims == 4 && is_small_padding && (ic <= (tile_col / 2))
            && ((ic % vnni_blk != 0 && kw * ic <= threshold)
                    || (ic % vnni_blk == 0)));
}

static void query_accu_info_for_rl(const context_ptr &ctx,
        const sc_data_type_t &dtype, const int kh, const int kw, const int ic,
        const int LDA, int &num_brgemm_k, int &brgemm_k, int &extra_padding,
        int &kind) {
    assert(ops::is_amx_dtype(ctx, dtype));
    bool is_vnni_low_fp = ops::is_vnni_low_fp(ctx, dtype);
    int vnni_blk = is_vnni_low_fp ? 2 : 4;
    int tile_col = is_vnni_low_fp ? 32 : 64;
    int threshold = get_full_lowering_threshold(tile_col);

    if (kw * ic <= threshold) {
        kind = ops::rl_kind::FULL_LOWERING;

        auto total_raw_accu = kw * kh * ic;
        num_brgemm_k = utils::divide_and_ceil(total_raw_accu, tile_col);
        auto total_padded_accu
                = utils::rnd_up(total_raw_accu, num_brgemm_k * vnni_blk);
        brgemm_k = total_padded_accu / num_brgemm_k;
        extra_padding = total_padded_accu - total_raw_accu;
    } else {
        kind = ops::rl_kind::KW_LOWERING;

        int padding1 = 0, padding2 = 0;
        int brgemm_k1 = 0, brgemm_k2 = 0;
        int num_brgemm_k1 = 0, num_brgemm_k2 = 0;
        auto get_num_brgemm_k_and_padding
                = [&ic, &kh, &kw](int brg_k, int &num_brg_k, int &padding) {
                      num_brg_k = utils::divide_and_ceil(ic * kw, brg_k) * kh;
                      auto padded_accu_per_row = brg_k * num_brg_k / kh;
                      padding = padded_accu_per_row - ic * kw;
                  };
        {
            // 1) greedy
            int k_len = 1;
            for (; k_len <= kw; ++k_len) {
                if (ic * k_len > tile_col) break;
            }
            assert(k_len > 1);
            brgemm_k1 = ic * (k_len - 1);
            get_num_brgemm_k_and_padding(brgemm_k1, num_brgemm_k1, padding1);
        }

        {
            // 2) compact
            int num_k = utils::divide_and_ceil(ic * kw, tile_col);
            int num_kw = utils::divide_and_ceil(kw, num_k);
            while (num_kw * ic > tile_col)
                num_kw--;
            brgemm_k2 = ic * num_kw;
            get_num_brgemm_k_and_padding(brgemm_k2, num_brgemm_k2, padding2);
        }

        if (padding1 < padding2) {
            brgemm_k = brgemm_k1;
            num_brgemm_k = num_brgemm_k1;
            extra_padding = padding1;
        } else {
            brgemm_k = brgemm_k2;
            num_brgemm_k = num_brgemm_k2;
            extra_padding = padding2;
        }
        COMPILE_ASSERT(extra_padding % ic == 0,
                "Expect extra_padding is dividable by ic, but got "
                "extra_padding="
                        << extra_padding << ",ic=" << ic << ".");
    }
}

void rl_conv_weight_transform(sc_graph_t &graph, const context_ptr &ctx) {
    if ((graph.attrs_.has_key("use_rl")
                && graph.attrs_.get<int>("use_rl") == ops::rl_kind::NO_LOWERING)
            || graph.is_dynamic()) {
        return;
    }
    auto vis = op_visitor_t::bfs();
    vis.visit_graph(graph, [&](op_visitor_t *vis, const sc_op_ptr &node) {
        if (auto op = node->dyn_cast<ops::conv_fwd_core_op_t>()) {
            auto data_plain_dims
                    = op->info_.inputs_[0]->details_.get_plain_dims();
            auto weight_plain_dims
                    = op->info_.inputs_[1]->details_.get_plain_dims();
            auto data_dtype = op->info_.inputs_[0]->details_.dtype_;
            auto ndims = data_plain_dims.size();
            sc_dim groups = op->attrs_.get_or_else("groups", 1);
            auto dilations = ops::get_dilations(op->attrs_);
            auto has_dilation = std::any_of(dilations.begin(), dilations.end(),
                    [](int x) { return x != 1; });
            // Note, dilation will introduce non-contiguous data when packing
            // kw, might be supported in future
            if (has_dilation) { return; };

            if (ndims != 4) { return; }
            COMPILE_ASSERT(weight_plain_dims.size() == ndims,
                    "Weight dims size is expected equal to data dims, but got "
                            << weight_plain_dims.size() << " vs. " << ndims
                            << ".");
            COMPILE_ASSERT(weight_plain_dims[1] == data_plain_dims[1] / groups,
                    "Weight_plain_dims[1] is expected equal to "
                    "data_plain_dims[1]/groups, but got "
                            << weight_plain_dims[1] << " vs. "
                            << data_plain_dims[1] / groups << ".");
            sc_dims pads_begin, pads_end;
            if (op->attrs_.has_key("pads_begin")) {
                pads_begin = op->attrs_.get<sc_dims>("pads_begin");
                pads_end = op->attrs_.get<sc_dims>("pads_end");
            } else {
                pads_begin = op->attrs_.get<sc_dims>("paddings");
                pads_end = pads_begin;
            }

            sc_dims strides = op->attrs_.get<sc_dims>("strides");
            if (!use_rl(ctx, data_dtype, data_plain_dims, weight_plain_dims,
                        pads_begin, pads_end)) {
                return;
            }

            auto oc = weight_plain_dims[0];
            auto kh = weight_plain_dims[ndims - 2];
            auto kw = weight_plain_dims[ndims - 1];
            auto ic = data_plain_dims[1] / groups;
            auto &stride = op->attrs_.get<sc_dims>("strides");
            auto sw = !stride.empty() ? stride[1] : stride[0];
            int num_brgemm_k = 1;
            int brgemm_k = 1;
            int extra_padding = 0;
            auto kind = ops::rl_kind::NO_LOWERING;
            auto LDA = kw * ic * sw;

            query_accu_info_for_rl(ctx, data_dtype, kh, kw, ic, LDA,
                    num_brgemm_k, brgemm_k, extra_padding, kind);
            if (kind == ops::rl_kind::NO_LOWERING) { return; }
#if 0
            //     fall-back for small amx utilization
                auto low_tmul_utilization
                        = [](int N, int K, sc_data_type_t &dtype) {
                              auto dtype_size = (dtype == datatypes::bf16) ?
                              2 : 4; if (N <= 8 || K <= (64 / dtype_size)) {
                                  return true;
                              } else {
                                  return false;
                              }
                          };
                if (ops::is_amx_dtype(ctx, data_dtype)
                        && low_tmul_utilization(
                                oc / groups, brgemm_k, data_dtype)) {
                    return;
                }
#endif

            op->attrs_["use_rl"] = kind;
            op->attrs_["num_brgemm_k"] = num_brgemm_k;
            op->attrs_["brgemm_k"] = brgemm_k;
            op->attrs_["extra_padding"] = extra_padding;
            op->attrs_["origin_wei_plain_dims"] = sc_dims {oc, ic, kh, kw};

            auto weight = op->get_inputs()[1];
            auto weight_op = weight->producer_owner_;
            auto is_constant_weight = (weight_op->isa<constant_op_t>()
                    || weight_op->attrs_.get_or_else(
                            "constant", const_kind::not_const));
            if (kind == ops::rl_kind::FULL_LOWERING) {
                // B-OIHW->reorder->HWIO->view->KN->(padding on "K")
                auto wei_shape_2d = sc_dims {ic * kh * kw, oc};
                auto trans_wei = graph.make("reorder", {weight}, {},
                        {{"out_format", sc_data_format_t(format_kinds::CDBA)},
                                {"internal", true}});
                if (is_constant_weight) {
                    trans_wei->attrs_.set("constant", const_kind::local_const);
                }

                auto view_wei = graph.make("tensor_view",
                        trans_wei->get_outputs(), {},
                        {{"shape", wei_shape_2d},
                                {"format", sc_data_format_t(format_kinds::AB)},
                                {"expand_dim", std::vector<int> {}}});
                if (is_constant_weight) {
                    view_wei->attrs_.set("constant", const_kind::local_const);
                }
                vis->update_state_for_visited(trans_wei);
                vis->update_state_for_visited(view_wei);

                if (extra_padding > 0) {
                    auto padded_wei = graph.make("padding",
                            view_wei->get_outputs(), {},
                            {{"pads_begin", sc_dims {0}},
                                    {"pads_end", sc_dims {extra_padding}}});

                    if (is_constant_weight) {
                        padded_wei->attrs_.set(
                                "constant", const_kind::local_const);
                    }
                    op->replace_input(1, padded_wei->get_outputs()[0], true);
                    vis->update_state_for_visited(padded_wei);
                } else {
                    op->replace_input(1, view_wei->get_outputs()[0], true);
                }
            } else {
                // B-OIHW->OIHW'(padding on rhs of "W")
                if (extra_padding > 0) {
                    auto padded_wei = graph.make("padding", {weight}, {},
                            {{"pads_begin", sc_dims {0, 0}},
                                    {"pads_end",
                                            sc_dims {0, extra_padding / ic}}});

                    if (is_constant_weight) {
                        padded_wei->attrs_.set(
                                "constant", const_kind::local_const);
                    }
                    op->replace_input(1, padded_wei->get_outputs()[0], true);
                    vis->update_state_for_visited(padded_wei);
                }
            }
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
