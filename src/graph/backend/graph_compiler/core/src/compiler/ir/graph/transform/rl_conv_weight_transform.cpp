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
#include <ops/templates/utils.hpp>
#include <runtime/config.hpp>

SC_MODULE(rl_conv_weight_transform)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static bool use_rl(const context_ptr &ctx, const sc_data_type_t &data_dtype,
        const sc_dims &data_dims, const sc_dims &weight_dims,
        const sc_dims &pads_begin, const sc_dims &pads_end) {
    auto ndims = data_dims.size();
    assert(ndims == 4 && weight_dims.size() == ndims);
    const bool is_1x1 = std::all_of(weight_dims.begin() + 2, weight_dims.end(),
            [](int x) { return x == 1; });
    constexpr int tile_col = 64;
    const int threshold = static_cast<int>(0.75 * tile_col);
    const bool is_small_padding
            = std::all_of(pads_begin.begin(), pads_begin.end(),
                      [weight_dims, ndims](
                              int x) { return x <= weight_dims[ndims - 1]; })
            && std::all_of(pads_end.begin(), pads_end.end(),
                    [weight_dims, ndims](
                            int x) { return x <= weight_dims[ndims - 1]; });
    auto dtype_size = utils::get_sizeof_type(data_dtype);
    auto kw = weight_dims[ndims - 1];
    auto ic = weight_dims[1];
    auto is_amx_dtype = ops::is_amx_dtype(ctx, data_dtype);

    return (!is_1x1 && ndims == 4 && is_amx_dtype
            && (kw * ic * dtype_size < threshold) && is_small_padding);
}

static void query_accu_info_for_rl(const context_ptr &ctx,
        const sc_data_type_t &dtype, const int kh, const int kw, const int ic,
        const int LDA, int &num_brgemm_k, int &brgemm_k, int &extra_padding) {
    assert(ops::is_amx_dtype(ctx, dtype));
    bool is_bf16 = (dtype == datatypes::bf16);
    int max_col = is_bf16 ? 32 : 64;
    auto total_raw_accu = kw * kh * ic;
    num_brgemm_k = utils::divide_and_ceil(total_raw_accu, max_col);

    auto vnni_blk = is_bf16 ? 2 : 4;
    auto total_padded_accu
            = utils::rnd_up(total_raw_accu, num_brgemm_k * vnni_blk);
    brgemm_k = total_padded_accu / num_brgemm_k;

    // Note: to accommodate oneDNN BRGEMM's SW limitation
    if ((ctx->flags_.kernel_optim_ == 1) && (brgemm_k > LDA)) {
        assert(LDA >= vnni_blk);
        SC_MODULE_WARN
                << "split to smaller K due to oneDNN BRGEMM's limitation.";

        brgemm_k = vnni_blk;
        for (int k = LDA; k > 0; --k) {
            if (k % vnni_blk == 0) {
                brgemm_k = k;
                break;
            }
        }
        num_brgemm_k = utils::divide_and_ceil(total_raw_accu, brgemm_k);
        total_padded_accu = num_brgemm_k * brgemm_k;
        COMPILE_ASSERT(brgemm_k <= LDA,
                "oneDNN BRGEMM requires K<=LDA, but got K="
                        << brgemm_k << ", LDA=" << LDA << ".");
    }

    extra_padding = total_padded_accu - total_raw_accu;
}

void rl_conv_weight_transform(sc_graph_t &graph, const context_ptr &ctx) {
    if (!graph.attrs_.get_or_else("use_rl", true) || graph.is_dynamic()) {
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

            if (ndims != 4) { return; }
            COMPILE_ASSERT(weight_plain_dims.size() == ndims,
                    "Weight dims size is expected equal to data dims, but got "
                            << weight_plain_dims.size() << " vs. " << ndims
                            << ".");
            COMPILE_ASSERT(weight_plain_dims[1] == data_plain_dims[1],
                    "Weight_plain_dims[1] is expected equal to "
                    "data_plain_dims[1], but got "
                            << weight_plain_dims[1] << " vs. "
                            << data_plain_dims[1] << ".");
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
            auto ic = data_plain_dims[1];
            auto &stride = op->attrs_.get<sc_dims>("strides");
            auto sw = !stride.empty() ? stride[1] : stride[0];
            int num_brgemm_k = 1;
            int brgemm_k = 1;
            int extra_padding = 0;
            auto LDA = kw * ic * sw;

            // TODO(ciyong): remove this constraint once LDX limitation is
            // removed in oneDNN BRGEMM.
            int max_col = data_dtype == datatypes::bf16 ? 32 : 64;
            if ((ctx->flags_.kernel_optim_ == 1) && (LDA < max_col / 4)) {
                return;
            }

            query_accu_info_for_rl(ctx, data_dtype, kh, kw, ic, LDA,
                    num_brgemm_k, brgemm_k, extra_padding);

            op->attrs_["use_rl"] = true;
            op->attrs_["num_brgemm_k"] = num_brgemm_k;
            op->attrs_["brgemm_k"] = brgemm_k;
            op->attrs_["extra_padding"] = extra_padding;
            op->attrs_["origin_wei_plain_dims"] = sc_dims {oc, ic, kh, kw};

            // B-OIHW->reorder->HWIO->view->KN->(padding on "K")
            auto weight = op->get_inputs()[1];
            auto wei_shape_2d = sc_dims {ic * kh * kw, oc};
            auto weight_op = weight->producer_owner_;
            auto is_constant_weight = (weight_op->isa<constant_op_t>()
                    || weight_op->attrs_.get_or_else(
                            "constant", const_kind::not_const));
            auto trans_wei = graph.make("reorder", {weight}, {},
                    {{"out_format", sc_data_format_t(format_kinds::CDBA)},
                            {"internal", true}});
            if (is_constant_weight) {
                trans_wei->attrs_.set("constant", const_kind::local_const);
            }

            auto view_wei = graph.make("tensor_view", trans_wei->get_outputs(),
                    {},
                    {{"shape", wei_shape_2d},
                            {"format", sc_data_format_t(format_kinds::AB)},
                            {"expand_dim", std::vector<int> {}}});
            if (is_constant_weight) {
                view_wei->attrs_.set("constant", const_kind::local_const);
            }
            vis->update_state_for_visited(trans_wei);
            vis->update_state_for_visited(view_wei);

            if (extra_padding > 0) {
                auto padded_wei
                        = graph.make("padding", view_wei->get_outputs(), {},
                                {{"pads_begin", sc_dims {0}},
                                        {"pads_end", sc_dims {extra_padding}}});
                if (is_constant_weight) {
                    padded_wei->attrs_.set("constant", const_kind::local_const);
                }
                op->replace_input(1, padded_wei->get_outputs()[0], true);
                vis->update_state_for_visited(padded_wei);
            } else {
                op->replace_input(1, view_wei->get_outputs()[0], true);
            }
        }
    });
    graph.reset_op_ids();
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
