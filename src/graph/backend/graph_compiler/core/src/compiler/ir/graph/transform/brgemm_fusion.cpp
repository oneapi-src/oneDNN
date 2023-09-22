/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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
#include <memory>
#include <vector>
#include <compiler/dimensions.hpp>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/tunable_op.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <ops/fusible/binary_elemwise.hpp>
#include <ops/fusible/unary_elemwise.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
#include <util/math_utils.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static const constexpr char *forbidden_brgemm_fusion
        = "forbidden_brgemm_fusion";

enum class brg_broadcast_type {
    non_broadcast = 0, // {N, C, H, W}
    non_channel, // {N, 1, H, W} => {N, C, H, W}
    only_channel, // {1, C, 1, 1} => {N, C, H, W}
    all_broadcast, // {1}
};

static bool all_ones_except_channel(const sc_dims &shapes, int channel_axis) {
    assert(!shapes.empty());
    for (int i = 0; i < static_cast<int>(shapes.size()); i++) {
        if (i != channel_axis && shapes[i] != 1) { return false; }
    }
    return true;
}

static op_traits::brgemm_fusion_acceptable_t *to_brg_fusible(
        const sc_op_ptr &p) {
    return p->dyn_cast<op_traits::brgemm_fusion_acceptable_t>();
}

static brg_broadcast_type get_brgemm_broadcast_type(
        const sc_data_format_t &l_format, const sc_data_format_t &r_format,
        const sc_dims &l_plain_dims, const sc_dims &r_plain_dims) {
    int channel_axis
            = l_format.format_code_.get(l_format.format_code_.ndims() - 1);
    sc_dim channel_size = l_plain_dims.at(channel_axis);
    COMPILE_ASSERT((l_format.format_code_ == r_format.format_code_
                           && l_plain_dims.size() == r_plain_dims.size())
                    || r_plain_dims.size() == 1,
            "The tensors on the left and right hands cannot be calculated.");
    sc_dims expected_all_broadcast {1};
    if (r_plain_dims == l_plain_dims
            && !all_ones_except_channel(l_plain_dims, channel_axis)) {
        return brg_broadcast_type::non_broadcast;
    } else if (r_plain_dims == expected_all_broadcast) {
        return brg_broadcast_type::all_broadcast;
    } else if (r_plain_dims.size() == 1
            || r_plain_dims.at(channel_axis) == l_plain_dims.at(channel_axis)) {
        return brg_broadcast_type::only_channel;
    } else if (r_plain_dims.at(channel_axis) == 1) {
        return brg_broadcast_type::non_channel;
    } else {
        throw std::runtime_error(
                "The shapes on left and right sides do not correspond!");
    }
}
struct fusion_state_t {
    int max_op_count_ = 1;
    int cur_op_count_ = 0;
    template <typename T>
    bool isa() const {
        static_assert(std::is_base_of<fusion_state_t, T>::value,
                "T is not a subclass of fusion_state_t.");
        return dynamic_cast<const T *>(this);
    }
    virtual void reset() { cur_op_count_ = 0; }
    // the cur_op can be accepted or not in this state.
    virtual bool is_acceptable(const sc_op_ptr &cur_op) = 0;
    virtual std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            = 0;
    virtual void do_transform(sc_graph_t &graph, const sc_op_ptr &main_op,
            const sc_op_ptr &cur_op) {
        to_brg_fusible(cur_op)->fuse_in_brgemm_ = true;
    }
    virtual ~fusion_state_t() = default;
};
#define GET_BROADCAST_TYPE() \
    auto l_format = cur_op->get_inputs()[0]->details_.get_format(); \
    auto r_format = cur_op->get_inputs()[1]->details_.get_format(); \
    auto l_plain_dims = cur_op->get_inputs()[0]->details_.get_plain_dims(); \
    auto r_plain_dims = cur_op->get_inputs()[1]->details_.get_plain_dims(); \
    auto bct_type = get_brgemm_broadcast_type( \
            l_format, r_format, l_plain_dims, r_plain_dims);

struct cast_state_t;
struct scale_state_t;
struct bias_state_t;
struct elem_state_t;
struct c_zp_state_t;
struct ab_zp_state_t : public fusion_state_t {
    bool has_a_zp = false;
    bool has_b_zp = false;
    ab_zp_state_t() : has_a_zp(false), has_b_zp(false) { max_op_count_ = 2; }
    void reset() override {
        cur_op_count_ = 0;
        has_a_zp = false;
        has_b_zp = false;
    }
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<sub_op_t>()) { return false; }
        auto sub_op = cur_op->stc_cast<sub_op_t>();
        if (sub_op->get_broadcast_input() == 0) { return false; }
        GET_BROADCAST_TYPE();
        // if non-broadcast or all broadcast, return
        if (bct_type == brg_broadcast_type::non_broadcast
                || bct_type == brg_broadcast_type::all_broadcast) {
            return false;
        }
        // should be s32
        if (!cur_op->get_inputs()[0]->details_.dtype_.is_etype(
                    sc_data_etype::S32)
                || !cur_op->get_inputs()[1]->details_.dtype_.is_etype(
                        sc_data_etype::S32)) {
            return false;
        }
        if (!has_a_zp && bct_type == brg_broadcast_type::only_channel) {
            has_a_zp = true;
            cur_op_count_++;
            sub_op->alg_kind_ = brgemm::alg_kind_t::a_zp;
            return true;
        }
        if (!has_b_zp && bct_type == brg_broadcast_type::non_channel) {
            has_b_zp = true;
            cur_op_count_++;
            sub_op->alg_kind_ = brgemm::alg_kind_t::b_zp;
            return true;
        }
        return false;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<ab_zp_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }
};

struct cast_state_t : public fusion_state_t {
    // calculation type cast, memory storage type cast
    cast_state_t() { max_op_count_ = 2; }
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<cast_op_t>()) { return false; }
        cur_op_count_++;
        cur_op->stc_cast<cast_op_t>()->alg_kind_
                = brgemm::alg_kind_t::out_dtype;
        return true;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<bias_state_t>() || st->isa<cast_state_t>()
                    || st->isa<elem_state_t>() || st->isa<scale_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }
};

struct scale_state_t : public fusion_state_t {
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<mul_op_t>()) { return false; }
        auto mul_op = cur_op->stc_cast<mul_op_t>();
        if (mul_op->get_broadcast_input() == 0) { return false; }

        GET_BROADCAST_TYPE();
        // support only_channel, all broadcast
        if (bct_type == brg_broadcast_type::non_broadcast
                || bct_type == brg_broadcast_type::non_channel) {
            return false;
        }
        if (bct_type == brg_broadcast_type::all_broadcast
                && !mul_op->get_inputs()[1]
                            ->producer_owner_->isa<constant_op_t>()) {
            return false;
        }
        // should be f32
        if (!cur_op->get_inputs()[0]->details_.dtype_.is_etype(
                    sc_data_etype::F32)
                || !cur_op->get_inputs()[1]->details_.dtype_.is_etype(
                        sc_data_etype::F32)) {
            return false;
        }
        cur_op_count_++;
        mul_op->alg_kind_ = brgemm::alg_kind_t::out_scales;
        return true;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<elem_state_t>()
                    || st->isa<c_zp_state_t>() || st->isa<bias_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }

    // scalar scale to vector
    void do_transform(sc_graph_t &graph, const sc_op_ptr &main_op,
            const sc_op_ptr &cur_op) override {
        if (auto const_op
                = cur_op->get_inputs()[1]
                          ->producer_owner_->dyn_cast<constant_op_t>()) {
            if (const_op->get_constant_plain_dims() == sc_dims {1}) {
                float scale = reinterpret_cast<float *>(
                        const_op->get_constant_values()->data_)[0];
                auto main_format
                        = main_op->get_outputs()[0]->details_.get_format();
                auto main_plain_shapes
                        = main_op->get_outputs()[0]->details_.get_plain_dims();
                auto &fcode = main_format.format_code_;
                int channel_axis = fcode.get(fcode.ndims() - 1);
                sc_dim channel_size = main_plain_shapes.at(channel_axis);
                std::vector<float> scales(channel_size, scale);
                sc_dims new_plain_dims(main_plain_shapes.size(), 1);
                new_plain_dims.at(channel_axis)
                        = main_plain_shapes.at(channel_axis);
                auto new_format = main_format;
                for (int i = 0; i < new_format.format_code_.norig_dims(); i++) {
                    auto blk_idxs
                            = new_format.format_code_.collect_blocking_index(i);
                    if (i != channel_axis) {
                        for (auto &idx : blk_idxs) {
                            new_format.blocks_[idx] = 1;
                        }
                    }
                }
                auto new_const_op = graph.make("constant", {}, {},
                        {{"values", std::make_shared<static_data_t>(scales)},
                                {"dtype", datatypes::f32},
                                {"plain_dims", new_plain_dims},
                                {"format", new_format}});
                auto new_scale_op = graph.make("mul",
                        {cur_op->get_inputs()[0],
                                new_const_op->get_outputs()[0]},
                        {}, cur_op->attrs_);
                auto new_scale_fu = to_brg_fusible(new_scale_op);
                new_scale_fu->alg_kind_ = brgemm::alg_kind_t::out_scales;
                new_scale_fu->fuse_in_brgemm_ = true;
                cur_op->replace_uses_with_and_remove(new_scale_op);
                const_op->remove();
                return;
            }
        }
        auto cur_op_fu = to_brg_fusible(cur_op);
        cur_op_fu->alg_kind_ = brgemm::alg_kind_t::out_scales;
        cur_op_fu->fuse_in_brgemm_ = true;
    }
};

struct bias_state_t : public fusion_state_t {
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<add_op_t>()) { return false; }
        auto add_op = cur_op->stc_cast<add_op_t>();
        if (add_op->get_broadcast_input() == 0) { return false; }
        GET_BROADCAST_TYPE();
        // support only_channel
        if (bct_type != brg_broadcast_type::only_channel) { return false; }
        // should be f32, bf16 or fp16
        if (!utils::is_one_of(cur_op->get_inputs()[0]->details_.dtype_,
                    datatypes::bf16, datatypes::f16, datatypes::f32)) {
            return false;
        }
        cur_op_count_++;
        add_op->alg_kind_ = brgemm::alg_kind_t::bias_add;
        return true;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<elem_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }

    // e.g. conv + cast + scale + bias => conv + bias + cast + scale
    void do_transform(sc_graph_t &graph, const sc_op_ptr &main_op,
            const sc_op_ptr &cur_op) override {
        // s32 + cast + f32
        if (main_op->get_outputs()[0]->details_.dtype_.is_etype(
                    sc_data_etype::S32)
                && cur_op->get_inputs()[0]->details_.dtype_.is_etype(
                        sc_data_etype::F32)) {
            COMPILE_ASSERT(main_op->attrs_.has_key("data_scales")
                            && main_op->attrs_.has_key("weight_scales"),
                    "Main op is not quantized but output s32.");
            auto l_plain_dims
                    = main_op->get_outputs()[0]->details_.get_plain_dims();
            auto r_plain_dims
                    = cur_op->get_inputs()[1]->details_.get_plain_dims();
            auto &r_format = cur_op->get_inputs()[1]->details_.get_format();
            auto &fcode = r_format.format_code_;
            int channel_axis = fcode.get(fcode.ndims() - 1);
            sc_dim channel_size = r_plain_dims.at(channel_axis);
            auto data_scales
                    = main_op->attrs_.get<std::vector<float>>("data_scales");
            auto weight_scales
                    = main_op->attrs_.get<std::vector<float>>("weight_scales");
            auto output_scales
                    = math_utils::vector_mul(data_scales, weight_scales);
            auto next_op_by_main = main_op->get_outputs()[0]->uses_[0].second;
            assert(main_op->get_outputs()[0]->uses_[0].first == 0);
            assert(next_op_by_main.get() != cur_op.get());
            output_scales = math_utils::vector_rcp(output_scales);
            if (output_scales.size() == 1) {
                output_scales
                        = std::vector<float>(channel_size, output_scales[0]);
            }
            sc_dims const_plain_dims = r_plain_dims;
            auto const_scales = graph.make("constant", {}, {},
                    {{"values", std::make_shared<static_data_t>(output_scales)},
                            {"dtype", datatypes::f32},
                            {"plain_dims", const_plain_dims},
                            {"format", r_format}});
            auto bias_mul_scales = graph.make("mul",
                    {cur_op->get_inputs()[1], const_scales->get_outputs()[0]},
                    {}, {});
            auto castf32 = graph.make("cast", main_op->get_outputs(), {},
                    {{"dtype", datatypes::f32}});
            auto bias_add = graph.make("add",
                    {castf32->get_outputs()[0],
                            bias_mul_scales->get_outputs()[0]},
                    {}, cur_op->attrs_);
            auto casts32 = graph.make("cast", bias_add->get_outputs(), {},
                    {{"dtype", datatypes::s32}});
            to_brg_fusible(bias_add)->alg_kind_ = brgemm::alg_kind_t::bias_add;
            to_brg_fusible(castf32)->fuse_in_brgemm_ = true;
            to_brg_fusible(bias_add)->fuse_in_brgemm_ = true;
            to_brg_fusible(casts32)->fuse_in_brgemm_ = true;
            next_op_by_main->replace_input(0, casts32->get_outputs()[0]);
            auto pre_op_by_cur = cur_op->get_inputs()[0]->producer_owner_;
            auto uses = cur_op->get_outputs()[0]->uses_;
            for (auto &use : uses) {
                use.second->replace_input(
                        use.first, pre_op_by_cur->get_outputs()[0]);
            }
            cur_op->remove();
            return;
        }
        auto cur_op_fu = to_brg_fusible(cur_op);
        cur_op_fu->alg_kind_ = brgemm::alg_kind_t::bias_add;
        cur_op_fu->fuse_in_brgemm_ = true;
    }
};

struct elem_state_t : public fusion_state_t {};
struct unary_elem_state_t : public elem_state_t {
    unary_elem_state_t() {
        max_op_count_ = brgemm::postops_setting_t::max_postops_num;
    }
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        // cast cur_op should be processed by cast_state_t.
        if (cur_op->isa<cast_op_t>()) { return false; }
        if (!cur_op->isa<unary_elementwise_op_t>()) { return false; }
        cur_op_count_++;
        return true;
    }

    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<elem_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }
};

struct binary_elem_state_t : public elem_state_t {
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<binary_elementwise_op_t>()) { return false; }
        auto bin_op = cur_op->stc_cast<binary_elementwise_op_t>();
        if (bin_op->get_broadcast_input() == 0) { return false; }
        GET_BROADCAST_TYPE();
        // support only_channel, non broadcast
        if (bct_type == brg_broadcast_type::all_broadcast
                || bct_type == brg_broadcast_type::non_channel) {
            return false;
        }
        cur_op_count_++;
        return true;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<unary_elem_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }
};

struct c_zp_state_t : public fusion_state_t {
    bool is_acceptable(const sc_op_ptr &cur_op) override {
        if (cur_op_count_ >= max_op_count_) { return false; }
        if (!cur_op->isa<add_op_t>()) { return false; }
        auto add_op = cur_op->stc_cast<add_op_t>();
        if (add_op->get_broadcast_input() == 0) { return false; }
        GET_BROADCAST_TYPE();
        if (bct_type != brg_broadcast_type::all_broadcast) { return false; }
        // should be f32
        if (!cur_op->get_inputs()[0]->details_.dtype_.is_etype(
                    sc_data_etype::F32)
                || !cur_op->get_inputs()[1]->details_.dtype_.is_etype(
                        sc_data_etype::F32)) {
            return false;
        }
        cur_op_count_++;
        add_op->alg_kind_ = brgemm::alg_kind_t::c_zp;
        return true;
    }
    std::vector<std::shared_ptr<fusion_state_t>> next_possible_states(
            const std::vector<std::shared_ptr<fusion_state_t>> &all_states)
            override {
        std::vector<std::shared_ptr<fusion_state_t>> ret;
        for (auto &st : all_states) {
            if (st->isa<cast_state_t>() || st->isa<elem_state_t>()
                    || st->isa<bias_state_t>()) {
                ret.emplace_back(st);
            }
        }
        return ret;
    }
};

std::vector<std::shared_ptr<fusion_state_t>> create_all_possible_states(
        bool include_zp = false, bool include_binary = false) {
    std::vector<std::shared_ptr<fusion_state_t>> ret;
    if (include_zp) { ret.emplace_back(std::make_shared<ab_zp_state_t>()); }
    ret.emplace_back(std::make_shared<cast_state_t>());
    ret.emplace_back(std::make_shared<scale_state_t>());
    ret.emplace_back(std::make_shared<bias_state_t>());
    ret.emplace_back(std::make_shared<unary_elem_state_t>());
    // onednn binary postop injector has redundant latency.
    if (include_binary) {
        ret.emplace_back(std::make_shared<binary_elem_state_t>());
    }
    if (include_zp) { ret.emplace_back(std::make_shared<c_zp_state_t>()); }
    return ret;
}

// Mark the alg kind of brgemm fusion ops and do brgemm-friendly
// transformation (like move bias closer to main op; use vector buffer for
// scales. It does not change calculation itself)
// As this transform pass is only used in `fuse_ops` pass, so we introduce two
// auxiliary ops
void brgemm_fusion_transform(sc_graph_t &graph, const context_ptr &ctx) {
    if (graph.attrs_.get_or_else("temp.disable_graph_fusion", 0) == 1) {
        return;
    }
    // todo: need interleave optimization on amx
    if (ctx->machine_.cpu_flags_.fAVX512AMXTILE) { return; }
    auto ops = graph.ops_;
    for (auto &main_op : ops) {
        if (!main_op->isa<tunable_op_t>()) { continue; }
        if (!main_op->is_single_output_single_use()) { continue; }
        sc_op_ptr cur_op = main_op;
        // todo : enable zp
        bool include_zp = graph.attrs_.get_or_else(
                "temp.brgemm_fusion_include_zp", false);
        // currently binary injector has slow implement.
        bool include_binary = graph.attrs_.get_or_else(
                "temp.brgemm_fusion_include_binary", false);
        static std::vector<std::shared_ptr<fusion_state_t>> all_possible_states
                = create_all_possible_states(false);
        static std::vector<std::shared_ptr<fusion_state_t>>
                all_possible_states_include_zp
                = create_all_possible_states(true);
        auto all_states = include_zp ? all_possible_states_include_zp
                                     : all_possible_states;
        if (include_binary) {
            all_states = create_all_possible_states(include_zp, true);
        }
        // reset because of the static states
        for (auto &st : all_states) {
            st->reset();
        }
        int fused_op_count = 0;
        auto next_op = cur_op->get_outputs()[0]->uses_[0].second;
        cur_op = next_op;
        auto next_states = all_states;
        while (fused_op_count < brgemm::postops_setting_t::max_postops_num
                && !cur_op->isa<output_op>()) {
            // store next op as cur op may be transformed by new op
            next_op = cur_op->get_outputs()[0]->uses_[0].second;
            bool has_next = cur_op->is_single_output_single_use();
            bool accept = false;
            for (auto &st : next_states) {
                if (st->is_acceptable(cur_op)) {
                    st->do_transform(graph, main_op, cur_op);
                    next_states = st->next_possible_states(all_states);
                    fused_op_count++;
                    accept = true;
                    break;
                }
            }
            if (!accept || !has_next) { break; }
            cur_op = next_op;
        }
    }
    graph.reset_op_ids();
}
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
