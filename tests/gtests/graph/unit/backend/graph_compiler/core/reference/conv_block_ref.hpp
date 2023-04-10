/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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
#ifndef GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_CONV_BLOCK_REF_HPP
#define GRAPH_UNIT_BACKEND_GRAPH_COMPILER_CORE_REFERENCE_CONV_BLOCK_REF_HPP

#include <assert.h>
#include <memory>
#include <utility>
#include <vector>
#include "conv_ref.hpp"
#include "eltwise_ref.hpp"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/body_generator.hpp>
#include <ops/convolution.hpp>
#include <ops/templates/conv_fwd.hpp>
#include <template/graph/conv_block.hpp>

namespace gc = dnnl::impl::graph::gc;
using gc::sc_dims;

static init_action buffer_action(bool need_rand) {
    return need_rand ? INIT_RANDOM : INIT_NOOP;
}
test_buffer<float> alloc_conv_block_array(size_t size, bool rand_non_negative) {
    if (rand_non_negative) {
        return alloc_array<float>(size, INIT_RANGE, 0.0, 1.0);
    } else {
        return alloc_array<float>(size);
    }
}
void compute_conv_block(gc::sc_graph_t &g, std::vector<gc::sc_op_ptr> &args,
        const sc_dims &input_dims,
        const std::vector<sc_dims> &weight_dims_block,
        const sc_dims &stride_block, const sc_dims &dilations_block,
        const sc_dims &padding_block,
        const std::vector<std::vector<gc::postop_type>> &post_types_block,
        const std::vector<gc::ops::conv_fwd_config_t> &cfg_ptr,
        test_buffer<float> &sc_output, test_buffer<float> &ref_output,
        bool rand_non_negative = false) {
    using namespace dnnl::impl::graph::gc;

    int nblocks = weight_dims_block.size();
    std::vector<bool> bias_block(nblocks, false), bn_relu_block(nblocks, false);
    for (int i = 0; i < nblocks; i++) {
        for (auto post_type : post_types_block.at(i)) {
            if (post_type == postop_type::bias) {
                bias_block[i] = true;
            } else if (post_type == postop_type::bn) {
                bn_relu_block[i] = true;
            }
        }
    }

    g.attrs_[sc_graph_t::attr_key_t::is_input_plain] = true;
    g.attrs_[sc_graph_t::attr_key_t::is_output_plain] = true;

    std::vector<std::pair<int, int>> p_q;
    std::vector<ops::conv_fwd_config_t> configs;
    int idx = 0;
    int N = input_dims.at(0);
    for (auto &op : g.ops_) {
        if (op->op_name_ == "conv_fwd_core") {
            auto conv_op = op->dyn_cast<ops::conv_fwd_core_op_t>();

            auto &out0_shape
                    = op->get_outputs().at(0)->details_.get_plain_dims();
            p_q.emplace_back(
                    std::make_pair(out0_shape.at(2), out0_shape.at(3)));
            if (!cfg_ptr.empty()) {
                conv_op->set_config(
                        reflection::general_object_t::make(cfg_ptr[idx]));
                configs.emplace_back(cfg_ptr[idx]);
            } else {
                body_generator_ptr gen = conv_op->create_generator();
                configs.emplace_back(
                        *(ops::conv_fwd_config_t *)gen
                                 ->get_default_config(get_default_context())
                                 .data_.get());
            }
            idx += 1;
        }
    }
    graph_driver(g);
    auto f = lower_graph(get_default_context(), g, args);
    auto fptr = jit_engine_t::make(get_default_context())
                        ->get_entry_func(f, true);
    int P, Q, H, W;
    H = input_dims.at(2);
    W = input_dims.at(3);

    std::tie(P, Q) = p_q.at(0);
    auto input = alloc_conv_block_array((size_t)N
                    * weight_dims_block.at(0).at(1) / configs.at(0).C_block * H
                    * W * configs.at(0).C_block,
            rand_non_negative);
    auto output = alloc_array<float>((size_t)N * weight_dims_block.at(0).at(0)
                    / configs.at(0).K_block * P * Q * configs.at(0).K_block,
            INIT_NOOP);
    auto weight = alloc_conv_block_array((size_t)weight_dims_block.at(0).at(0)
                    / configs.at(0).K_block * weight_dims_block.at(0).at(1)
                    / configs.at(0).C_block * weight_dims_block.at(0).at(2)
                    * weight_dims_block.at(0).at(3) * configs.at(0).C_block
                    * configs.at(0).K_block,
            rand_non_negative);
    auto ele_add = alloc_conv_block_array((size_t)N
                    * weight_dims_block.at(0).at(0) / configs.at(0).K_block * P
                    * Q * configs.at(0).K_block,
            rand_non_negative);
    auto bias = alloc_array<float>(
            weight_dims_block.at(0).at(0), buffer_action(bias_block.at(0)));
    auto bn_mul = alloc_array<float>(
            weight_dims_block.at(0).at(0), buffer_action(bn_relu_block.at(0)));
    auto bn_add = alloc_array<float>(
            weight_dims_block.at(0).at(0), buffer_action(bn_relu_block.at(0)));

    // ref data
    auto ref_input = input.copy();
    auto ref_weight = weight.copy();
    auto ref_ele_add = ele_add.copy();
    auto ref_bias = bias.copy();
    auto ref_bn_mul = bn_mul.copy();
    auto ref_bn_add = bn_add.copy();
    test_buffer<float> ref_out(
            (size_t)N * weight_dims_block.at(0).at(0) * P * Q);

    std::vector<test_buffer<float>> sc_args;
    std::vector<test_buffer<float>> ref_outputs, ref_weights, ref_biases,
            ref_bn_muls, ref_bn_adds, ref_ele_adds;

    std::vector<generic_val> generic_args;
    sc_args.emplace_back(std::move(input));
    sc_args.emplace_back(std::move(weight));

    for (auto post_type : post_types_block.at(0)) {
        switch (post_type) {
            case postop_type::bias:
                sc_args.emplace_back(std::move(bias));
                break;
            case postop_type::bn:
                sc_args.emplace_back(std::move(bn_mul));
                sc_args.emplace_back(std::move(bn_add));
                break;
            case postop_type::eleadd:
                sc_args.emplace_back(std::move(ele_add));
                break;
            default: break;
        }
    }
    // IF nblocks==1, push final output
    if (nblocks == 1) { sc_args.emplace_back(std::move(output)); }

    ref_outputs.emplace_back(std::move(ref_out));
    ref_weights.emplace_back(std::move(ref_weight));
    ref_biases.emplace_back(std::move(ref_bias));
    ref_ele_adds.emplace_back(std::move(ref_ele_add));
    ref_bn_muls.emplace_back(std::move(ref_bn_mul));
    ref_bn_adds.emplace_back(std::move(ref_bn_add));

    for (auto i = 1; i < nblocks; ++i) {
        std::tie(P, Q) = p_q.at(i);
        output = alloc_array<float>((size_t)N * weight_dims_block.at(i).at(0)
                        / configs.at(i).K_block * P * Q * configs.at(i).K_block,
                INIT_NOOP);
        weight = alloc_conv_block_array((size_t)weight_dims_block.at(i).at(0)
                        / configs.at(i).K_block * weight_dims_block.at(i).at(1)
                        / configs.at(i).C_block * weight_dims_block.at(i).at(2)
                        * weight_dims_block.at(i).at(3) * configs.at(i).C_block
                        * configs.at(i).K_block,
                rand_non_negative);
        ele_add = alloc_conv_block_array((size_t)N
                        * weight_dims_block.at(i).at(0) / configs.at(i).K_block
                        * P * Q * configs.at(i).K_block,
                rand_non_negative);
        bias = alloc_array<float>(
                weight_dims_block.at(i).at(0), buffer_action(bias_block.at(i)));
        bn_mul = alloc_array<float>(weight_dims_block.at(i).at(0),
                buffer_action(bn_relu_block.at(i)));
        bn_add = alloc_array<float>(weight_dims_block.at(i).at(0),
                buffer_action(bn_relu_block.at(i)));

        // ref data
        ref_weight = weight.copy();
        ref_ele_add = ele_add.copy();
        ref_bias = bias.copy();
        ref_bn_mul = bn_mul.copy();
        ref_bn_add = bn_add.copy();
        test_buffer<float> ref_out(
                (size_t)N * weight_dims_block.at(i).at(0) * P * Q);

        sc_args.emplace_back(std::move(weight));
        for (auto post_type : post_types_block.at(i)) {
            switch (post_type) {
                case postop_type::bias:
                    sc_args.emplace_back(std::move(bias));
                    break;
                case postop_type::bn:
                    sc_args.emplace_back(std::move(bn_mul));
                    sc_args.emplace_back(std::move(bn_add));
                    break;
                case postop_type::eleadd:
                    sc_args.emplace_back(std::move(ele_add));
                    break;
                default: break;
            }
        }
        // only push final output
        if (i == nblocks - 1) { sc_args.emplace_back(std::move(output)); }

        ref_outputs.emplace_back(std::move(ref_out));
        ref_weights.emplace_back(std::move(ref_weight));
        ref_ele_adds.emplace_back(std::move(ref_ele_add));
        ref_biases.emplace_back(std::move(ref_bias));
        ref_bn_muls.emplace_back(std::move(ref_bn_mul));
        ref_bn_adds.emplace_back(std::move(ref_bn_add));
    }

    for (unsigned i = 0; i < sc_args.size(); ++i)
        generic_args.emplace_back(sc_args.at(i).data());

    fptr->call_generic_default(generic_args.data());

    std::tie(P, Q) = p_q.at(0);
    compute_ref_direct_fwd(N, 1, weight_dims_block.at(0).at(0),
            weight_dims_block.at(0).at(1), H, W, P, Q,
            weight_dims_block.at(0).at(2), weight_dims_block.at(0).at(3),
            stride_block.at(0), stride_block.at(0), padding_block.at(0),
            padding_block.at(0), ref_input.data(), ref_weights.at(0).data(),
            ref_biases.at(0).data(), ref_outputs.at(0).data(),
            bias_block.at(0) ? dir_t::FWD_B : FWD_I, ref_bn_muls.at(0).data(),
            ref_bn_adds.at(0).data(), bn_relu_block.at(0), 1, 1, 1, 0, 1, 1,
            dilations_block.at(0), dilations_block.at(0));
    if (post_types_block.at(0).end()
            != std::find(post_types_block.at(0).begin(),
                    post_types_block.at(0).end(), postop_type::eleadd)) {
        compute_elementwise_ref_direct_fwd(ref_outputs.at(0).data(),
                ref_ele_adds.at(0).data(),
                {N, weight_dims_block.at(0).at(0), P, Q});
    }

    for (auto i = 1; i < nblocks; ++i) {
        H = P;
        W = Q;
        std::tie(P, Q) = p_q.at(i);
        compute_ref_direct_fwd(N, 1, weight_dims_block.at(i).at(0),
                weight_dims_block.at(i).at(1), H, W, P, Q,
                weight_dims_block.at(i).at(2), weight_dims_block.at(i).at(3),
                stride_block.at(i), stride_block.at(i), padding_block.at(i),
                padding_block.at(i), ref_outputs.at(i - 1).data(),
                ref_weights.at(i).data(), ref_biases.at(i).data(),
                ref_outputs.at(i).data(),
                bias_block.at(i) ? dir_t::FWD_B : FWD_I,
                ref_bn_muls.at(i).data(), ref_bn_adds.at(i).data(),
                bn_relu_block.at(i), 1, 1, 1, 0, 1, 1, dilations_block.at(i),
                dilations_block.at(i));
        if (post_types_block.at(i).end()
                != std::find(post_types_block.at(i).begin(),
                        post_types_block.at(i).end(), postop_type::eleadd)) {
            compute_elementwise_ref_direct_fwd(ref_outputs.at(i).data(),
                    ref_ele_adds.at(i).data(),
                    {N, weight_dims_block.at(i).at(0), P, Q});
        }
    }

    std::tie(P, Q) = p_q.at(nblocks - 1);
    sc_output = sc_args.back().copy();
    ref_output = std::move(ref_outputs.back());
}

void compute_conv_block(gc::sc_graph_t &g, std::vector<gc::sc_op_ptr> &args,
        const sc_dims &input_dims,
        const std::vector<sc_dims> &weight_dims_block,
        const sc_dims &stride_block, const sc_dims &padding_block,
        const std::vector<std::vector<gc::postop_type>> &post_types_block,
        const std::vector<gc::ops::conv_fwd_config_t> &cfg_ptr,
        test_buffer<float> &sc_output, test_buffer<float> &ref_output,
        bool rand_non_negative = false) {
    compute_conv_block(g, args, input_dims, weight_dims_block, stride_block,
            sc_dims(stride_block.size(), 1), padding_block, post_types_block,
            cfg_ptr, sc_output, ref_output, rand_non_negative);
}
#endif
