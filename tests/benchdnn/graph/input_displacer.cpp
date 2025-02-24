/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include <random>

#include "dnnl_common.hpp"
#include "input_displacer.hpp"
#include "ref_partition.hpp"

namespace graph {

partition_data_displacer_t::partition_data_displacer_t(
        const deserialized_graph_t &dg, const dnnl::graph::partition &par)
    : dg_(&dg) {
    const auto &op_ids = par.get_ops();
    op_ids_set_ = std::unordered_set<size_t>(op_ids.begin(), op_ids.end());

    static const std::unordered_set<std::string> main_op_kind {"Convolution",
            "ConvTranspose", "AvgPool", "MaxPool", "MatMul", "Add", "Divide",
            "Maximum", "Minimum", "Multiply", "Substract", "Select"};

    static const std::unordered_set<std::string> go_through_op_kind {
            "StaticTranspose", "StaticReshape", "TypeCast", "Quantize",
            "Dequantize"};

    static const std::unordered_set<std::string> f8_main_op_kind {
            "MatMul", "Convolution"};

    // The logic below relies on the assumption that deserialized_graph_t is
    // sorted in the chronological order.
    for (const auto &aop : dg_->ops_) {
        // Skip the check if op is not in the partition.
        if (op_ids_set_.find(aop.id_) == op_ids_set_.end()) continue;

        // Here is how quantize filling work
        //
        // partition input (lt)
        // |
        // [go through op]*
        // |
        // x<- quantize filling on this tensor (dq_lt)
        // |
        // dequantize <- The first dq met
        // |
        // [go through op except dq]*
        // |
        // main op (applied for all inputs the op has)

        if (main_op_kind.find(aop.kind_) == main_op_kind.end()) continue;

        // search along the branch for each input of main op
        for (size_t i = 0; i < aop.in_lts_.size(); i++) {
            // Traversing over a chain of allowed ops from the bottom to the
            // top searching for a first dequantize op in the chain.
            // Note: traversing can't be done on non-const references as
            // they will replace the starting point, but const references
            // can't be done because of assignment. So, pointers only.
            auto *parent_op = &aop;
            for (auto *lt = &aop.in_lts_[i]; true;
                    lt = &parent_op->in_lts_[0]) {
                parent_op = &dg_->get_op_by_out_lt(lt->id_);
                if (parent_op->empty()) {
                    if (aop.kind_ == "Divide" || aop.kind_ == "Multiply") {
                        // There's a special case for Divide, when second (user)
                        // input should be displaced with power-of-2 values.
                        quantize_displace_.emplace(lt->id_,
                                std::make_tuple(
                                        aop, i, *lt, filling_type_t::pow2));
                    }
                    break;
                }

                if (parent_op->kind_ == "Dequantize"
                        || parent_op->kind_ == "DynamicDequantize") {
                    // Dequantize is accepted when it doesn't have any
                    // predecessors in the partition (though it may have it in
                    // the graph).
                    const auto &parent_op_in_lt = parent_op->in_lts_[0];
                    const auto &prev_parent_op
                            = dg_->get_op_by_out_lt(parent_op_in_lt.id_);
                    if (prev_parent_op.empty()
                            || op_ids_set_.find(prev_parent_op.id_)
                                    == op_ids_set_.end()) {

                        // Skip input displacement for unusupported f8 ops.
                        const auto &lt_dt = parent_op_in_lt.get_data_type();
                        if ((lt_dt == logical_tensor::data_type::f8_e5m2
                                    || lt_dt
                                            == logical_tensor::data_type::
                                                    f8_e4m3)
                                && (f8_main_op_kind.find(aop.kind_)
                                        == f8_main_op_kind.end()))
                            break;

                        quantize_displace_.emplace(parent_op_in_lt.id_,
                                std::make_tuple(aop, i, parent_op_in_lt,
                                        filling_type_t::quantization));
                        break;
                    }
                }
                if (parent_op->kind_ == "StaticReshape") {
                    // StaticReshape is accepted when the pattern is
                    // "StaticReshape + Matmul" and it doesn't have any
                    // predecessors in the partition
                    const auto &parent_op_in_lt = parent_op->in_lts_[0];
                    const auto &prev_parent_op
                            = dg_->get_op_by_out_lt(parent_op_in_lt.id_);
                    if (prev_parent_op.empty()
                            || op_ids_set_.find(prev_parent_op.id_)
                                    == op_ids_set_.end()) {
                        if (aop.kind_ == "MatMul") {
                            quantize_displace_.emplace(parent_op_in_lt.id_,
                                    std::make_tuple(aop, i, parent_op_in_lt,
                                            filling_type_t::quantization));
                        }
                        break;
                    }
                }
                // Continue only on allowed ops.
                if (go_through_op_kind.find(parent_op->kind_)
                        == go_through_op_kind.end()) {
                    break;
                }
            }
        }

        // Alternatively, looking for Add->SoftMax chain, which represents
        // explicit SDPA mask, and should be filled with upper-corner with -inf:
        // 0 -inf -inf -inf
        // 0    0 -inf -inf
        // 0    0    0 -inf
        // 0    0    0    0
        // This is done to avoid taking future tokens into account by
        // influencing SoftMax input values.
        while (aop.kind_ == "Add" || aop.kind_ == "Select") {
            auto *aop_out_lt = &aop.out_lts_[0];
            auto *child_op = &dg_->get_op_by_in_lt(aop_out_lt->id_);
            if (child_op->kind_ != "SoftMax") break;

            // Softmax must be a part of same partition as the mask. This is to
            // avoid cases, where mask is the last op in the partition, from
            // being modified.
            if (op_ids_set_.find(child_op->id_) == op_ids_set_.end()) break;

            // Search for an input lt without a parent, this is the one to
            // modify for both explicit and implicit masks.
            const deserialized_lt *causal_mask_lt = nullptr;
            size_t offset = SIZE_MAX;
            size_t qk_data_offset = SIZE_MAX;
            // Select condition having a parent or not is the only reliable
            // difference between explicit and implicit causal mask.
            bool select_cond_has_parent = false;
            // Need to iterate over all inputs to handle padding mask expressed
            // through Select op.
            for (size_t i = 0; i < aop.in_lts_.size(); i++) {
                auto *aop_in_lt = &aop.in_lts_[i];
                auto *parent_op = &dg_->get_op_by_out_lt(aop_in_lt->id_);
                if (!parent_op->empty()) {
                    if (aop_in_lt->get_data_type()
                            != logical_tensor::data_type::boolean) {
                        // This is the qk_data, need to know its offset to
                        // properly fill condition for padding mask.
                        qk_data_offset = i;
                    } else {
                        // This means it's implicit causal mask.
                        select_cond_has_parent = true;
                    }
                    continue;
                }

                // Explicit padding mask expressed through the Select op would
                // have two user inputs: condition, hinting where padding
                // occurred and a special value (-inf) to use. In such scenario,
                // unlike for implicit causal mask, it's required to update the
                // condition to always take qk values instead of a special one.
                //
                // Checking for data type to make sure that in case of two user
                // inputs, the condition one will be updated. For implicit
                // causal mask, the condition would have a parent and a check
                // for `causal_mask_lt` being non-empty will fail.
                if (causal_mask_lt
                        && aop_in_lt->get_data_type()
                                != logical_tensor::data_type::boolean)
                    continue;

                causal_mask_lt = aop_in_lt;
                offset = i;
            }
            // No suitable tensor/subgraph for a mask displacement.
            if (!causal_mask_lt) break;

            filling_type_t filling_type = filling_type_t::undef;
            if (aop.kind_ == "Add") {
                const auto ndims = causal_mask_lt->shape_.size();
                if (ndims < 2) {
                    BENCHDNN_PRINT(7, "%s\n",
                            "[DISPLACE]: Causal mask ndims is less than 2");
                    break;
                }

                const auto M = causal_mask_lt->shape_[ndims - 2];
                if (M == 1) {
                    // This is a padding mask case, when padded tokens should
                    // be removed from the final computations. In case of
                    // benchdnn, there's no such thing as padding as all tokens
                    // are computed. To avoid numerical instabilities, a zero
                    // mask can be applied without compromising validation
                    // capabilities.
                    filling_type = filling_type_t::zero;
                } else {
                    // This is a look-ahead (or causal) mask case, when future
                    // tokens (row < col) are set to infinity to remove all
                    // connections of current tokens to unissued ones.
                    filling_type = filling_type_t::causal_mask;
                }
            } else if (aop.kind_ == "Select") {
                if (select_cond_has_parent) {
                    // Implicit causal mask case.
                    filling_type = filling_type_t::minus_infinity;
                } else {
                    // Padding mask.
                    assert(qk_data_offset == 1 || qk_data_offset == 2);
                    // Fill condition depending on qk values tensor to use only
                    // its values, which is equivalent of not using a mask.
                    if (qk_data_offset == 1) {
                        filling_type = filling_type_t::one;
                    } else if (qk_data_offset == 2) {
                        filling_type = filling_type_t::zero;
                    }
                }
            }

            if (filling_type == filling_type_t::undef) {
                BENCHDNN_PRINT(
                        7, "%s\n", "[DISPLACE]: Filling type was not set");
                break;
            }

            quantize_displace_.emplace(causal_mask_lt->id_,
                    std::make_tuple(
                            aop, offset, *causal_mask_lt, filling_type));
            break;
        }
    }
}

int partition_data_displacer_t::displace_input_data(
        size_t lt_id, dnn_mem_t &mem, res_t *res) {
    if (!dg_) {
        res->state = FAILED;
        return FAIL;
    }

    if (quantize_displace_.find(lt_id) == quantize_displace_.end()) {
        // no need to displace the data of this tensor
        return OK;
    }
    const displace_t &displace = quantize_displace_.at(lt_id);

    const auto &main_op = ::std::get<0>(displace);
    auto main_op_offset = ::std::get<1>(displace);
    auto tensor = ::std::get<2>(displace);
    const auto filling_type = ::std::get<3>(displace);

    auto opkind = opstr2kind(main_op.kind_);
    int main_op_arg = get_prim_arg_name_from_graph_op_input_offset(
            opkind, main_op_offset);

    BENCHDNN_PRINT(3, "[DISPLACE]: Op:%s; Arg:%s;\n", main_op.kind_.c_str(),
            data_kind2str(exec_arg2data_kind(main_op_arg)));

    dnn_mem_t mem_replace;
    if (filling_type == filling_type_t::quantization) {
        SAFE(gen_quantize_filling(
                     main_op, main_op_arg, mem_replace, tensor.data_type_, res),
                WARN);
    } else if (filling_type == filling_type_t::pow2) {
        // This custom displace serves a shrinking the output purpose. Values
        // are picked in the way to have more chances of ending with exact lower
        // data type representative in the output, which in turn decreases the
        // number of points with non-zero absolute diff.

        // Division has values > 1.f to reduce final values.
        static const std::vector<float> pow2_div_vals {2.f, 4.f, 8.f};
        // Multiplication has values <= 1.f to reduce final values.
        static const std::vector<float> pow2_mul_vals {0.25f, 0.5f, 1.f};
        // Guard set.
        static const std::vector<float> dummy {};

        const bool is_div = main_op.kind_ == "Divide";
        const bool is_mul = main_op.kind_ == "Multiply";
        const auto &user_set
                = is_div ? pow2_div_vals : (is_mul ? pow2_mul_vals : dummy);
        fill_cfg_t fill_cfg(user_set, "Mul/Div displacer");
        SAFE(gen_fixed_set_filling(mem_replace, mem.md_, fill_cfg, res), WARN);
    } else if (filling_type == filling_type_t::causal_mask) {
        SAFE(gen_causal_mask_filling(mem_replace, mem.md_, res), WARN);
    } else if (filling_type == filling_type_t::minus_infinity) {
        static const std::vector<float> user_set {-INFINITY};
        fill_cfg_t fill_cfg(user_set, "Implicit_causal_mask");
        SAFE(gen_fixed_set_filling(mem_replace, mem.md_, fill_cfg, res), WARN);
    } else if (filling_type == filling_type_t::zero) {
        static const std::vector<float> user_set {0.f};
        fill_cfg_t fill_cfg(user_set, "Explicit_padding_mask");
        SAFE(gen_fixed_set_filling(mem_replace, mem.md_, fill_cfg, res), WARN);
    } else if (filling_type == filling_type_t::one) {
        static const std::vector<float> user_set {1.f};
        fill_cfg_t fill_cfg(user_set, "Explicit_padding_mask");
        SAFE(gen_fixed_set_filling(mem_replace, mem.md_, fill_cfg, res), WARN);
    } else {
        assert(!"unexpected filling type");
    }

    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    // do the reverse job
    auto *parent_op = &dg_->get_op_by_out_lt(tensor.id_);
    bool backward_path_launched = false;
    while (filling_type == filling_type_t::quantization && !parent_op->empty()
            && op_ids_set_.find(parent_op->id_) != op_ids_set_.end()) {
        backward_path_launched = true;
        // generate the reverse op based on OP kind
        // make a copy of deserialized_op_t to avoid impact on graph execution
        // Currently, we support the following OPs' reverse execution:
        // All of the execution need to swap the input lt and output lt first

        // 1. StaticTranspose: re-permute the 'order' attr to get an inversed effect
        // 2. TypeCast: Do nothing special because the type is already swapped
        // 3. StaticReshape: Do nothing special because the shape is already swapped
        // 4. Quantize: change opkind to Dequantize and keep scales and zps
        // 5. Dequantize: change opkind to Quantize and keep scales and zps

        auto op = dg_->get_op_by_out_lt(tensor.id_);
        BENCHDNN_PRINT(
                3, "[DISPLACE]: Backward path for Op:%s;\n", op.kind_.c_str());

        ::std::swap(op.in_lts_, op.out_lts_);

        auto opkind = opstr2kind(op.kind_);

        switch (opkind) {
            case ::graph::op::kind::Quantize: op.kind_ = "Dequantize"; break;
            case ::graph::op::kind::Dequantize: op.kind_ = "Quantize"; break;
            case ::graph::op::kind::StaticTranspose: {
                ::std::vector<int64_t> order;
                op.get_attr_s64_vector(order, "order");
                size_t ndims = order.size();
                ::std::vector<int64_t> new_order(ndims, 0);
                for (size_t i = 0; i < ndims; i++) {
                    new_order[(order[i] + ndims) % ndims] = i;
                }
                op.attrs_["order"].s64_vector_ = new_order;
                break;
            }
            case ::graph::op::kind::TypeCast:
            case ::graph::op::kind::StaticReshape: break;
            default:
                assert(!"not support opkind for reverse execution");
                return FAIL;
        }

        // execute the reverse op

        std::unordered_set<size_t> empty_set;
        res_t res {};

        ref_primitive_t ref_prim(op);
        ref_prim.init_prb(&res);
        SAFE_V(ref_prim.init_prim(
                get_cpu_engine(), &res, /* force_override = */ true));

        ref_prim.init_memory_args(get_cpu_engine());
        SAFE_V(ref_prim.init_ref_memory_args(get_cpu_engine(), &res));

        const auto &src_mem = ref_prim.get_arg(DNNL_ARG_SRC);
        bool mds_are_equal
                = dnnl_memory_desc_equal(mem_replace.md_, src_mem.md_) == 1;
        SAFE(mds_are_equal ? OK : FAIL, WARN);

        // Always use the md generated by current reversed op. E.g., a matmul op
        // will unsqeeze 1 to fit the dimension so the md generated by matmul
        // prb_t will not be the same as defined in graph.
        dnnl_memory_desc_destroy(mem_replace.md_);
        dnnl_memory_desc_clone(&mem_replace.md_, src_mem.md_);
        ref_prim.replace_arg(DNNL_ARG_SRC, mem_replace);
        SAFE_V(ref_prim.execute_prim(&res));

        mem_replace = ::std::move(
                const_cast<dnn_mem_t &>(ref_prim.get_arg(DNNL_ARG_DST)));
        tensor = op.out_lts_[0];
        parent_op = &dg_->get_op_by_out_lt(tensor.id_);
    }

    if (backward_path_launched) {
        BENCHDNN_PRINT(3, "%s\n", "[DISPLACE]: Backward path ended.");
    }

    bool mds_are_equal = dnnl_memory_desc_equal(mem_replace.md_, mem.md_) == 1;
    bool mds_are_int8 = is_integral_dt(mem_replace.dt())
            && is_integral_dt(mem.dt()) && mem_replace.sizeof_dt() == 1
            && mem.sizeof_dt() == 1;
    bool is_grouped_conv = false;
    if (main_op.kind_ == "Convolution" || main_op.kind_ == "ConvTranspose") {
        int64_t groups;
        main_op.get_attr_s64(groups, "groups");
        is_grouped_conv = groups > 1;
    }

    bool is_reshaped_dims = mem_replace.nelems() == mem.nelems()
            && mem_replace.ndims() != mem.ndims();

    bool mds_ok = IMPLICATION(!mds_are_equal,
            mds_are_int8 || is_grouped_conv || is_reshaped_dims);
    SAFE(mds_ok ? OK : FAIL, WARN);

    dnnl_memory_desc_t md = mem.md_;
    if (is_reshaped_dims) {
        DNN_SAFE_V(dnnl_memory_desc_create_with_strides(
                &md, mem.ndims(), mem.dims(), mem_replace.dt(), mem.strides()));
    }
    dnnl_memory_desc_destroy(mem_replace.md_);
    dnnl_memory_desc_clone(&mem_replace.md_, md);
    SAFE(mem.reorder(mem_replace), WARN);

    if (is_reshaped_dims) dnnl_memory_desc_destroy(md);
    return OK;
}

int partition_data_displacer_t::gen_quantize_filling(
        const ::graph::deserialized_op_t &main_op, int arg, dnn_mem_t &mem,
        const ::std::string &dt, res_t *res) {
    // clone a deserialized op object and modify to specified data type
    ::graph::deserialized_op_t op = main_op;
    auto driver = opkind2driver(opstr2kind(op.kind_));
    bool is_f8_quantization = (dt == "f8_e5m2" || dt == "f8_e4m3");

    op.in_lts_[0].data_type_ = dt;
    if (op.in_lts_.size() > 1) {
        op.in_lts_[1].data_type_ = dt;
        // Matmul/Conv/Deconv have limited support for quantized configurations.
        if (op.kind_ == "MatMul" || op.kind_ == "Convolution"
                || op.kind_ == "ConvTranspose") {
            if (dt == "u8") {
                // None of them supports u8u8, replace with u8s8.
                op.in_lts_[1].data_type_ = "s8";
            } else if (dt == "s4" || dt == "u4") {
                // None of them supports x4x4, replace with f32x4f32 or
                // xf16x4xf16.
                op.in_lts_[0].data_type_ = op.out_lts_[0].data_type_;
            }
        }
    }
    if (driver == dnnl_driver_t::pool || driver == dnnl_driver_t::binary
            || is_f8_quantization) {
        // pool does not support x8f32 on cpu, and binary does not support
        // x8x8bf16 on gpu, hence replace output with x8.
        // f8 data types needs setting output data type to f8
        op.out_lts_[0].data_type_ = dt;
    } else if (op.out_lts_[0].data_type_ != "bf16") {
        if (op.in_lts_.size() > 1 && op.in_lts_[1].data_type_ == "s8") {
            // Use u8 as output data type for two-input operations to avoid
            // data overflow due to the specific driver logic.
            op.out_lts_[0].data_type_ = "u8";
        } else {
            // Use f32 as output data type since not all primitives support
            // different data types for input and output.
            op.out_lts_[0].data_type_ = "f32";
        }
    }

    ::std::unordered_set<size_t> empty_set;
    const auto &eng = get_test_engine();
    ref_primitive_t ref_prim(op);
    ref_prim.init_prb(res);
    if (res->state == INVALID_ARGUMENTS) return FAIL;
    SAFE_V(ref_prim.init_prim(eng, res, /* force_override = */ true));
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;
    ref_prim.init_memory_args(eng);
    SAFE_V(ref_prim.init_ref_memory_args(eng, res));
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    mem = ::std::move(const_cast<dnn_mem_t &>(ref_prim.get_arg(arg)));

    return OK;
}

int partition_data_displacer_t::gen_fixed_set_filling(dnn_mem_t &mem,
        const_dnnl_memory_desc_t md, const fill_cfg_t &fill_cfg,
        res_t *res) const {

    dnn_mem_t m(md, get_test_engine());
    const int64_t nelems = m.nelems();

    BENCHDNN_PRINT(6, "%s\n", fill_cfg.print_verbose().c_str());

    const auto &vals = fill_cfg.predefined_set_;
    const int n_vals = static_cast<int>(vals.size());

    /* Do fixed partitioning to have same filling for any number of threads */
    static constexpr int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);
    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(idx_start + 1);
        int_seed.discard(1);

        std::uniform_int_distribution<> gen(0, n_vals - 1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            const float val = vals[gen(int_seed)];
            m.set_elem(idx, val);
        }
    });

    mem = std::move(m);
    return OK;
}

int partition_data_displacer_t::gen_causal_mask_filling(
        dnn_mem_t &mem, const_dnnl_memory_desc_t md, res_t *res) const {

    dnn_mem_t tmp_mem(md, get_test_engine());

    const int ndims = query_md_ndims(md);
    assert(ndims >= 2); // This was checked at displacer initialization.
    const auto &dims = query_md_dims(md);
    const int64_t batch = std::accumulate(dims, dims + ndims - 2, (dnnl_dim_t)1,
            std::multiplies<dnnl_dim_t>());
    const int64_t M = dims[ndims - 2];
    const int64_t N = dims[ndims - 1];

    benchdnn_parallel_nd(batch, M, N, [&](int64_t b, int64_t m, int64_t n) {
        int64_t idx = b * M * N + m * N + n;
        float val = m >= n ? 0.f : -INFINITY;
        tmp_mem.set_elem(idx, val);
    });

    mem = std::move(tmp_mem);
    return OK;
}

} // namespace graph
