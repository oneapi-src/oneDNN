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
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include "compiler/ir/builder.hpp"
#include "compiler/ir/graph/fusible_op_utils.hpp"
#include "compiler/ir/graph/fusion_anchor.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/sc_data_format.hpp"
#include "pooling.hpp"
#include "util/bf16.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
static inline any_map_t add_pl_type(const any_map_t &attrs, int pl_type) {
    auto ret = attrs;
    ret[pooling_attr_key::pooling_type] = pl_type;
    return ret;
}

static inline any_map_t add_pl_type_and_in_shape(
        const any_map_t &attrs, int pl_type, const sc_dims &input_shape) {
    auto ret = attrs;
    ret[pooling_attr_key::pooling_type] = pl_type;
    ret[pooling_attr_key::input_shape] = input_shape;
    return ret;
}

// return the vector storing the indices of pooling axis in format
// for example format{0,2,3,1,1} returns [1, 2] when not channel_last
// and [1, 3, 4] when channel_last
static std::vector<int> get_real_pooling_axis_form_tensor(
        const graph_tensor_ptr &t, bool channel_last) {
    auto ndims = t->details_.get_plain_dims().size();
    std::vector<int> required_axis(ndims - 2);
    int kernel_axis_begin = channel_last ? 1 : 2;
    std::iota(required_axis.begin(), required_axis.end(), kernel_axis_begin);
    auto real_required_axis = transform_axis_plain2blocking(t, required_axis);

    return real_required_axis;
}

std::vector<int> pooling_op_t::get_real_pooling_axis() const {
    return get_real_pooling_axis_form_tensor(get_inputs()[0], channel_last_);
}

// will change format code from ncx/nxc -> ncx to help compute_block
static std::vector<int> get_ncx_formatcode_vector_form_tensor(
        const graph_tensor_ptr &t, bool channal_last) {
    std::vector<int> out;
    auto &in_fmt = t->details_.get_format();
    int channel_axis = t->details_.get_plain_dims().size() - 1, bs_axis = 0;
    for (int i = 0; i <= sc_data_format_kind_t::MAX_DIMS; i++) {
        int n = in_fmt.format_code_.get(i);
        if (n == sc_data_format_kind_t::UNDEF_DIM) break;
        if (channal_last) {
            if (n == bs_axis)
                out.push_back(n);
            else if (n == channel_axis)
                out.push_back(1);
            else
                out.push_back(n + 1);
        } else {
            out.push_back(n);
        }
    }

    return out;
}

std::vector<int> pooling_op_t::get_channel_axis() const {
    std::vector<int> ret;
    auto ncx_fmt_vec = get_ncx_formatcode_vector_form_tensor(
            get_inputs()[0], channel_last_);
    for (int i = 0; i < static_cast<int>(ncx_fmt_vec.size()); i++) {
        if (ncx_fmt_vec[i] == 1) ret.emplace_back(i);
    }
    return ret;
}

static void check_format(sc_data_format_t fmt, bool channel_last) {
    int n_kernel_dims = fmt.format_code_.norig_dims() - 2;
    std::unordered_map<int, std::vector<int>> blocked_axis
            = fmt.get_blocked_axis();
    std::vector<std::string> pos_name = {"h", "w"};
    if (n_kernel_dims == 3) { pos_name.insert(pos_name.begin(), "d"); }
    uint64_t kernel_axis_begin = channel_last ? 1 : 2;
    for (int i = 0; i < n_kernel_dims; i++) {
        COMPILE_ASSERT(
                blocked_axis.find(kernel_axis_begin + i) == blocked_axis.end(),
                pos_name[i]
                        + " axis should not be blocked in fusible pooling!");
    }
}

slice_range_list infer_pool_slice_ranges(const graph_tensor_ptr &infered_tensor,
        const slice_range_list &in_range_list, bool channel_last) {
    auto real_pooling_axis
            = get_real_pooling_axis_form_tensor(infered_tensor, channel_last);
    auto &o_blocked_dims = infered_tensor->details_.get_blocking_dims();
    slice_range_list o_ranges_list;
    for (auto &range_list : in_range_list) {
        slice_range out_range;
        for (unsigned i = 0; i < range_list.size(); i++) {
            if (!(std::find(
                          real_pooling_axis.begin(), real_pooling_axis.end(), i)
                        == real_pooling_axis.end())) {
                out_range.emplace_back(0, uint64_t(o_blocked_dims[i]));
                continue;
            }
            out_range.emplace_back(range_list[i]);
        }
        o_ranges_list.emplace_back(out_range);
    }
    return o_ranges_list;
};

static void check_and_set_pads_begin_and_pads_end(any_map_t &attrs,
        const sc_dims &input_plain_dims, sc_dims &pads_begin, sc_dims &pads_end,
        bool channel_last) {
    std::string auto_pad = attrs.get_or_else<std::string>(
            pooling_attr_key::auto_pad, auto_pad_options::none);
    uint64_t n_kernel_dims = input_plain_dims.size() - 2;
    uint64_t kernel_axis_begin = channel_last ? 1 : 2;
    sc_dims kernel = attrs.get<sc_dims>(pooling_attr_key::kernel);
    sc_dims stride = attrs.get<sc_dims>(pooling_attr_key::strides);
    // compute pads_begin and pads_end
    COMPILE_ASSERT(auto_pad == auto_pad_options::none
                    || auto_pad == auto_pad_options::same_upper
                    || auto_pad == auto_pad_options::same_lower
                    || auto_pad == auto_pad_options::valid,
            "auto_pad type should be none/same_upper(same_lower)/valid , but "
            "got" << auto_pad);

    if (auto_pad == auto_pad_options::none) {
        if (attrs.has_key(pooling_attr_key::paddings)) {
            auto &padding = attrs.get<sc_dims>(pooling_attr_key::paddings);

            if (padding.size() == 1) {
                pads_begin = std::vector<int64_t>(n_kernel_dims, padding[0]);
                pads_end = std::vector<int64_t>(n_kernel_dims, padding[0]);
            } else if (padding.size() == n_kernel_dims) {
                pads_begin = padding;
                pads_end = padding;
            } else {
                COMPILE_ASSERT(false,
                        "padding should have " << n_kernel_dims
                                               << " or 1 n-dims, but got "
                                               << padding.size());
            }
        } else {
            COMPILE_ASSERT(attrs.has_key(pooling_attr_key::pads_begin)
                            && attrs.has_key(pooling_attr_key::pads_end),
                    "max/avg pooling op should have pads_begin and pads_end");
            pads_begin = attrs.get<sc_dims>(pooling_attr_key::pads_begin);
            pads_end = attrs.get<sc_dims>(pooling_attr_key::pads_end);
            COMPILE_ASSERT(pads_begin.size() == n_kernel_dims,
                    "pads_begin should have " << n_kernel_dims
                                              << "n-dims, but got "
                                              << pads_begin.size());
            COMPILE_ASSERT(pads_end.size() == n_kernel_dims,
                    "pads_end should have " << n_kernel_dims
                                            << "n-dims, but got "
                                            << pads_end.size());
        }
    } else if (auto_pad == auto_pad_options::valid) {
        pads_begin = std::vector<int64_t>(n_kernel_dims, 0);
        pads_end = std::vector<int64_t>(n_kernel_dims, 0);
    } else if (auto_pad == auto_pad_options::same_upper
            || auto_pad == auto_pad_options::same_lower) {
        pads_begin = std::vector<int64_t>(n_kernel_dims);
        pads_end = std::vector<int64_t>(n_kernel_dims);
        for (unsigned int i = 0; i < n_kernel_dims; i++) {
            auto in_dim = input_plain_dims[kernel_axis_begin + i];
            auto out_dim = (in_dim + stride[i] - 1) / stride[i];
            auto total_pad = (out_dim - 1) * stride[i] + kernel[i] - in_dim;
            if (total_pad < 0) total_pad = 0;
            auto half_pad_small = total_pad / 2;
            auto half_pad_big = total_pad - half_pad_small;
            if (auto_pad == auto_pad_options::same_upper) {
                pads_begin[i] = half_pad_small;
                pads_end[i] = half_pad_big;
            } else if (auto_pad == auto_pad_options::same_lower) {
                pads_begin[i] = half_pad_big;
                pads_end[i] = half_pad_small;
            }
        }
    }
    // update attrs
    attrs[pooling_attr_key::pads_begin] = pads_begin;
    attrs[pooling_attr_key::pads_end] = pads_end;
    if (attrs.has_key(pooling_attr_key::paddings)) {
        attrs.remove(pooling_attr_key::paddings);
    }
}

static void check_and_set_kernel_strides_and_pooling_type(any_map_t &attrs,
        uint64_t n_kernel_dims, sc_dims &kernel, sc_dims &stride,
        pooling_type_t &pooling_type) {
    COMPILE_ASSERT(attrs.has_key(pooling_attr_key::kernel)
                    && attrs.has_key(pooling_attr_key::strides)
                    && attrs.has_key(pooling_attr_key::pooling_type),
            "max/avg pooling op takes 3 attributes, kernel strides and "
            "pooling_type")
    stride = attrs.get<sc_dims>(pooling_attr_key::strides);
    if (stride.size() == 1) {
        stride = sc_dims(n_kernel_dims, stride[0]);
    } else {
        COMPILE_ASSERT(stride.size() == n_kernel_dims,
                "strides should have " << n_kernel_dims << "n-dims, but got "
                                       << stride.size());
    }
    kernel = attrs.get<sc_dims>(pooling_attr_key::kernel);
    COMPILE_ASSERT(kernel.size() == n_kernel_dims,
            "kernel should have " << n_kernel_dims << "n-dims, but got "
                                  << kernel.size());

    pooling_type
            = pooling_type_t(attrs.get<int>(pooling_attr_key::pooling_type));
}

static bool check_data_format_channel_last(const any_map_t &attrs) {
    std::string data_format = attrs.get_or_else<std::string>(
            pooling_attr_key::data_format, data_format_options::NXC);
    COMPILE_ASSERT(data_format == data_format_options::NXC
                    || data_format == data_format_options::NCX,
            "Error data_format:" + data_format);
    return data_format == data_format_options::NXC;
}

pooling_op_t::pooling_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    // set inputs and attrs_
    COMPILE_ASSERT(ins.size() == 1, "Expecting 1 input for pooling_op_t");
    uint64_t n_plain_dims = ins[0]->details_.get_plain_dims().size();
    COMPILE_ASSERT(n_plain_dims == 4 || n_plain_dims == 5,
            "input should have 4 or 5 n-dims,but got " << n_plain_dims);
    info_.inputs_ = ins;
    attrs_ = attrs;

    // set kernel_ and  stride_
    uint64_t n_kernel_dims = n_plain_dims - 2;
    check_and_set_kernel_strides_and_pooling_type(
            attrs_, n_kernel_dims, kernel_, stride_, pooling_type_);

    channel_last_ = check_data_format_channel_last(attrs_);
    // set pads_begin_ and pads_begin_
    check_and_set_pads_begin_and_pads_end(attrs_,
            ins[0]->details_.get_plain_dims(), pads_begin_, pads_end_,
            channel_last_);

    // set outputs
    std::string rounding_type = attrs.get_or_else<std::string>(
            pooling_attr_key::rounding_type, rounding_type_options::floor);
    COMPILE_ASSERT(rounding_type == rounding_type_options::floor
                    || rounding_type == rounding_type_options::ceil,
            "rounding type should be floor or ceil, but got" << rounding_type);
    bool rounding_floor = rounding_type == rounding_type_options::floor;
    sc_dims output_dims = _calculate_output_dims(rounding_floor, channel_last_);
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                info_.inputs_[0]->details_.get_format(), output_dims,
                ins[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "pooling expect 1 output");
        COMPILE_ASSERT(outs[0]->details_.get_plain_dims() == output_dims,
                "Bad output shape for pooling")
        info_.outputs_ = outs;
    }
}

pooling_op_t::pooling_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs,
        const pooling_type_t &pl_type, const any_map_t &attrs)
    : pooling_op_t(ins, outs, add_pl_type(attrs, static_cast<int>(pl_type))) {}

void pooling_op_t::prepare_fusion_data(fdata_map &fdmap) {
    fdmap.get(info_.inputs_[0]).use_count_++;
}

void pooling_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);

    // judge whether input dims full on w and h (and d)
    auto real_required_axis = get_real_pooling_axis_form_tensor(
            info_.inputs_[0], channel_last_);
    auto &in_blocked_dims = info_.inputs_[0]->details_.get_blocking_dims();
    for (auto &range_list : known_ranges_map[0]) {
        if (!slice_full_on_axis(
                    in_blocked_dims, range_list, real_required_axis)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }

    // compute output slice range list
    auto output_ranges_list = infer_pool_slice_ranges(
            info_.outputs_[0], known_ranges_map[0], channel_last_);

    // return final result
    fsmap.get(info_.outputs_[0]) = output_ranges_list;
}

void pooling_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    if (fsmap.get(get_inputs()[0]).empty()) {
        slice_range_list known_ranges_list = fsmap.get(get_outputs()[0]);
        slice_range_list input_slice_list = infer_pool_slice_ranges(
                info_.inputs_[0], known_ranges_list, channel_last_);
        if (input_slice_list.size() != 1) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
        fsmap.get(get_inputs()[0]) = input_slice_list;
        if (auto inp_op
                = info_.inputs_[0]->producer_owner_->dyn_cast<fusible_op_t>()) {
            inp_op->pre_slice_ranges(fsmap, stat_map);
        }
    }
}

static void compute_block_pooling(
        const std::vector<const tensor_slice *> &inputs,
        const tensor_slice &dst, pooling_type_t pooling_typ, sc_dims kernel,
        sc_dims stride, sc_dims pads_begin,
        const std::vector<int> &in_fmt_vector, const vectorized_info_t &vx_info,
        sc_data_type_t in_dtype, sc_data_type_t out_dtype, any_map_t &attrs,
        const graph_tensor_ptr &output_tensor = nullptr, size_t wkld = 0UL) {
    /*** The final IR may look like below:
     * _for_(_fuseiter_i, 0, I, 1)
     *   _for_(_fuseiter_j, 0, J, 1)
     *     _for_(_fuseiter_k, 0, K, 1)
     *        _for_(_fuseiter_l, 0, L, 1)
     *          max: sum = -inf; avg: sum =0
     *          num = 0
     *          src_idx = [i,j,k * stride_h - pad_begin_h,
     *                     l * stride_l - pad_begin_l]
     *          dst_idx = [i,j,k,l]
     *          _for_(_fuseiter_s, 0, S, 1)
     *            _if(tmp_src_h>=0 && tmp_src_h< H )
     *               _for_(_fuseiter_r, 0, R, 1)
     *               tmp_src = src_idx + [0,0,s,r]
     *               _if (tmp_src_w>=0 && tmp_src_w< W)
     *                  sum += src[tmp_src]; or sum = max(sum,src[src_idx])
     *                  num++
     *               _else
     *                  sum += 0; or sum = max(sum,0)
     *            _else
     *               sum += 0; or sum = max(sum,0)
     *          dst[dst_idx] = sum/num or sum;
     ***/

    auto in_vectorized_dtype
            = sc_data_type_t(in_dtype.type_code_, vx_info.lanes);
    auto out_vectorized_dtype
            = sc_data_type_t(out_dtype.type_code_, vx_info.lanes);
    auto pool_buf_vectorized_dtype
            = sc_data_type_t(in_dtype.type_code_, vx_info.lanes);
    // nested loop vars
    std::vector<expr> iter_vars, kernel_iter_vars;

    // the indices for the output tensor
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        iter_vars.emplace_back(range_from_outer_loop(dst.get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_iter") + fusion_create_idx()));
    }

    // the indices dor the kernel inner loop
    for (unsigned i = 0; i < kernel.size(); i++) {
        kernel_iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_kernel") + fusion_create_idx()));
    }

    // input indices
    std::vector<std::vector<expr>> src_indices(inputs.size());
    auto &src_idx = src_indices.at(0);
    std::vector<expr> conds(kernel.size());
    for (unsigned i = 0; i < inputs[0]->nslice_dims(); i++) {
        if (in_fmt_vector[i] < 2) {
            src_idx.emplace_back(iter_vars[i]);
        } else {
            int plan_axis = in_fmt_vector[i];
            int pads_stride_index = plan_axis - 2;
            // k * stride_h - pad_begin_h + kernel_var_h
            expr out_h_idx = dst.get_offset()[i];
            if (out_h_idx.isa<constant>()) out_h_idx = iter_vars[i];
            auto idx = int(stride[pads_stride_index]) * out_h_idx
                    - int(pads_begin[pads_stride_index])
                    + kernel_iter_vars[pads_stride_index];
            expr tmp_cond = builder::make_logic_and(
                    builder::make_cmp_ge(idx, 0),
                    builder::make_cmp_lt(idx, inputs[0]->get_shape()[i]));
            conds[pads_stride_index] = tmp_cond;
            src_idx.emplace_back(idx);
        }
    }

    expr indexed_target
            = builder::make_indexing(dst.tptr_, iter_vars, vx_info.lanes);
    expr indexed_input = builder::make_indexing(
            inputs[0]->tptr_, src_indices.at(0), vx_info.lanes);

    // builder
    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");

    // assign init value
    variant<float, int64_t> init_value;
    bool is_int = utils::is_one_of(in_dtype.type_code_, sc_data_etype::U8,
            sc_data_etype::U32, sc_data_etype::S8, sc_data_etype::S32);
    if (pooling_typ == pooling_type_t::avg) {
        init_value = 0.f;
    } else {
        COMPILE_ASSERT(
                pooling_typ == pooling_type_t::max, "wrong pooling type");
        init_value = numeric_limits_minimum(in_dtype.type_code_);
    }

    int kernel_size = 1;
    bool exclude_pad = false;
    if (pooling_typ == pooling_type_t::avg) {
        exclude_pad = attrs.get<bool>("exclude_pad");
        if (exclude_pad) {
            kernel_size = 0;
        } else {
            for (auto kn : kernel)
                kernel_size = kernel_size * kn;
        }
    }
    expr zero_constant, one_constant, kernel_size_constant,
            pooling_buf_constant;
    if (in_dtype.type_code_ == sc_data_etype::F32
            || in_dtype.type_code_ == sc_data_etype::BF16) {
        zero_constant = make_expr<constant_node>(0.f, in_vectorized_dtype);
        one_constant = make_expr<constant_node>(1.f, in_vectorized_dtype);
        kernel_size_constant = make_expr<constant_node>(
                float(kernel_size), in_vectorized_dtype);
        pooling_buf_constant = make_expr<constant_node>(
                init_value.get<float>(), in_vectorized_dtype);
    } else if (in_dtype.type_code_ == sc_data_etype::U8
            || in_dtype.type_code_ == sc_data_etype::U32
            || in_dtype.type_code_ == sc_data_etype::S8
            || in_dtype.type_code_ == sc_data_etype::S32) {
        zero_constant = make_expr<constant_node>(0.f, in_vectorized_dtype);
        one_constant = make_expr<constant_node>(1.f, in_vectorized_dtype);
        if (pooling_typ == pooling_type_t::avg) {
            pool_buf_vectorized_dtype
                    = sc_data_type_t(sc_data_etype::F32, vx_info.lanes);
            kernel_size_constant = make_expr<constant_node>(
                    float(kernel_size), pool_buf_vectorized_dtype);
            pooling_buf_constant = make_expr<constant_node>(
                    init_value.get<float>(), pool_buf_vectorized_dtype);
        } else {
            kernel_size_constant = make_expr<constant_node>(
                    int64_t(kernel_size), pool_buf_vectorized_dtype);
            pooling_buf_constant = make_expr<constant_node>(
                    uint64_t(init_value.get<int64_t>()),
                    pool_buf_vectorized_dtype);
        }
    } else {
        COMPILE_ASSERT(0, "unsupported in_dtype.");
    }

    // define local vars
    expr kernel_size_var
            = builder::make_var(pool_buf_vectorized_dtype, "kernel_size");
    expr pooling_buf_var
            = builder::make_var(pool_buf_vectorized_dtype, "pool_buf");
    stmt pooling_buf_asnode
            = make_stmt<assign_node_t>(pooling_buf_var, pooling_buf_constant);
    stmt define_pool_buf_var
            = make_stmt<define_node_t>(pooling_buf_var, linkage::local, expr());

    // build inner kernel loop
    stmt cur, body;
    for (int i = kernel_iter_vars.size() - 1; i >= 0; i--) {
        stmt else_stmt, then_stmt, additional_assign;
        if (i == int(kernel_iter_vars.size() - 1)) {
            if (pooling_typ == pooling_type_t::avg) {
                if (pool_buf_vectorized_dtype != in_vectorized_dtype) {
                    indexed_input = builder::make_cast(
                            pool_buf_vectorized_dtype, indexed_input);
                }
                then_stmt = make_stmt<assign_node_t>(pooling_buf_var,
                        builder::make_add(indexed_input, pooling_buf_var));
                then_stmt->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
                if (exclude_pad) {
                    additional_assign = make_stmt<assign_node_t>(
                            kernel_size_var,
                            builder::make_add(kernel_size_var, one_constant));
                    additional_assign->attr()
                            [op_traits::workload_computable_t::workload_number]
                            = wkld;
                    then_stmt = make_stmt<stmts_node_t>(std::vector<stmt> {
                            std::move(then_stmt), additional_assign});
                }
            } else if (pooling_typ == pooling_type_t::max) {
                then_stmt = make_stmt<assign_node_t>(pooling_buf_var,
                        builder::make_max(indexed_input, pooling_buf_var));
                then_stmt->attr()
                        [op_traits::workload_computable_t::workload_number]
                        = wkld;
            }
        } else {
            then_stmt = cur;
        }

        if (pooling_typ == pooling_type_t::max) {
            else_stmt = make_stmt<assign_node_t>(pooling_buf_var,
                    builder::make_max(zero_constant, pooling_buf_var));
            else_stmt->attr()[op_traits::workload_computable_t::workload_number]
                    = wkld;
        }

        cur = make_stmt<if_else_node_t>(conds[i], then_stmt, else_stmt);

        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(kernel_iter_vars.at(i)),
                expr(0), int(kernel[i]), expr(1), std::move(body), true,
                for_type::NORMAL);
    }

    // build outter loop to generate pooling result
    stmt target_assign;
    std::vector<fuse_anchor_map_ptr> inner_anchors;
    std::vector<stmt> inital_stmts = {define_pool_buf_var, pooling_buf_asnode};
    if (pooling_typ == pooling_type_t::avg && exclude_pad) {
        stmt define_kernel_size_var = make_stmt<define_node_t>(
                kernel_size_var, linkage::local, expr());
        stmt kernel_size_asnode = make_stmt<assign_node_t>(
                kernel_size_var, kernel_size_constant);
        inital_stmts.emplace_back(define_kernel_size_var);
        inital_stmts.emplace_back(kernel_size_asnode);
    }
    for (int i = iter_vars.size() - 1; i >= 0; i--) {
        if (i == int(iter_vars.size() - 1)) {
            if (pooling_typ == pooling_type_t::avg) {
                expr kernel_size_expr = kernel_size_constant;
                if (exclude_pad) kernel_size_expr = kernel_size_var;
                expr pooling_result = builder::make_div(pooling_buf_var,
                        builder::make_cast(
                                pool_buf_vectorized_dtype, kernel_size_expr));
                if (out_vectorized_dtype.type_code_
                        != pool_buf_vectorized_dtype.type_code_)
                    pooling_result = builder::make_cast(
                            out_vectorized_dtype, pooling_result);
                target_assign = make_stmt<assign_node_t>(
                        indexed_target, pooling_result);
            } else if (pooling_typ == pooling_type_t::max) {
                target_assign = make_stmt<assign_node_t>(
                        indexed_target, pooling_buf_var);
            }
            inital_stmts.emplace_back(std::move(cur));
            inital_stmts.emplace_back(std::move(target_assign));
            cur = make_stmt<stmts_node_t>(std::move(inital_stmts));
        }
        // Do not generate those dummy loops
        if (!iter_vars.at(i).isa<var>()) continue;

        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {cur});
        if (!body.ptr_same(cur)) add_parent_node(cur, body);

        if (output_tensor != nullptr) {
            // create output inner anchors for postop fusion
            auto anchor_stmt = make_stmt<stmts_node_t>(std::vector<stmt> {});
            body.static_as<stmts>()->seq_.emplace_back(anchor_stmt);
            add_parent_node(anchor_stmt, body);
            slice_range inner_slice = dst.get_ranges();
            for (int j = i; j >= 0; j--) {
                inner_slice[j].first = dst.get_offset()[j] + iter_vars[j];
                inner_slice[j].second = ((static_cast<int>(j) == vx_info.axis)
                                ? expr(int(vx_info.lanes))
                                : expr(1));
            }
            fslice_map fsmap;
            fsmap.get(output_tensor) = slice_range_list {inner_slice};
            inner_anchors.emplace_back(
                    std::make_shared<fuse_anchor_map_t>(anchor_stmt, fsmap));
        }

        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), 0,
                dst.get_shape()[i],
                (i == int(iter_vars.size() - 1)) ? int(vx_info.lanes) : 1, body,
                true, i == 0 ? for_type::PARALLEL : for_type::NORMAL);
        cur->attr()[stmt_attr_key::merge_loop] = true;
        add_parent_node(body, cur);
    }

    bld->emit(cur);
    attrs[op_attr_key::fusible_inner_anchors] = inner_anchors;
}
void pooling_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    // set up vx_info
    vx_info_.axis = dst[0]->get_shape().size() - 1;
    vx_info_.lanes = 1;
    // if last axis are not h or w (or d) ,lanes can be not 1
    auto last_axis = info_.inputs_[0]->details_.get_format().format_code_.get(
            inputs[0]->nbase_dims() - 1);
    const int channel_axis = channel_last_
            ? info_.inputs_[0]->details_.get_plain_dims().size() - 1
            : 1;
    bool last_axis_not_compute = last_axis == 0 || last_axis == channel_axis;
    if (last_axis_not_compute) {
        int last_dim = 1;
        auto &dim_tmp = inputs[0]->get_shape().back();
        if (dim_tmp.isa<constant>()) {
            last_dim = get_const_as_int(dim_tmp.checked_as<constant_c>());
        }
        auto vector_lanes = vectorize_step(
                ctx, info_.inputs_[0]->details_.dtype_.type_code_);
        if (last_dim / vector_lanes && last_dim % vector_lanes == 0) {
            vx_info_.lanes = vector_lanes;
        }
    }
    auto in_dtype = info_.inputs_[0]->details_.dtype_;
    auto out_dtype = info_.outputs_[0]->details_.dtype_;
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);

    compute_block_pooling(inputs, *dst[0], pooling_type_, kernel_, stride_,
            pads_begin_,
            get_ncx_formatcode_vector_form_tensor(
                    info_.inputs_[0], channel_last_),
            vx_info_, in_dtype, out_dtype, attrs_, info_.outputs_[0], wkld);
}

void pooling_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;
    auto &in_fmt = info_.inputs_[0]->details_.get_format();
    check_format(in_fmt, channel_last_);
    in_formats.push_back({in_fmt});
    out_formats.push_back({in_fmt});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

size_t pooling_op_t::compute_workload(const std::vector<shape_dtype_pair> &ins,
        const std::vector<shape_dtype_pair> &outs) {
    auto &in_shape = ins[0].first;
    auto &out_shape = outs[0].first;
    auto &in_dtype = ins[0].second;
    auto real_compute_axis = get_real_pooling_axis_form_tensor(
            info_.inputs_[0], channel_last_);

    size_t wkld = utils::get_sizeof_type(in_dtype) * read_weight;
    size_t wkld_out = utils::get_sizeof_type(in_dtype) * write_weight;
    for (auto &compute_axis : real_compute_axis) {
        wkld *= in_shape[compute_axis];
        wkld_out *= out_shape[compute_axis];
    }

    wkld += wkld_out;
    wkld *= workload_penalty_coefficient;
    return wkld;
}

sc_dims pooling_op_t::_calculate_output_dims(
        bool rounding_floor, bool channel_last) {
    auto &input_dims = info_.inputs_[0]->details_.get_plain_dims();
    unsigned n_plain_dims = input_dims.size();
    unsigned n_pads_dims = n_plain_dims - 2;
    const int channel_axis = channel_last ? input_dims.size() - 1 : 1;
    const int shape_begin_axis = channel_last ? 1 : 2;

    sc_dims output_dims(n_plain_dims);
    output_dims[0] = input_dims[0];
    output_dims[channel_axis] = input_dims[channel_axis];

    for (unsigned i = 0; i < n_pads_dims; i++) {
        int padding = pads_begin_[i] + pads_end_[i];
        if (rounding_floor) {
            output_dims[shape_begin_axis + i]
                    = (input_dims[shape_begin_axis + i] + padding - kernel_[i])
                            / stride_[i]
                    + 1;
        } else {
            output_dims[shape_begin_axis + i]
                    = utils::divide_and_ceil(input_dims[shape_begin_axis + i]
                                      + padding - kernel_[i],
                              stride_[i])
                    + 1;
        }
    }
    return output_dims;
}

pooling_avg_op_t::pooling_avg_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : parent(ins, outs, pooling_type_t::avg, attrs) {
    op_name_ = "pooling_avg";
    COMPILE_ASSERT(
            attrs.has_key("exclude_pad"), "avg pooling must have exclude_pad");
}

pooling_max_op_t::pooling_max_op_t(const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : parent(ins, outs, pooling_type_t::max, attrs) {
    op_name_ = "pooling_max";
}

pooling_backprop_op_t::pooling_backprop_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs) {
    // set inputs_ and attr_
    COMPILE_ASSERT(!ins.empty(), "at least 1 input for pooling_backprop_op_t ");
    unsigned n_plain_dims = ins[0]->details_.get_plain_dims().size();
    COMPILE_ASSERT(n_plain_dims == 4 || n_plain_dims == 5,
            "input should have 4 or 5 n-dims,but got " << n_plain_dims);
    info_.inputs_ = ins;
    attrs_ = attrs;

    channel_last_ = check_data_format_channel_last(attrs_);
    // set kernel_ and  stride_ vars
    uint64_t n_kernel_dims = n_plain_dims - 2;
    check_and_set_kernel_strides_and_pooling_type(
            attrs_, n_kernel_dims, kernel_, stride_, pooling_type_);

    // set pads_begin_ and pads_begin_
    COMPILE_ASSERT(attrs.has_key(pooling_attr_key::input_shape),
            "the pooling_backprop_op_t op should have input_shape "
            "attribute");
    sc_dims input_plain_shape
            = attrs.get<sc_dims>(pooling_attr_key::input_shape);
    check_and_set_pads_begin_and_pads_end(
            attrs_, input_plain_shape, pads_begin_, pads_end_, channel_last_);

    // set outputs
    sc_dims out_delta_dims = input_plain_shape;
    if (outs.empty()) {
        info_.outputs_.emplace_back(std::make_shared<graph_tensor>(this,
                info_.inputs_[0]->details_.get_format(), out_delta_dims,
                ins[0]->details_.dtype_));
    } else {
        COMPILE_ASSERT(outs.size() == 1, "pooling backprop expect 1 output");
        COMPILE_ASSERT(outs[0]->details_.get_plain_dims() == out_delta_dims
                        && outs[0]->details_.dtype_ == ins[0]->details_.dtype_,
                "Bad output shape for pooling backprop")
        info_.outputs_ = outs;
    }
}

pooling_avg_backprop_op_t::pooling_avg_backprop_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : pooling_backprop_op_t(ins, outs,
            add_pl_type(attrs, static_cast<int>(pooling_type_t::avg))) {
    COMPILE_ASSERT(ins.size() == 1, " pooling_avg_backprop_op_t have 1 inputs");
    op_name_ = "pooling_avg_backprop";
    COMPILE_ASSERT(
            attrs.has_key("exclude_pad"), "avg pooling must have exclude_pad");
}

pooling_avg_backprop_op_t::pooling_avg_backprop_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, sc_dims &input_shape,
        const any_map_t &attrs)
    : pooling_backprop_op_t(ins, outs,
            add_pl_type_and_in_shape(attrs,
                    static_cast<int>(pooling_type_t::max), input_shape)) {
    COMPILE_ASSERT(ins.size() == 1, " pooling_avg_backprop_op_t have 1 inputs");
    op_name_ = "pooling_avg_backprop";
    COMPILE_ASSERT(
            attrs.has_key("exclude_pad"), "avg pooling must have exclude_pad");
}

pooling_max_backprop_op_t::pooling_max_backprop_op_t(
        const std::vector<graph_tensor_ptr> &ins,
        const std::vector<graph_tensor_ptr> &outs, const any_map_t &attrs)
    : pooling_backprop_op_t(ins, outs,
            add_pl_type_and_in_shape(attrs,
                    static_cast<int>(pooling_type_t::max),
                    ins[1]->details_.get_plain_dims())) {
    COMPILE_ASSERT(info_.inputs_.size() == 2,
            " pooling_max_backprop_op_t have 2 inputs");
    const auto output_delta_ndims
            = info_.inputs_[0]->details_.get_plain_dims().size();
    const auto input_tensor_ndims
            = info_.inputs_[1]->details_.get_plain_dims().size();
    COMPILE_ASSERT(input_tensor_ndims == output_delta_ndims,
            "delta should have n dims as input tensor");
    op_name_ = "pooling_max_backprop";
}

void pooling_backprop_op_t::query_format(context_ptr ctx,
        std::vector<std::vector<format_stride_pair>> &supported_ins,
        std::vector<std::vector<format_stride_pair>> &supported_outs) {
    std::vector<std::vector<sc_data_format_t>> in_formats, out_formats;

    // infer output format,same as in_fmormat_of_output_delta
    sc_data_format_t in_fmt_of_delta = info_.inputs_[0]->details_.get_format();
    check_format(in_fmt_of_delta, channel_last_);
    sc_data_format_t out_fmt = in_fmt_of_delta;

    // infer inputs formats
    std::vector<sc_data_format_t> in_fmts;
    in_fmts.reserve(info_.inputs_.size());
    for (const auto &in_tensor : info_.inputs_) {
        sc_data_format_t in_fmt = in_tensor->details_.get_format();
        check_format(in_fmt, channel_last_);
        in_fmts.emplace_back(in_fmt);
    }

    // set result
    in_formats.push_back(in_fmts);
    out_formats.push_back({out_fmt});
    format_to_dense_format_stride_pair(
            in_formats, out_formats, supported_ins, supported_outs);
}

void pooling_backprop_op_t::prepare_fusion_data(fdata_map &fdmap) {
    fdmap.get(info_.inputs_[0]).use_count_++;
}
void pooling_backprop_op_t::infer_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {
    // search known ranges from any input of cur fusbile op
    slice_range_map known_ranges_map
            = search_known_slice_ranges(this, fsmap, stat_map);

    // judge inputs' dims full on w and h (and d)
    auto real_required_axis = get_real_pooling_axis_form_tensor(
            info_.inputs_[0], channel_last_);
    auto &in_blocked_dims = info_.inputs_[0]->details_.get_blocking_dims();
    for (auto &range_list : known_ranges_map[0]) {
        if (!slice_full_on_axis(
                    in_blocked_dims, range_list, real_required_axis)) {
            stat_map.append_ops_by_status(this, infer_status_code::RETRY);
            return;
        }
    }

    // infer other input slice range
    if (known_ranges_map.size() < get_inputs().size()) {
        for (size_t i = 0; i < get_inputs().size(); i++) {
            if (i == 0) continue;
            auto o_ranges_list = infer_pool_slice_ranges(
                    info_.inputs_[i], known_ranges_map[0], channel_last_);
            fsmap.get(info_.inputs_[i]) = o_ranges_list;
        }
    }

    // compute output slice range list
    auto o_ranges_list = infer_pool_slice_ranges(
            info_.outputs_[0], known_ranges_map[0], channel_last_);
    fsmap.get(info_.outputs_[0]) = o_ranges_list;
}
void pooling_backprop_op_t::pre_slice_ranges(
        fslice_map &fsmap, infer_status_map_t &stat_map) {}

static void pooling_backward_fill_zero_dst(const tensor_slice &dst,
        pooling_type_t pooling_typ, sc_data_type_t in_dtype,
        const vectorized_info_t &vx_info) {
    auto in_vectorized_dtype
            = sc_data_type_t(in_dtype.type_code_, vx_info.lanes);

    auto bld = builder::get_current_builder();
    COMPILE_ASSERT(bld, "No active builder is set");
    // make iter vars as index
    std::vector<expr> dst_idx;
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        dst_idx.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
    }
    // make assign node
    expr indexed_in_delta
            = builder::make_indexing(dst.tptr_, dst_idx, vx_info.lanes);
    stmt indelta_zero_asnode = make_stmt<assign_node_t>(indexed_in_delta,
            make_expr<constant_node>(0.f, in_vectorized_dtype));
    // make loops
    stmt body, cur;
    for (int i = dst_idx.size() - 1; i >= 0; i--) {
        if (i == int(dst_idx.size() - 1)) { cur = indelta_zero_asnode; }
        body = make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(dst_idx.at(i)), 0,
                dst.get_shape()[i],
                (i == int(dst_idx.size() - 1)) ? int(vx_info.lanes) : 1,
                std::move(body), true, for_type::NORMAL);
    }
    cur->attr()[stmt_attr_key::merge_loop] = false;
    bld->emit(cur);
}

static void compute_block_pooling_backward_avg(
        const std::vector<const tensor_slice *> &inputs,
        const tensor_slice &dst, sc_dims kernel, sc_dims stride,
        sc_dims pads_begin, const std::vector<int> &dst_fmt_vector,
        const vectorized_info_t &vx_info, sc_data_type_t in_dtype,
        any_map_t &attrs, size_t wkld = 0UL) {
    /***The final IR may look like below:
     * _for_(_fuseiter_i, 0, I, 1)
     *   _for_(_fuseiter_j, 0, J, 1)
     *     // set src_delta to zero
     *     _for_(_fuseiter_src_k, 0, K, 1)
     *       _for_(_fuseiter_src_l, 0, L, 1)
     *         src_idx = [i,j,k,l]
     *         src_delta[src_idx] = 0
     *     _for_(_fuseiter_k, 0, K, 1)
     *       _for_(_fuseiter_l, 0, L, 1)
     *         src_idx = [i,j,k * stride_h - pad_begin_h,l * stride_l -
     *               pad_begin_l]
     *         dst_idx = [i,j,k,l]
     *         _for_(_fuseiter_s, 0, S, 1)
     *           _if(tmp_src_h>=0 && tmp_h< H )
     *             _for_(_fuseiter_r, 0, R, 1)
     *               tmp_src = src_idx + [0,0,s,r]
     *               _if (tmp_src_w>=0 && tmp_src_w< W)
     *                 src_delta[src_idx] += dst_delta[dst_idx]/num;
     ***/

    // fill dst delta with 0
    pooling_backward_fill_zero_dst(dst, pooling_type_t::avg, in_dtype, vx_info);

    auto in_vectorized_dtype
            = sc_data_type_t(in_dtype.type_code_, vx_info.lanes);
    // builder
    auto bld = builder::get_current_builder();

    // nested loop vars
    std::vector<expr> iter_vars, kernel_iter_vars;

    // the indices for the source delta
    for (unsigned i = 0; i < inputs[0]->nslice_dims(); i++) {
        iter_vars.emplace_back(range_from_outer_loop(inputs[0]->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
    }
    expr indexed_src_delta = builder::make_indexing(
            inputs[0]->tptr_, iter_vars, vx_info.lanes);

    // the indices dor the kernel inner loop
    for (unsigned i = 0; i < kernel.size(); i++) {
        kernel_iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
    }

    auto kernel_size_var
            = builder::make_var(in_vectorized_dtype, "kernel_size");
    auto define_kernel_size_var
            = make_stmt<define_node_t>(kernel_size_var, linkage::local, expr());

    // the indices of output delta
    std::vector<expr> dst_delta_idx;
    std::vector<expr> conds(kernel.size());
    auto vectorized_int
            = sc_data_type_t(datatypes::s32.type_code_, vx_info.lanes);

    expr kernel_size_multi_expr = kernel_size_var;
    bool multi_first = true;
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        if (dst_fmt_vector[i] < 2) {
            dst_delta_idx.emplace_back(iter_vars[i]);
        } else {
            int plan_axis = dst_fmt_vector[i];
            int pads_stride_index = plan_axis - 2;
            // k * stride_h - pad_begin_h + kernel_var_h
            auto idx = int(stride[pads_stride_index]) * iter_vars[i]
                    - int(pads_begin[pads_stride_index])
                    + kernel_iter_vars[pads_stride_index];
            expr tmp_cond
                    = builder::make_logic_and(builder::make_cmp_ge(idx, 0),
                            builder::make_cmp_lt(idx, dst.get_shape()[i]));
            conds[pads_stride_index] = tmp_cond;
            dst_delta_idx.emplace_back(idx);
            auto window_start = make_expr<intrin_call_node>(intrin_type::max,
                    std::vector<expr> {make_expr<constant_node>(
                                               int64_t(0), vectorized_int),
                            int(stride[pads_stride_index])
                                            * make_expr<cast_node>(
                                                    vectorized_int,
                                                    iter_vars[i])
                                    - int(pads_begin[pads_stride_index])},
                    any_map_t());
            auto window_end = make_expr<intrin_call_node>(intrin_type::min,
                    std::vector<expr> {make_expr<cast_node>(vectorized_int,
                                               dst.get_shape()[i]),
                            int(stride[pads_stride_index])
                                            * make_expr<cast_node>(
                                                    vectorized_int,
                                                    iter_vars[i])
                                    - int(pads_begin[pads_stride_index])
                                    + int(kernel[pads_stride_index])},
                    any_map_t());
            if (multi_first) {
                kernel_size_multi_expr
                        = builder::make_sub(window_end, window_start);
                multi_first = false;
            } else
                kernel_size_multi_expr
                        = builder::make_mul(kernel_size_multi_expr,
                                builder::make_sub(window_end, window_start));
        }
    }
    expr indexed_dst_delta
            = builder::make_indexing(dst.tptr_, dst_delta_idx, vx_info.lanes);

    // assign kernel_size node
    stmt kernel_size_asnode;
    int kernel_size = 1;
    for (auto kn : kernel)
        kernel_size = kernel_size * int(kn);

    bool exclude_pad = attrs.get<bool>("exclude_pad");
    if (in_dtype.type_code_ == sc_data_etype::F32
            || in_dtype.type_code_ == sc_data_etype::BF16) {
        if (exclude_pad) {
            kernel_size_asnode = make_stmt<assign_node_t>(
                    kernel_size_var, kernel_size_multi_expr);
        } else {
            kernel_size_asnode = make_stmt<assign_node_t>(kernel_size_var,
                    make_expr<constant_node>(
                            float(kernel_size), in_vectorized_dtype));
        }

    } else {
        COMPILE_ASSERT(0, "unsupported in_dtype.");
    }

    // build inner kernel loop
    stmt cur, body;
    for (int i = kernel_iter_vars.size() - 1; i >= 0; i--) {
        stmt else_stmt, then_stmt;
        if (i == int(kernel_iter_vars.size() - 1)) {
            then_stmt = make_stmt<assign_node_t>(indexed_dst_delta,
                    builder::make_add(indexed_dst_delta,
                            builder::make_div(
                                    indexed_src_delta, kernel_size_var)));
        } else {
            then_stmt = cur;
        }

        cur = make_stmt<if_else_node_t>(conds[i], then_stmt, else_stmt);

        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(kernel_iter_vars.at(i)),
                expr(0), int(kernel[i]), expr(1), std::move(body), true,
                for_type::NORMAL);
    }

    // build outter loop
    stmt target_assign;
    for (int i = iter_vars.size() - 1; i >= 0; i--) {
        if (i == int(iter_vars.size() - 1)) {
            cur = cur.isa<stmts>() ? cur
                                   : make_stmt<stmts_node_t>(std::vector<stmt> {
                                           kernel_size_asnode, std::move(cur)});
        }
        // Do not generate those dummy loops
        if (!iter_vars.at(i).isa<var>()) continue;
        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});

        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), 0,
                inputs[0]->get_shape()[i],
                (i == int(iter_vars.size() - 1)) ? int(vx_info.lanes) : 1,
                std::move(body), true, for_type::NORMAL);

        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    }

    if (cur.isa<for_loop>()) cur->attr()[stmt_attr_key::merge_loop] = false;
    cur = make_stmt<stmts_node_t>(
            std::vector<stmt> {define_kernel_size_var, std::move(cur)});

    bld->emit(cur);
}

static void compute_block_pooling_backward_max(
        const std::vector<const tensor_slice *> &inputs,
        const tensor_slice &dst, sc_dims kernel, sc_dims stride,
        sc_dims pads_begin, const std::vector<int> &dst_fmt_vector,
        const vectorized_info_t &vx_info, sc_data_type_t in_dtype,
        any_map_t &attrs, size_t wkld = 0UL) {
    /***The final IR may look like below:
     * _for_(_fuseiter_i, 0, I, 1)
     *   _for_(_fuseiter_j, 0, J, 1)
     *     // set src_delta to zero
     *     _for_(_fuseiter_src_k, 0, K, 1)
     *       _for_(_fuseiter_src_l, 0, L, 1)
     *         src_idx = [i,j,k,l]
     *         src_delta[src_idx] = 0
     *     _for_(_fuseiter_k, 0, K, 1)
     *       _for_(_fuseiter_l, 0, L, 1)
     *         src_idx = [i,j,k * stride_h - pad_begin_h,
     *                     l * stride_l - pad_begin_l]
     *         dst_idx = [i,j,k,l]
     *         max_idx_h = -1
     *         max_idx_w = -1
     *         max_val = -inf
     *         has_max = true
     *         _for_(_fuseiter_s, 0, S, 1)
     *           _if(tmp_src_h>=0 && tmp_src_h< H )
     *             _for_(_fuseiter_r, 0, R, 1)
     *               tmp_src = src_idx + [0,0,s,r]
     *                 _if (tmp_src_w>=0 && tmp_src_w< W)
     *                   _if in_tenor[src_idx] >= max_val:
     *                     max_index_h&w = src_idx_h&w;
     *                     max_val = in_tenor[src_idx]
     *                     has_max = true
     *                  _else
     *                     _if max_val<0:
     *                       has_max = false;
     *                       max_val = 0
     *           _if max_index_h >=0:
     *             src_delta[max_idx]+=dst_delta[dst_idx]
     ***/

    // fill dst delta with 0
    pooling_backward_fill_zero_dst(dst, pooling_type_t::avg, in_dtype, vx_info);

    // builder
    auto bld = builder::get_current_builder();

    // set expr max_val to save max value and max_val_pos vector to save its
    // position
    expr max_val, has_max;
    stmt define_has_max;
    std::vector<expr> max_val_pos(kernel.size());
    std::vector<stmt> defines_of_max;
    std::vector<stmt> assigns_of_max;

    max_val = builder::make_var(in_dtype, "max_val");
    defines_of_max.emplace_back(
            make_stmt<define_node_t>(max_val, linkage::local, expr()));
    std::string pos_name[] = {"i", "j", "k"};
    for (size_t i = 0; i < kernel.size(); i++) {
        max_val_pos[i] = builder::make_var(
                datatypes::index, "max_val_pos_" + pos_name[i]);
        defines_of_max.emplace_back(make_stmt<define_node_t>(
                max_val_pos[i], linkage::local, expr()));
    }
    if (in_dtype.type_code_ == sc_data_etype::F32
            || in_dtype.type_code_ == sc_data_etype::BF16) {
        assigns_of_max.emplace_back(make_stmt<assign_node_t>(max_val,
                make_expr<constant_node>(
                        -std::numeric_limits<float>::infinity(), in_dtype)));
    } else {
        COMPILE_ASSERT(0, "unsupported in_dtype.");
    }

    has_max = builder::make_var(datatypes::boolean, "has_max");
    define_has_max = make_stmt<define_node_t>(has_max, linkage::local, false);

    // nested loop vars
    std::vector<expr> iter_vars, kernel_iter_vars;

    // the indices for the source delta
    for (unsigned i = 0; i < inputs[0]->nslice_dims(); i++) {
        iter_vars.emplace_back(range_from_outer_loop(inputs[0]->get_ranges()[i])
                        ? expr(0)
                        : builder::make_var(datatypes::index,
                                std::string("_fuseiter")
                                        + fusion_create_idx()));
    }
    expr indexed_src_delta
            = builder::make_indexing(inputs[0]->tptr_, iter_vars);

    // the indices dor the kernel inner loop
    for (unsigned i = 0; i < kernel.size(); i++) {
        kernel_iter_vars.emplace_back(builder::make_var(datatypes::index,
                std::string("_fuseiter") + fusion_create_idx()));
    }

    // the indices of output delta and max_value position's delta
    std::vector<expr> dst_delta_idices;
    std::vector<expr> max_value_delta_idices;
    std::vector<expr> cur_tensor_idices;
    std::vector<expr> conds(kernel.size());
    std::vector<stmt> update_max_pos_stmts;
    for (unsigned i = 0; i < dst.nslice_dims(); i++) {
        if (dst_fmt_vector[i] < 2) {
            dst_delta_idices.emplace_back(iter_vars[i]);
            max_value_delta_idices.emplace_back(iter_vars[i]);
            cur_tensor_idices.emplace_back(iter_vars[i]);
        } else {
            int plan_axis = dst_fmt_vector[i];
            int pads_stride_index = plan_axis - 2;
            // k * stride_h - pad_begin_h + kernel_var_h
            auto idx = int(stride[pads_stride_index]) * iter_vars[i]
                    - int(pads_begin[pads_stride_index])
                    + kernel_iter_vars[pads_stride_index];
            expr tmp_cond
                    = builder::make_logic_and(builder::make_cmp_ge(idx, 0),
                            builder::make_cmp_lt(idx, dst.get_shape()[i]));
            conds[pads_stride_index] = tmp_cond;
            dst_delta_idices.emplace_back(idx);
            cur_tensor_idices.emplace_back(idx);
            update_max_pos_stmts.emplace_back(make_stmt<assign_node_t>(
                    max_val_pos[pads_stride_index], idx));
            max_value_delta_idices.emplace_back(max_val_pos[pads_stride_index]);
        }
    }
    expr indexed_dst_delta
            = builder::make_indexing(dst.tptr_, dst_delta_idices);
    expr indexed_max_dst_delta, indexed_cur_tensor_val;

    indexed_max_dst_delta
            = builder::make_indexing(dst.tptr_, max_value_delta_idices);
    indexed_cur_tensor_val
            = builder::make_indexing(inputs[1]->tptr_, cur_tensor_idices);

    // build inner kernel loop
    stmt cur, body;
    for (int i = kernel_iter_vars.size() - 1; i >= 0; i--) {
        stmt else_stmt, then_stmt;
        if (i == int(kernel_iter_vars.size() - 1)) {
            update_max_pos_stmts.emplace_back(
                    make_stmt<assign_node_t>(max_val, indexed_cur_tensor_val));
            update_max_pos_stmts.emplace_back(
                    make_stmt<assign_node_t>(has_max, true));
            stmt update_max_info
                    = make_stmt<stmts_node_t>(std::move(update_max_pos_stmts));
            then_stmt = make_stmt<if_else_node_t>(
                    max_val <= indexed_cur_tensor_val, update_max_info, stmt());
        } else {
            then_stmt = cur;
        }

        expr zero_constant = make_expr<constant_node>(0.f, in_dtype);
        else_stmt = make_stmt<if_else_node_t>(max_val < zero_constant,
                make_stmt<stmts_node_t>(std::vector<stmt> {
                        make_stmt<assign_node_t>(has_max, false),

                        make_stmt<assign_node_t>(max_val, zero_constant)}),
                stmt());

        cur = make_stmt<if_else_node_t>(conds[i], then_stmt, else_stmt);

        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});
        cur = make_stmt<for_loop_node_t>(std::move(kernel_iter_vars.at(i)),
                expr(0), int(kernel[i]), expr(1), std::move(body), true,
                for_type::NORMAL);
    }

    // build outter loop
    for (int i = iter_vars.size() - 1; i >= 0; i--) {
        if (i == int(iter_vars.size() - 1)) {
            std::vector<stmt> kernel_body = assigns_of_max;
            kernel_body.emplace_back(define_has_max);
            kernel_body.emplace_back(std::move(cur));
            kernel_body.emplace_back(make_stmt<if_else_node_t>(has_max,
                    make_stmt<assign_node_t>(indexed_max_dst_delta,
                            builder::make_add(
                                    indexed_max_dst_delta, indexed_src_delta)),
                    stmt()));
            cur = make_stmt<stmts_node_t>(std::move(kernel_body));
        }

        // Do not generate those dummy loops
        if (!iter_vars.at(i).isa<var>()) continue;

        body = cur.isa<stmts>()
                ? cur
                : make_stmt<stmts_node_t>(std::vector<stmt> {std::move(cur)});

        cur = make_stmt<for_loop_node_t>(std::move(iter_vars.at(i)), 0,
                inputs[0]->get_shape()[i], 1, std::move(body), true,
                for_type::NORMAL);

        cur->attr()[op_traits::workload_computable_t::workload_number] = wkld;
    }

    if (cur.isa<for_loop>()) cur->attr()[stmt_attr_key::merge_loop] = false;
    std::vector<stmt> func_body = defines_of_max;
    func_body.emplace_back(std::move(cur));
    cur = make_stmt<stmts_node_t>(std::move(func_body));

    bld->emit(cur);
}

void pooling_backprop_op_t::compute_block(context_ptr ctx,
        const std::vector<tensor_slice *> &dst,
        const std::vector<const tensor_slice *> &inputs) {
    // set up vx_info
    vx_info_.axis = dst[0]->get_shape().size() - 1;
    vx_info_.lanes = 1;
    // if last axis of all inputs are not h or w (or d) ,lanes can be not 1
    auto vector_lanes
            = vectorize_step(ctx, info_.inputs_[0]->details_.dtype_.type_code_);
    for (size_t i = 0; i < info_.inputs_.size(); i++) {
        auto last_axis
                = info_.inputs_[i]->details_.get_format().format_code_.get(
                        inputs[i]->nbase_dims() - 1);
        bool last_axis_not_compute
                = last_axis != 2 && last_axis != 3 && last_axis != 4;
        if (last_axis_not_compute) {
            int last_dim = 1;
            auto &dim_tmp = inputs[i]->get_shape().back();
            if (dim_tmp.isa<constant>()) {
                last_dim = get_const_as_int(dim_tmp.checked_as<constant_c>());
            }
            if (last_dim / vector_lanes && last_dim % vector_lanes == 0) {
                vx_info_.lanes = vector_lanes;
            } else {
                vx_info_.lanes = 1;
            }
        }
    }
    auto in_dtype = info_.inputs_[0]->details_.dtype_;
    size_t wkld = compute_fusible_workload(ctx, dst, inputs);

    if (pooling_type_ == pooling_type_t::avg)
        compute_block_pooling_backward_avg(inputs, *dst[0], kernel_, stride_,
                pads_begin_,
                get_ncx_formatcode_vector_form_tensor(
                        info_.inputs_[0], channel_last_),
                vx_info_, in_dtype, attrs_);
    else
        compute_block_pooling_backward_max(inputs, *dst[0], kernel_, stride_,
                pads_begin_,
                get_ncx_formatcode_vector_form_tensor(
                        info_.inputs_[0], channel_last_),
                vx_info_, in_dtype, attrs_);
}

size_t pooling_backprop_op_t::compute_workload(
        const std::vector<shape_dtype_pair> &,
        const std::vector<shape_dtype_pair> &) {
    return 0;
}
OP_REGISTER(pooling_avg_op_t, pooling_avg)
OP_REGISTER(pooling_max_op_t, pooling_max)
OP_REGISTER(pooling_avg_backprop_op_t, pooling_avg_backprop)
OP_REGISTER(pooling_max_backprop_op_t, pooling_max_backprop)
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
