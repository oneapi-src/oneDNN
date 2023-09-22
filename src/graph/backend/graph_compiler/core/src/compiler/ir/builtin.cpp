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

#include "builtin.hpp"
#include <array>
#include <tuple>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/builder.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <unordered_map>
#include <util/math_utils.hpp>
#include <util/utils.hpp>

SC_MODULE(microkernel.builtin)

using namespace dnnl::impl::graph::gc::builder;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
expr get_ir_null() {
    return make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
}
expr get_ir_zero_index() {
    return make_expr<constant_node>(UINT64_C(0), datatypes::index);
}
bool is_pure_func_call(const expr_c &v) {
    if (!v.isa<call>()) { return false; }
    auto func = v.static_as<call_c>()->get_prototype();
    return func->attr_ && func->attr_->get_or_else(function_attrs::pure, false);
}

namespace builtin {
sc_data_type_t infer_output_dtype(sc_data_type_t dtype_A) {
    if (dtype_A == datatypes::u8 || dtype_A == datatypes::s8) {
        return datatypes::s32;
    }
    return datatypes::f32;
}

void print_index(expr v) {
    static const func_t print_index_f = make_func("print_index",
            {make_var(datatypes::index, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_index_f, std::move(v));
}

void print_int(expr v) {
    static const func_t print_int_f = make_func("print_int",
            {make_var(datatypes::s32, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_int_f, std::move(v));
}

void print_float(expr v) {
    static const func_t print_float_f = make_func("print_float",
            {make_var(datatypes::f32, "v")}, stmt(), datatypes::void_t);
    _evaluate_call_(print_float_f, std::move(v));
}

void print_str(expr v) {
    static const func_t print_str_f = make_func("print_str",
            {make_var(sc_data_type_t::pointerof(sc_data_etype::U8), "v")},
            stmt(), datatypes::void_t);
    _evaluate_call_(print_str_f, std::move(v));
}

void print_str(const std::string &v) {
    print_str(builder::get_current_builder()->make_str(v));
}

void print_str(const char *v) {
    print_str(std::string(v));
}

static const func_t &mark_trace_func(const func_t &f) {
    f->attr()[function_attrs::is_trace_func] = true;
    return f;
}

expr make_trace(expr func_name, expr in_or_out, expr arg) {
    static func_t make_trace_f = mark_trace_func(make_func("sc_make_trace",
            {make_var(datatypes::s32, "func_name"),
                    make_var(datatypes::s32, "in_or_out"),
                    make_var(datatypes::s32, "arg")},
            stmt(), datatypes::void_t));
    return make_trace_f(
            std::move(func_name), std::move(in_or_out), std::move(arg));
}

expr make_trace_kernel(expr func_name, expr in_or_out, expr arg) {
    static func_t make_trace_f
            = mark_trace_func(make_func("sc_make_trace_kernel",
                    {make_var(datatypes::s32, "func_name"),
                            make_var(datatypes::s32, "in_or_out"),
                            make_var(datatypes::s32, "arg")},
                    stmt(), datatypes::void_t));
    return make_trace_f(
            std::move(func_name), std::move(in_or_out), std::move(arg));
}

func_t get_brgemm_init_func() {
    static func_t brgemm_func = _decl_func("dnnl_brgemm_init", datatypes::s32,
            {_arg_("C", datatypes::pointer), _arg_("M", datatypes::s32),
                    _arg_("N", datatypes::s32), _arg_("LDC", datatypes::s32),
                    _arg_("dtypeC", datatypes::s32),
                    _arg_("value", datatypes::f32)});
    return brgemm_func;
}

void dnnl_brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value) {
    _evaluate_call_(get_brgemm_init_func(), std::move(C), std::move(M),
            std::move(N), std::move(LDC), dtypeC.as_etype_int(),
            std::move(value));
}

static const char *brgemm_names[] = {
        "dnnl",
        "sc",
};

static const char *get_brgemm_name(scflags_t::brgemm_t backend) {
    return brgemm_names[static_cast<int>(backend)];
}

static const func_t &mark_brgemm_func(const func_t &f) {
    f->attr()["is_brgemm_func_with_stream"] = true;
    return f;
}

// returns the kernel creator and kernel caller pair
static std::pair<func_t, func_t> declare_brgemm_kernel_creator(
        scflags_t::brgemm_t backend, brgemm_mode mode, bool has_postop) {
    std::stringstream ss;
    std::string postfix = has_postop ? "_call_postops" : "_call";
    std::vector<std::vector<expr>> post_args
            = {_arg_("postops_data", datatypes::pointer),
                    _arg_("c_buf", datatypes::pointer)};
    if (mode == brgemm_mode::stride) {
        ss << get_brgemm_name(backend) << "_brgemm";
        static func_t creator
                = _decl_func(ss.str() + "_func", datatypes::pointer,
                        {_arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                                _arg_("K", datatypes::s32),
                                _arg_("LDA", datatypes::s32),
                                _arg_("LDB", datatypes::s32),
                                _arg_("LDC", datatypes::s32),
                                _arg_("stride_a", datatypes::s32),
                                _arg_("stride_b", datatypes::s32),
                                _arg_("beta", datatypes::f32),
                                _arg_("dtypeA", datatypes::s32),
                                _arg_("dtypeB", datatypes::s32),
                                _arg_("brg_attrs", datatypes::pointer),
                                _arg_("bd_mask", datatypes::pointer),
                                _arg_("postops_setting", datatypes::pointer)});
        auto caller_args = std::vector<std::vector<expr>> {
                _arg_("func", datatypes::pointer),
                _arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                _arg_("C", datatypes::pointer), _arg_("num", datatypes::s32),
                _arg_("stream", datatypes::pointer)};
        if (has_postop) {
            caller_args.insert(
                    caller_args.end() - 1, post_args.begin(), post_args.end());
        }
        func_t caller = mark_brgemm_func(_decl_func(
                ss.str() + postfix, datatypes::void_t, std::move(caller_args)));
        return std::pair<func_t, func_t>(creator, caller);
    } else {
        ss << get_brgemm_name(backend) << "_brgemm_list";
        static func_t creator
                = _decl_func(ss.str() + "_func", datatypes::pointer,
                        {_arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                                _arg_("K", datatypes::s32),
                                _arg_("LDA", datatypes::s32),
                                _arg_("LDB", datatypes::s32),
                                _arg_("LDC", datatypes::s32),
                                _arg_("beta", datatypes::f32),
                                _arg_("dtypeA", datatypes::s32),
                                _arg_("dtypeB", datatypes::s32),
                                _arg_("brg_attrs", datatypes::pointer),
                                _arg_("bd_mask", datatypes::pointer),
                                _arg_("postops_setting", datatypes::pointer)});
        auto caller_args = std::vector<std::vector<expr>> {
                _arg_("func", datatypes::pointer),
                _arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                _arg_("C", datatypes::pointer), _arg_("num", datatypes::s32),
                _arg_("stride_a", datatypes::s32),
                _arg_("stride_b", datatypes::s32), _arg_("len", datatypes::s32),
                _arg_("dtypeA", datatypes::s32),
                _arg_("dtypeB", datatypes::s32),
                _arg_("stream", datatypes::pointer)};
        if (has_postop) {
            caller_args.insert(
                    caller_args.end() - 1, post_args.begin(), post_args.end());
        }
        func_t caller = mark_brgemm_func(_decl_func(
                ss.str() + postfix, datatypes::void_t, std::move(caller_args)));
        return std::pair<func_t, func_t>(creator, caller);
    }
}

std::pair<func_t, func_t> get_brgemm_creator_and_call_func(
        brgemm_mode mode, scflags_t::brgemm_t backend, bool has_postop) {
#define DEF_FUNC(back, list_stride) \
    if (mode == brgemm_mode::list_stride \
            && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f0 \
                = declare_brgemm_kernel_creator(backend, mode, false); \
        static std::pair<func_t, func_t> f1 \
                = declare_brgemm_kernel_creator(backend, mode, true); \
        return has_postop ? f1 : f0; \
    }
    // we need a static variable each branch to ensure there will be no
    // duplicated decl for the same func_t.
    DEF_FUNC(dnnl, stride)
    DEF_FUNC(dnnl, addr_list)
#undef DEF_FUNC
    assert(0 && "Unreachable");
    return std::pair<func_t, func_t>();
}

// returns the kernel creator and kernel caller pair
static func_t declare_brgemm_update_funcs(
        scflags_t::brgemm_t backend, brgemm_mode mode, bool init) {
    std::stringstream ss;
    if (mode == brgemm_mode::stride) {
        ss << get_brgemm_name(backend) << "_brgemm_";
        if (init) { ss << "init_"; }
        ss << "update";
        auto update_args = std::vector<std::vector<expr>> {
                _arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                _arg_("C", datatypes::pointer), _arg_("num", datatypes::s32),
                _arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                _arg_("K", datatypes::s32), _arg_("LDA", datatypes::s32),
                _arg_("LDB", datatypes::s32), _arg_("LDC", datatypes::s32),
                _arg_("stride_a", datatypes::s32),
                _arg_("stride_b", datatypes::s32),
                _arg_("dtypeA", datatypes::s32),
                _arg_("dtypeB", datatypes::s32),
                _arg_("brg_attrs", datatypes::pointer),
                _arg_("bd_mask", datatypes::pointer),
                _arg_("postops_setting", datatypes::pointer),
                _arg_("postops_data", datatypes::pointer),
                _arg_("c_buf", datatypes::pointer),
                _arg_("stream", datatypes::pointer)};

        func_t update = mark_brgemm_func(
                _decl_func(ss.str(), datatypes::s32, std::move(update_args)));
        return update;
    } else {
        ss << get_brgemm_name(backend) << "_brgemm_";
        if (init) { ss << "init_"; }
        ss << "list_update";
        auto update_args = std::vector<std::vector<expr>> {
                _arg_("A", datatypes::pointer), _arg_("B", datatypes::pointer),
                _arg_("C", datatypes::pointer), _arg_("num", datatypes::s32),
                _arg_("M", datatypes::s32), _arg_("N", datatypes::s32),
                _arg_("K", datatypes::s32), _arg_("LDA", datatypes::s32),
                _arg_("LDB", datatypes::s32), _arg_("LDC", datatypes::s32),
                _arg_("stride_a", datatypes::s32),
                _arg_("stride_b", datatypes::s32), _arg_("len", datatypes::s32),
                _arg_("dtypeA", datatypes::s32),
                _arg_("dtypeB", datatypes::s32),
                _arg_("brg_attrs", datatypes::pointer),
                _arg_("bd_mask", datatypes::pointer),
                _arg_("postops_setting", datatypes::pointer),
                _arg_("postops_data", datatypes::pointer),
                _arg_("c_buf", datatypes::pointer),
                _arg_("stream", datatypes::pointer)};
        func_t brgemm_func = mark_brgemm_func(
                _decl_func(ss.str(), datatypes::s32, std::move(update_args)));
        return brgemm_func;
    }
}

func_t get_brgemm_call_range_func(brgemm_mode mode) {
    if (mode == brgemm_mode::stride) {
        auto update_args = std::vector<std::vector<expr>> {
                _arg_("func", datatypes::pointer),
                _arg_("M_real", datatypes::s32),
                _arg_("N_real", datatypes::s32),
                _arg_("K_real", datatypes::s32), _arg_("A", datatypes::pointer),
                _arg_("B", datatypes::pointer), _arg_("C", datatypes::pointer),
                _arg_("num", datatypes::s32),
                _arg_("stream", datatypes::pointer)};

        static func_t update
                = mark_brgemm_func(_decl_func("dnnl_brgemm_call_range",
                        datatypes::s32, std::move(update_args)));
        return update;
    } else {
        auto update_args = std::vector<std::vector<expr>> {
                _arg_("func", datatypes::pointer),
                _arg_("M_real", datatypes::s32),
                _arg_("N_real", datatypes::s32),
                _arg_("K_real", datatypes::s32), _arg_("A", datatypes::pointer),
                _arg_("B", datatypes::pointer), _arg_("C", datatypes::pointer),
                _arg_("num", datatypes::s32), _arg_("stride_a", datatypes::s32),
                _arg_("stride_b", datatypes::s32), _arg_("len", datatypes::s32),
                _arg_("dtypeA", datatypes::s32),
                _arg_("dtypeB", datatypes::s32),
                _arg_("stream", datatypes::pointer)};
        static func_t brgemm_func
                = mark_brgemm_func(_decl_func("dnnl_brgemm_list_call_range",
                        datatypes::s32, std::move(update_args)));
        return brgemm_func;
    }
}

std::pair<func_t, func_t> get_brgemm_update_funcs(
        brgemm_mode mode, scflags_t::brgemm_t backend) {
#define DEF_FUNC(back) \
    if (mode == brgemm_mode::stride && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f = std::pair<func_t, func_t>( \
                declare_brgemm_update_funcs(backend, mode, false), \
                declare_brgemm_update_funcs(backend, mode, true)); \
        return f; \
    }
#define DEF_LIST_FUNC(back) \
    if (mode == brgemm_mode::addr_list \
            && backend == scflags_t::brgemm_t::back) { \
        static std::pair<func_t, func_t> f = std::pair<func_t, func_t>( \
                declare_brgemm_update_funcs(backend, mode, false), \
                declare_brgemm_update_funcs(backend, mode, true)); \
        return f; \
    }
    // we need a static variable each branch to ensure there will be no
    // duplicated decl for the same func_t.
    DEF_FUNC(dnnl)
    DEF_LIST_FUNC(dnnl)
#undef DEF_FUNC
#undef DEF_LIST_FUNC
    assert(0 && "Unreachable");
    return std::pair<func_t, func_t>();
}

evaluate brgemm_init_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    return builder::get_current_builder()
            ->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a, stride_b,
                    postops_data, c_buf, bd_mask_idx,
                    {brgemm_args::cpu_t {true}, dtypeA, dtypeB,
                            infer_output_dtype(dtypeA), brg_attrs, bd_mask,
                            bd_mask_set_num, postops_set})
            .checked_as<evaluate>();
}

evaluate brgemm_init_update_allow_fusion(const expr &A, const expr &B,
        const expr &C, const expr &num, const expr &M, const expr &N,
        const expr &K, const expr &LDA, const expr &LDB, const expr &LDC,
        const expr &stride_a, const expr &stride_b,
        const sc_data_type_t &dtypeA, const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs, const sc_brgemm_bd_mask_t &bd_mask,
        const expr &bd_mask_idx, const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    auto ret = brgemm_init_update(A, B, C, num, M, N, K, LDA, LDB, LDC,
            stride_a, stride_b, dtypeA, dtypeB, brg_attrs, bd_mask, bd_mask_idx,
            bd_mask_set_num, postops_set, postops_data, c_buf)
                       .checked_as<evaluate>();
    builder::get_current_builder()
            ->get_current_scope()
            .body.back()
            .checked_as<evaluate>()
            ->value_.checked_as<intrin_call>()
            ->intrin_attrs_->set(intrin_attr::allow_brgemm_fusion, true);
    return ret;
}

evaluate brgemm_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    return builder::get_current_builder()
            ->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a, stride_b,
                    postops_data, c_buf, bd_mask_idx,
                    {brgemm_args::cpu_t {false}, dtypeA, dtypeB,
                            infer_output_dtype(dtypeA), brg_attrs, bd_mask,
                            bd_mask_set_num, postops_set})
            .checked_as<evaluate>();
}

evaluate brgemm_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const expr &len, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    A->attr()[attr_keys::no_index2var] = true;
    A->attr()["list_brgemm_arg"] = true;
    B->attr()[attr_keys::no_index2var] = true;
    B->attr()["list_brgemm_arg"] = true;
    return builder::get_current_builder()
            ->list_brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a,
                    stride_b, len, postops_data, c_buf, bd_mask_idx,
                    brgemm_args::extra_args_t(brgemm_args::cpu_t {false},
                            dtypeA, dtypeB, infer_output_dtype(dtypeA),
                            brg_attrs, bd_mask, bd_mask_set_num, postops_set))
            .checked_as<evaluate>();
}

evaluate brgemm_init_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const expr &len, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    A->attr()[attr_keys::no_index2var] = true;
    A->attr()["list_brgemm_arg"] = true;
    B->attr()[attr_keys::no_index2var] = true;
    B->attr()["list_brgemm_arg"] = true;
    return builder::get_current_builder()
            ->list_brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a,
                    stride_b, len, postops_data, c_buf, bd_mask_idx,
                    brgemm_args::extra_args_t(brgemm_args::cpu_t {true}, dtypeA,
                            dtypeB, infer_output_dtype(dtypeA), brg_attrs,
                            bd_mask, bd_mask_set_num, postops_set))
            .checked_as<evaluate>();
}

void brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value) {
    dnnl_brgemm_init(std::move(C), std::move(M), std::move(N), std::move(LDC),
            dtypeC, std::move(value));
}

func_t get_mem_set_func() {
    static func_t memzerofunc = _decl_func("memset", datatypes::pointer,
            {_arg_("ptr", datatypes::pointer), _arg_("v", datatypes::s32),
                    _arg_("len", datatypes::index)});
    return memzerofunc;
}

void mem_zero(expr C, const expr &size, sc_data_type_t dtype) {
    _evaluate_call_(get_mem_set_func(), std::move(C), 0,
            size * utils::get_sizeof_type(dtype));
}

func_t get_brgemm_postops_data_init_func() {
    static func_t data_init_func
            = _decl_func("dnnl_brgemm_postops_data_init", datatypes::void_t,
                    {_arg_("dnnl_data", datatypes::pointer),
                            _arg_("bias", datatypes::pointer),
                            _arg_("scales", datatypes::pointer),
                            _arg_("binary_post_ops_rhs", datatypes::pointer),
                            _arg_("oc_logical_off", datatypes::index),
                            _arg_("dst_row_logical_off", datatypes::index),
                            _arg_("data_C_ptr_", datatypes::pointer),
                            _arg_("first_mb_matrix_addr_off", datatypes::index),
                            _arg_("a_zp_compensations", datatypes::pointer),
                            _arg_("b_zp_compensations", datatypes::pointer),
                            _arg_("c_zp_values", datatypes::pointer),
                            _arg_("skip_accumulation", datatypes::boolean),
                            _arg_("zp_a_val", datatypes::s32),
                            _arg_("do_only_comp", datatypes::boolean),
                            _arg_("do_only_zp_a_val", datatypes::boolean)});
    return data_init_func;
}

std::vector<expr> create_initialed_postops_data() {
    std::vector<expr> data(brgemm::postops_data_init_func_nargs, get_ir_null());
    auto false_node = make_expr<constant_node>(UINT64_C(0), datatypes::boolean);
    // oc_logical_off
    data[3] = get_ir_zero_index();
    // dst_row_logical_off
    data[4] = get_ir_zero_index();
    // first_mb_matrix_addr_off
    data[6] = get_ir_zero_index();
    // skip_accumulation
    data[10] = false_node;
    // zp_a_val
    data[11] = make_expr<constant_node>(UINT64_C(0), datatypes::s32);
    // do_only_comp
    data[12] = false_node;
    // do_only_zp_a_val
    data[13] = false_node;
    return data;
}

// dynamic query format function call
expr call_matmul_core_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &ori_in0,
        const expr &ori_in1, const expr &out_format0, const expr &in_format0,
        const expr &in_format1, const expr &ori_in_format0,
        const expr &ori_in_format1, const expr &out_size, const expr &kernel,
        const expr &impl) {
    static func_t matmul_core_query_f = make_func("query_format_matmul_core_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp0"),
                    make_var(datatypes::pointer, "inp1"),
                    make_var(datatypes::pointer, "ori_inp0"),
                    make_var(datatypes::pointer, "ori_inp1"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt0"),
                    make_var(datatypes::pointer, "inp_fmt1"),
                    make_var(datatypes::pointer, "ori_inp_fmt0"),
                    make_var(datatypes::pointer, "ori_inp_fmt1"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel"),
                    make_var(datatypes::pointer, "impl")},
            stmt(), datatypes::void_t);
    return matmul_core_query_f(tb, out0, in0, in1, ori_in0, ori_in1,
            out_format0, in_format0, in_format1, ori_in_format0, ori_in_format1,
            out_size, kernel, impl);
}

expr call_managed_matmul_core_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &ori_in0,
        const expr &ori_in1, const expr &out_format0, const expr &in_format0,
        const expr &in_format1, const expr &ori_in_format0,
        const expr &ori_in_format1, const expr &out_size, const expr &kernel,
        const expr &impl) {
    static func_t mmm_query_f = make_func("query_format_managed_matmul_core_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp0"),
                    make_var(datatypes::pointer, "inp1"),
                    make_var(datatypes::pointer, "ori_inp0"),
                    make_var(datatypes::pointer, "ori_inp1"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt0"),
                    make_var(datatypes::pointer, "inp_fmt1"),
                    make_var(datatypes::pointer, "ori_inp_fmt0"),
                    make_var(datatypes::pointer, "ori_inp_fmt1"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel"),
                    make_var(datatypes::pointer, "impl")},
            stmt(), datatypes::void_t);
    return mmm_query_f(tb, out0, in0, in1, ori_in0, ori_in1, out_format0,
            in_format0, in_format1, ori_in_format0, ori_in_format1, out_size,
            kernel, impl);
}

expr call_conv_fwd_core_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &ori_in0,
        const expr &ori_in1, const expr &out_format0, const expr &in_format0,
        const expr &in_format1, const expr &ori_in_format0,
        const expr &ori_in_format1, const expr &out_size, const expr &kernel,
        const expr &impl) {
    static func_t conv_fwd_core_query_f
            = make_func("query_format_conv_fwd_core_op",
                    {make_var(datatypes::pointer, "op_table"),
                            make_var(datatypes::pointer, "out"),
                            make_var(datatypes::pointer, "inp0"),
                            make_var(datatypes::pointer, "inp1"),
                            make_var(datatypes::pointer, "ori_inp0"),
                            make_var(datatypes::pointer, "ori_inp1"),
                            make_var(datatypes::pointer, "out_fmt"),
                            make_var(datatypes::pointer, "inp_fmt0"),
                            make_var(datatypes::pointer, "inp_fmt1"),
                            make_var(datatypes::pointer, "ori_inp_fmt0"),
                            make_var(datatypes::pointer, "ori_inp_fmt1"),
                            make_var(datatypes::pointer, "out_size"),
                            make_var(datatypes::pointer, "kernel"),
                            make_var(datatypes::pointer, "impl")},
                    stmt(), datatypes::void_t);
    return conv_fwd_core_query_f(tb, out0, in0, in1, ori_in0, ori_in1,
            out_format0, in_format0, in_format1, ori_in_format0, ori_in_format1,
            out_size, kernel, impl);
}

expr call_unary_fusible_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel) {
    static func_t unary_query_f = make_func("query_format_unary_fusible_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return unary_query_f(
            tb, out0, in0, out_format0, in_format0, out_size, kernel);
}

expr call_padding_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel) {
    static func_t padding_query_f = make_func("query_format_padding_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return padding_query_f(
            tb, out0, in0, out_format0, in_format0, out_size, kernel);
}

expr call_binary_fusible_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &out_format0,
        const expr &in_format0, const expr &in_format1, const expr &out_size,
        const expr &kernel) {
    static func_t binary_query_f = make_func("query_format_binary_fusible_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp0"),
                    make_var(datatypes::pointer, "inp1"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt0"),
                    make_var(datatypes::pointer, "inp_fmt1"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return binary_query_f(tb, out0, in0, in1, out_format0, in_format0,
            in_format1, out_size, kernel);
}

expr call_reorder_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel, const expr &impl) {
    static func_t reorder_query_f = make_func("query_format_reorder_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel"),
                    make_var(datatypes::pointer, "impl")},
            stmt(), datatypes::void_t);
    return reorder_query_f(
            tb, out0, in0, out_format0, in_format0, out_size, kernel, impl);
}

expr call_reduce_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel) {
    static func_t reorder_query_f = make_func("query_format_reduce_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    //     make_var(datatypes::s32, "rd_axis"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return reorder_query_f(
            tb, out0, in0, out_format0, in_format0, out_size, kernel);
}

expr call_tensor_view_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel) {
    static func_t reorder_query_f = make_func("query_format_tensor_view_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return reorder_query_f(
            tb, out0, in0, out_format0, in_format0, out_size, kernel);
}

expr call_select_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &in2,
        const expr &out_format0, const expr &in_format0, const expr &in_format1,
        const expr &in_format2, const expr &out_size, const expr &kernel) {
    static func_t select_query_f = make_func("query_format_select_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "out"),
                    make_var(datatypes::pointer, "inp0"),
                    make_var(datatypes::pointer, "inp1"),
                    make_var(datatypes::pointer, "inp2"),
                    make_var(datatypes::pointer, "out_fmt"),
                    make_var(datatypes::pointer, "inp0_fmt"),
                    make_var(datatypes::pointer, "inp1_fmt"),
                    make_var(datatypes::pointer, "inp2_fmt"),
                    make_var(datatypes::pointer, "out_size"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return select_query_f(tb, out0, in0, in1, in2, out_format0, in_format0,
            in_format1, in_format2, out_size, kernel);
}

expr call_fused_op_query_combined(const expr &tb, const expr &combined_keys,
        const expr &combined_algs, const expr &each_op_num_key,
        const expr &op_num, const expr &kernel) {
    static func_t fused_query_f = make_func("query_combined_fused_op",
            {make_var(datatypes::pointer, "op_table"),
                    make_var(datatypes::pointer, "combined_keys"),
                    make_var(datatypes::pointer, "combined_algs"),
                    make_var(datatypes::pointer, "each_op_num_key"),
                    make_var(datatypes::s32, "op_num"),
                    make_var(datatypes::pointer, "kernel")},
            stmt(), datatypes::void_t);
    return fused_query_f(
            tb, combined_keys, combined_algs, each_op_num_key, op_num, kernel);
}

expr call_cal_blocking_dims(const expr &placeholder, const expr &format) {
    static func_t cal_blocking_f = make_func("calculate_blocking_dims",
            {make_var(datatypes::pointer, "placeholder"),
                    make_var(datatypes::pointer, "format")},
            stmt(), datatypes::index);
    return cal_blocking_f(placeholder, format);
}

expr call_get_matmul_dyn_cfg_single(const expr &in, const expr &is_batch) {
    static func_t matmul_dyn_cfg_f = make_func("get_matmul_dyn_cfg_single",
            {make_var(datatypes::s32, "in"),
                    make_var(datatypes::boolean, "is_batch")},
            stmt(), datatypes::s32);
    return matmul_dyn_cfg_f(in, is_batch);
}

static func_t set_pure_function(func_t f) {
    f->attr()[function_attrs::pure] = true;
    return f;
}

func_t get_thread_id_func() {
    static func_t func = set_pure_function(
            _decl_func("sc_get_thread_id", datatypes::s32, {}));
    return func;
}

func_t get_is_in_parallel_func() {
    static func_t func = set_pure_function(
            _decl_func("sc_is_in_parallel", datatypes::s32, {}));
    return func;
}

func_t get_barrier_arrive_func() {
    static func_t func = _decl_func("sc_arrive_at_barrier", datatypes::void_t,
            {_arg_("b", datatypes::pointer),
                    _arg_("idle_func", datatypes::pointer),
                    _arg_("idle_args", datatypes::pointer)});
    return func;
}

func_t get_init_barrier_func() {
    static func_t func = _decl_func("sc_init_barrier", datatypes::void_t,
            {_arg_("b", datatypes::pointer), _arg_("num", datatypes::s32),
                    _arg_("thread_cnt", datatypes::index)});
    return func;
}

func_t get_set_idle_func_managed_func() {
    static func_t f = builder::make_func("sc_set_idle_func_managed",
            {builder::make_var(datatypes::pointer, "funcptr"),
                    builder::make_var(datatypes::pointer, "arg1")},
            stmt(), datatypes::void_t);
    return f;
}

func_t get_tls_amx_buffer_func() {
    static func_t f = set_pure_function(_decl_func("sc_get_tls_amx_buffer",
            datatypes::pointer, {_arg_("stream", datatypes::pointer)}));
    return f;
}

// get the sum of elements with indices [0,1,...,gid] of the
// array: [A,A,..., A, B, B,...,B], where the count of element "A" is num_A
static expr generate_balance211_job_id_base(const expr &gid, const expr &num_A,
        const expr &val_A, const expr &val_B) {
    return builder::make_select(
            gid <= num_A, gid * val_A, num_A * val_A + (gid - num_A) * val_B);
}

uint64_t generate_balance211(int num_threads, const expr &start_e,
        const expr &end_e, const expr &step_e, const expr &gid,
        const std::function<std::string(const char *)> &namer, expr *out_start,
        expr *out_len, expr *out_end, std::vector<stmt> *out_seq) {
    // the util function to define a var with init value
    auto def_var = [&](const char *name, const expr &init_v,
                           sc_data_type_t dtype = datatypes::index) {
        auto ret = builder::make_var(dtype, namer ? namer(name) : name);
        auto def = builder::make_var_tensor_def_unattached(
                ret, linkage::local, do_cast_and_fold(init_v));
        if (out_seq) {
            out_seq->emplace_back(def);
        } else {
            builder::get_current_builder()->emit(def);
        }
        return ret;
    };
    // the util function to define the expressions to return
    auto make_output_vars = [&](expr &my_begin, const expr &cur_jobs) {
        my_begin = do_cast_and_fold(my_begin);
        *out_start = def_var("_start", my_begin);

        expr out_len_e;
        if (out_end || out_len) {
            out_len_e = def_var("_len", do_cast_and_fold(cur_jobs * step_e));
        }
        if (out_end) {
            expr my_end = *out_start + out_len_e;
            my_end = do_cast_and_fold(my_end);
            *out_end = def_var("_end", my_end);
        }
        if (out_len) { *out_len = out_len_e; }
    };
    expr the_tid, my_jobs_e, my_jobs_2_e;
    if (start_e.isa<constant>() && end_e.isa<constant>()
            && step_e.isa<constant>()) {
        // if is constant-for (in most cases)
        uint64_t end = get_const_as_int(end_e.static_as<constant>());
        uint64_t begin = get_const_as_int(start_e.static_as<constant>());
        uint64_t step = get_const_as_int(step_e.static_as<constant>());
        auto len = end - begin;
        auto num_jobs = utils::divide_and_ceil(len, step);
        uint64_t my_jobs = utils::divide_and_ceil(num_jobs, num_threads);
        COMPILE_ASSERT(my_jobs > 0, "Bad number of jobs");
        if (num_jobs % num_threads == 0) {
            // fast path. the jobs is divisible by the thread number
            *out_start = def_var("_start", gid * (my_jobs * step) + begin);
            if (out_end) {
                *out_end = def_var("_end", *out_start + (my_jobs * step));
            }
            if (out_len) { *out_len = my_jobs * step; }
            return num_threads;
        }
        uint64_t my_jobs_2 = my_jobs - 1;
        // number of threads doing my_jobs work
        uint64_t num_thread_larger_work = num_jobs - my_jobs_2 * num_threads;
        // number of threads doing my_jobs - 1 work
        uint64_t num_thread_less_work = num_threads - num_thread_larger_work;
        // the loop is divisible with num_thread_less_work parts
        // and each part has same number of works
        // the loop can be further "split" into outer and inner loops and
        // the outer loop may be merged with another loop
        uint64_t num_split
                = math_utils::get_gcd(num_thread_larger_work, num_threads);
        if (num_split > 1UL) {
            // e.g. 22 jobs and 8 threads
            // my_jobs=3 my_jobs_2=2
            // num_thread_larger_work = 6
            // num_split = gcd(num_thread_larger_work, num_threads) = 2
            // num_thread_each_split = 4
            // num_thread_larger_work_per_split = 6/2 = 3
            // cur_jobs for each thread: 3,3,3,2 | 3,3,3,2 <<< 2 splits
            uint64_t num_thread_each_split = num_threads / num_split;
            uint64_t num_thread_larger_work_per_split
                    = num_thread_larger_work / num_split;
            expr id_in_split
                    = def_var("id_in_split", gid % num_thread_each_split);
            expr is_large_work = id_in_split < num_thread_larger_work_per_split;
            expr cur_jobs
                    = builder::make_select(is_large_work, my_jobs, my_jobs_2);
            // for gid=5, base_job_id = 11
            expr base_job_id
                    = (gid / num_thread_each_split) * (num_jobs / num_split);
            // for gid=5, job_id_offset = 3
            expr job_id_offset;
            if (num_thread_larger_work_per_split == num_thread_each_split - 1) {
                job_id_offset = id_in_split * my_jobs;
            } else {
                job_id_offset = generate_balance211_job_id_base(id_in_split,
                        num_thread_larger_work_per_split, my_jobs, my_jobs_2);
            }
            expr my_begin = start_e + (base_job_id + job_id_offset) * step_e;
            make_output_vars(my_begin, cur_jobs);
            return num_split;
        }
        // fallback way: non-divisible loop
        the_tid = num_thread_larger_work;
        my_jobs_e = my_jobs;
        my_jobs_2_e = my_jobs_2;

    } else {
        auto &end = end_e;
        auto &begin = start_e;
        auto &step = step_e;
        auto len = end - begin;
        auto num_jobs = def_var("num_jobs", (len + step - 1) / step);
        my_jobs_e = def_var(
                "my_jobs", (num_jobs + (num_threads - 1)) / num_threads);
        // assert(my_jobs > 0);
        my_jobs_2_e = def_var("my_jobs2", my_jobs_e - 1);
        the_tid = def_var("the_tid", num_jobs - my_jobs_2_e * num_threads);
    }
    expr cur_jobs = builder::make_select(gid < the_tid, my_jobs_e, my_jobs_2_e);
    expr my_begin = start_e
            + generate_balance211_job_id_base(
                      gid, the_tid, my_jobs_e, my_jobs_2_e)
                    * step_e;
    make_output_vars(my_begin, cur_jobs);
    return 0;
}

} // namespace builtin
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
