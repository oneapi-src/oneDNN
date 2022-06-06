/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
#include <unordered_map>
#include <util/utils.hpp>

SC_MODULE(microkernel.builtin)

using namespace sc::builder;
namespace sc {
expr get_ir_null() {
    static expr ret = make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
    return ret;
}
expr get_ir_zero_index() {
    static expr ret = make_expr<constant_node>(UINT64_C(0), datatypes::index);
    return ret;
}

namespace builtin {
static sc_data_type_t infer_output_dtype(sc_data_type_t dtype_A) {
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

expr boundary_check(expr name, expr idx, expr access_len, expr boundary_len) {
    static func_t boundary_check_f = make_func("boundary_check",
            {make_var(datatypes::pointer, "name"),
                    make_var(datatypes::index, "idx"),
                    make_var(datatypes::index, "access_len"),
                    make_var(datatypes::index, "boundary_len")},
            stmt(), datatypes::index);
    return boundary_check_f(std::move(name), std::move(idx),
            std::move(access_len), std::move(boundary_len));
}

expr make_trace(expr func_name, expr in_or_out) {
    static func_t make_trace_f = make_func("sc_make_trace",
            {make_var(datatypes::s32, "func_name"),
                    make_var(datatypes::s32, "in_or_out")},
            stmt(), datatypes::void_t);
    return make_trace_f(std::move(func_name), std::move(in_or_out));
}

expr call_dump_tensor(expr tsr, expr name, expr shape, expr size, expr limit,
        expr outpath, expr format, expr dtype) {
    static func_t dump_tensor_f = make_func("sc_dump_tensor",
            {make_var(datatypes::pointer, "tsr"),
                    make_var(datatypes::pointer, "name"),
                    make_var(datatypes::pointer, "shape"),
                    make_var(datatypes::index, "size"),
                    make_var(datatypes::index, "limit"),
                    make_var(datatypes::pointer, "outpath"),
                    make_var(datatypes::boolean, "format"),
                    make_var(datatypes::index, "dtype")},
            stmt(), datatypes::void_t);
    return dump_tensor_f(std::move(tsr), std::move(name), std::move(shape),
            std::move(size), std::move(limit), std::move(outpath),
            std::move(format), std::move(dtype));
}

expr call_value_check(expr tsr, expr name, expr size) {
    static func_t value_check_f = make_func("sc_value_check",
            {make_var(datatypes::pointer, "tsr"),
                    make_var(datatypes::pointer, "name"),
                    make_var(datatypes::index, "size")},
            stmt(), datatypes::void_t);
    return value_check_f(std::move(tsr), std::move(name), std::move(size));
}

void dnnl_brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value) {
    static func_t brgemm_func = _decl_func("dnnl_brgemm_init", datatypes::s32,
            {_arg_("C", datatypes::pointer), _arg_("M", datatypes::s32),
                    _arg_("N", datatypes::s32), _arg_("LDC", datatypes::s32),
                    _arg_("dtypeC", datatypes::s32),
                    _arg_("value", datatypes::f32)});
    _evaluate_call_(brgemm_func, std::move(C), std::move(M), std::move(N),
            std::move(LDC), dtypeC.as_etype_int(), std::move(value));
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
        ss << get_brgemm_name(backend) << "_brgemm_list_update";
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
                declare_brgemm_update_funcs(backend, mode, false), nullptr); \
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

void brgemm_init_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    builder::get_current_builder()->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC,
            stride_a, stride_b, postops_data, c_buf, bd_mask_idx,
            {brgemm_args::cpu_t {true}, dtypeA, dtypeB,
                    infer_output_dtype(dtypeA), brg_attrs, bd_mask,
                    bd_mask_set_num, postops_set});
}

void brgemm_init_update_allow_fusion(const expr &A, const expr &B,
        const expr &C, const expr &num, const expr &M, const expr &N,
        const expr &K, const expr &LDA, const expr &LDB, const expr &LDC,
        const expr &stride_a, const expr &stride_b,
        const sc_data_type_t &dtypeA, const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs, const sc_brgemm_bd_mask_t &bd_mask,
        const expr &bd_mask_idx, const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    brgemm_init_update(A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a, stride_b,
            dtypeA, dtypeB, brg_attrs, bd_mask, bd_mask_idx, bd_mask_set_num,
            postops_set, postops_data, c_buf);
    builder::get_current_builder()
            ->get_current_scope()
            .body.back()
            .checked_as<evaluate>()
            ->value_.checked_as<intrin_call>()
            ->intrin_attrs_->set(intrin_attr::allow_brgemm_fusion, true);
}

void brgemm_update(const expr &A, const expr &B, const expr &C, const expr &num,
        const expr &M, const expr &N, const expr &K, const expr &LDA,
        const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    builder::get_current_builder()->brgemm(A, B, C, num, M, N, K, LDA, LDB, LDC,
            stride_a, stride_b, postops_data, c_buf, bd_mask_idx,
            {brgemm_args::cpu_t {false}, dtypeA, dtypeB,
                    infer_output_dtype(dtypeA), brg_attrs, bd_mask,
                    bd_mask_set_num, postops_set});
}

void brgemm_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const expr &len, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB, const sc_brgemm_attrs_t &brg_attrs,
        const sc_brgemm_bd_mask_t &bd_mask, const expr &bd_mask_idx,
        const int &bd_mask_set_num,
        const sc_brgemm_postops_setting_t &postops_set,
        const std::vector<expr> &postops_data, const expr &c_buf) {
    builder::get_current_builder()->list_brgemm(A, B, C, num, M, N, K, LDA, LDB,
            LDC, stride_a, stride_b, len, postops_data, c_buf, bd_mask_idx,
            brgemm_args::extra_args_t(brgemm_args::cpu_t {false}, dtypeA,
                    dtypeB, infer_output_dtype(dtypeA), brg_attrs, bd_mask,
                    bd_mask_set_num, postops_set));
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
                            _arg_("skip_accumulation", datatypes::boolean)});
    return data_init_func;
}

std::vector<expr> create_initialed_postops_data() {
    std::vector<expr> data(brgemm::postops_data_init_func_nargs, get_ir_null());
    // oc_logical_off
    data[3] = get_ir_zero_index();
    // dst_row_logical_off
    data[4] = get_ir_zero_index();
    // first_mb_matrix_addr_off
    data[6] = get_ir_zero_index();
    // skip_accumulation
    data[10] = make_expr<constant_node>(UINT64_C(0), datatypes::boolean);
    return data;
}

static func_t set_pure_function(func_t f) {
    f->attr()["pure"] = true;
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

} // namespace builtin
} // namespace sc
