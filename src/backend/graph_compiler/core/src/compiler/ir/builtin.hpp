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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_BUILTIN_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_BUILTIN_HPP

#include <functional>
#include <string>
#include <utility>
#include <vector>
#include <compiler/config/context.hpp>
#include <compiler/ir/sc_function.hpp>
#include <runtime/microkernel/cpu/brgemm_common.hpp>
#include <util/utils.hpp>

namespace sc {
SC_INTERNAL_API expr get_ir_null();
SC_INTERNAL_API expr get_ir_zero_index();
namespace builtin {

/**
 * Infer output dtype based on dtype of input A for brgemm
 * */
SC_INTERNAL_API sc_data_type_t infer_output_dtype(sc_data_type_t dtype_A);

/**
 * Generates a call node to print_index, and wrap the call node with
 * evaluate node. Also declares the print_index function in the builder
 * */
void print_index(expr v);

/**
 * Generates a call node to print_int, and wrap the call node with
 * evaluate node. Also declares the print_int function in the builder
 * */
void print_int(expr v);

/**
 * Generates a call node to print_float, and wrap the call node with
 * evaluate node. Also declares the print_float function in the builder
 * */
void print_float(expr v);

/**
 * Generates a call node to print_str function, and wrap the call node
 * with evaluate node. Also declares the print_str function in the builder
 * */
void print_str(expr v);

/**
 * Generates a string and pass the string to a call node to print_str function,
 * and wrap the call node with evaluate node. Also declares the print_str
 * function in the builder
 * */
void print_str(const std::string &v);

/**
 * Generates a string and pass the string to a call node to print_str function,
 * and wrap the call node with evaluate node. Also declares the print_str
 * function in the builder
 * */
void print_str(const char *v);

/**
 * Generates a call to boundary_check
 * @param name the tensor name, should be a string
 * @param idx the index to check
 * @param access_len the length of the access in number of elements, for
 * accessing 1 element, this parameter should be 1
 * @param boundary_len max boundary in number of elements
 * @return the return value of boundary_check, should equal to `idx`
 * */
expr boundary_check(expr name, expr idx, expr access_len, expr boundary_len);

/**
 * Generates a evaluate_call to sc_make_trace
 * @param func_id the function id, s32
 * @param in_or_out s32, if 0, this is the entry trace of the function. if 1,
 * this is the exit trace of the function
 * @param arg the argument in the trace
 * */
expr make_trace(expr func_name, expr in_or_out, expr arg);

/**
 * Generates a evaluate_call to sc_make_trace_kernel
 * @param func_id the function id, s32
 * @param in_or_out s32, if 0, this is the entry trace of the function. if 1,
 * this is the exit trace of the function
 * @param arg the argument in the trace
 * */
expr make_trace_kernel(expr func_name, expr in_or_out, expr arg);

// Create a initialized postops data vector. Its length matches the number of
// dnnl postop data init func args
SC_INTERNAL_API std::vector<expr> create_initialed_postops_data();

/**
 * Generates a call node to dnnl_brgemm_init, and wrap the call node
 * with evaluate node. Also declares the dnnl_brgemm_init function in
 * the builder
 *
 * @param C pointer (float)
 * @param M s32
 * @param N s32
 * @param LDC s32
 * @param dtypeC sc_data_type_t
 * @param value f32
 * */
void dnnl_brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value);

/**
 * Generates a generate call node to brgemm_init_update.
 *
 * @param A pointer (float)
 * @param B pointer (float)
 * @param C pointer (float)
 * @param num s32
 * @param M s32
 * @param N s32
 * @param K s32
 * @param LDA s32
 * @param LDB s32
 * @param LDC s32
 * @param stride_a s32
 * @param stride_b s32
 * @param dtypeA sc_data_type_t
 * @param dtypeB sc_data_type_t
 * @param brg_attrs any_map
 * @param bd_mask bd_mask
 * @param bd_mask_idx bd_mask idx for same brgemm with different bd_mask
 * @param bd_mask_set_num num of different bd_mask for same brgemm
 * @param brg_postops_setting postops_setting
 * @param brg_postops_data postops_data
 * @param brg_c_buf c_buf
 * */
void brgemm_init_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &LDA, const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
        const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
        const expr &bd_mask_idx = get_ir_zero_index(),
        const int &bd_mask_set_num = 1,
        const sc_brgemm_postops_setting_t &brg_postops_setting
        = sc_brgemm_postops_setting_t(),
        const std::vector<expr> &brg_postops_data
        = create_initialed_postops_data(),
        const expr &brg_c_buf = get_ir_null());

/**
 * Generates a generate call node to brgemm_init_update. If you want to use
 * brgemm fusion, please use this interface. Notice that brgemm fusion needs all
 * calculation of reduce axis done.
 *
 * @param A pointer (float)
 * @param B pointer (float)
 * @param C pointer (float)
 * @param num s32
 * @param M s32
 * @param N s32
 * @param K s32
 * @param LDA s32
 * @param LDB s32
 * @param LDC s32
 * @param stride_a s32
 * @param stride_b s32
 * @param dtypeA sc_data_type_t
 * @param dtypeB sc_data_type_t
 * @param brg_attrs any_map
 * @param bd_mask bd_mask
 * @param bd_mask_idx bd_mask idx for same brgemm with different bd_mask
 * @param bd_mask_set_num num of different bd_mask for same brgemm
 * @param brg_postops_setting postops_setting
 * @param brg_postops_data postops_data
 * @param brg_c_buf c_buf
 * */
void brgemm_init_update_allow_fusion(const expr &A, const expr &B,
        const expr &C, const expr &num, const expr &M, const expr &N,
        const expr &K, const expr &LDA, const expr &LDB, const expr &LDC,
        const expr &stride_a, const expr &stride_b,
        const sc_data_type_t &dtypeA, const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
        const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
        const expr &bd_mask_idx = get_ir_zero_index(),
        const int &bd_mask_set_num = 1,
        const sc_brgemm_postops_setting_t &brg_postops_setting
        = sc_brgemm_postops_setting_t(),
        const std::vector<expr> &brg_postops_data
        = create_initialed_postops_data(),
        const expr &brg_c_buf = get_ir_null());

/**
 * Generates a generate call node to brgemm_init_f32.
 *
 * @param C pointer (void)
 * @param M s32
 * @param N s32
 * @param LDC s32
 * @param dtypeC sc_data_type_t
 * @param value f32
 * */
void brgemm_init(
        expr C, expr M, expr N, expr LDC, sc_data_type_t dtypeC, expr value);

/**
 * Generates a generate call node to brgemm_update.
 *
 * @param A pointer (void)
 * @param B pointer (void)
 * @param C pointer (void)
 * @param num s32
 * @param M s32
 * @param N s32
 * @param K s32
 * @param LDA s32
 * @param LDB s32
 * @param LDC s32
 * @param stride_a s32
 * @param stride_b s32
 * @param dtypeA sc_data_type_t
 * @param dtypeB sc_data_type_t
 * @param brg_attrs any_map
 * @param bd_mask bd_mask
 * @param bd_mask_idx bd_mask idx for same brgemm with different bd_mask
 * @param bd_mask_set_num num of different bd_mask for same brgemm
 * @param brg_postops_setting postops_setting
 * @param brg_postops_data postops_data
 * @param brg_c_buf c_buf
 * */
void brgemm_update(const expr &A, const expr &B, const expr &C, const expr &num,
        const expr &M, const expr &N, const expr &K, const expr &LDA,
        const expr &LDB, const expr &LDC, const expr &stride_a,
        const expr &stride_b, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
        const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
        const expr &bd_mask_idx = get_ir_zero_index(),
        const int &bd_mask_set_num = 1,
        const sc_brgemm_postops_setting_t &brg_postops_setting
        = sc_brgemm_postops_setting_t(),
        const std::vector<expr> &brg_postops_data
        = create_initialed_postops_data(),
        const expr &brg_c_buf = get_ir_null());

/**
 * Generates a generate call node to brgemm_list_update.
 *
 * @param A pointer (void)
 * @param B pointer (void)
 * @param C pointer (void)
 * @param num s32
 * @param M s32
 * @param N s32
 * @param K s32
 * @param lda s32
 * @param ldb s32
 * @param ldc s32
 * @param stride_a s32
 * @param stride_b s32
 * @param len s32 (len of addr list, each addr list contains num addrs which
 * inferred via stride_a and strid_b)
 * @param dtypeA sc_data_type_t
 * @param dtypeB sc_data_type_t
 * @param brg_attrs any_map
 * @param bd_mask bd_mask
 * @param bd_mask_idx bd_mask idx for same brgemm with different bd_mask
 * @param bd_mask_set_num num of different bd_mask for same brgemm
 * @param brg_postops_setting postops_setting
 * @param brg_postops_data postops_data
 * @param brg_c_buf c_buf
 * */
void brgemm_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &lda, const expr &ldb, const expr &ldc, const expr &stride_a,
        const expr &stride_b, const expr &len, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
        const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
        const expr &bd_mask_idx = get_ir_zero_index(),
        const int &bd_mask_set_num = 1,
        const sc_brgemm_postops_setting_t &brg_postops_setting
        = sc_brgemm_postops_setting_t(),
        const std::vector<expr> &brg_postops_data
        = create_initialed_postops_data(),
        const expr &brg_c_buf = get_ir_null());

/**
 * Generates a generate call node to brgemm_init_list_update.
 *
 * @param A pointer (void)
 * @param B pointer (void)
 * @param C pointer (void)
 * @param num s32
 * @param M s32
 * @param N s32
 * @param K s32
 * @param lda s32
 * @param ldb s32
 * @param ldc s32
 * @param stride_a s32
 * @param stride_b s32
 * @param len s32 (len of addr list, each addr list contains num addrs which
 * inferred via stride_a and strid_b)
 * @param dtypeA sc_data_type_t
 * @param dtypeB sc_data_type_t
 * @param brg_attrs any_map
 * @param bd_mask bd_mask
 * @param bd_mask_idx bd_mask idx for same brgemm with different bd_mask
 * @param bd_mask_set_num num of different bd_mask for same brgemm
 * @param brg_postops_setting postops_setting
 * @param brg_postops_data postops_data
 * @param brg_c_buf c_buf
 * */
void brgemm_init_list_update(const expr &A, const expr &B, const expr &C,
        const expr &num, const expr &M, const expr &N, const expr &K,
        const expr &lda, const expr &ldb, const expr &ldc, const expr &stride_a,
        const expr &stride_b, const expr &len, const sc_data_type_t &dtypeA,
        const sc_data_type_t &dtypeB,
        const sc_brgemm_attrs_t &brg_attrs = sc_brgemm_attrs_t(),
        const sc_brgemm_bd_mask_t &bd_mask = sc_brgemm_bd_mask_t(),
        const expr &bd_mask_idx = get_ir_zero_index(),
        const int &bd_mask_set_num = 1,
        const sc_brgemm_postops_setting_t &brg_postops_setting
        = sc_brgemm_postops_setting_t(),
        const std::vector<expr> &brg_postops_data
        = create_initialed_postops_data(),
        const expr &brg_c_buf = get_ir_null());

/**
 * Generates a generate call node to mem_zero.
 * use 'memset' to initialize a specified buffer to zero, size of which equals
 * to size*sizeof(dtype).
 *
 * @param C pointer (void)
 * @param size index
 * @param dtype sc_data_type_t
 * */
void mem_zero(expr C, const expr &size, sc_data_type_t dtype);
func_t get_mem_set_func();

/**
 * Generates a call node to convert multiple buffers(like scales, bias) to one
 * dnnl postop data type struct.
 *
 * @return the created func.
 * */
func_t get_brgemm_postops_data_init_func();

enum class brgemm_mode {
    // offset doesn't used for now.
    // offset,
    stride,
    addr_list,
};

// returns <kernerl creator, caller> pair
std::pair<func_t, func_t> get_brgemm_creator_and_call_func(
        brgemm_mode mode, scflags_t::brgemm_t backend, bool has_postop);

// returns <update, init_update> pair
std::pair<func_t, func_t> get_brgemm_update_funcs(
        brgemm_mode mode, scflags_t::brgemm_t backend);

// makes a call node to sc_dump_tensor function
expr call_dump_tensor(expr tsr, expr name, expr shape, expr size, expr limit,
        expr outpath, expr format, expr dtype, expr is_dynamic);

// makes a call node to sc_value_check function
expr call_value_check(expr tsr, expr name, expr size);

// dynamic query format function evaluation at runtime.
expr call_matmul_core_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &ori_in0,
        const expr &ori_in1, const expr &out_format0, const expr &in_format0,
        const expr &in_format1, const expr &ori_in_format0,
        const expr &ori_in_format1, const expr &out_size, const expr &kernel,
        const expr &impl = get_ir_null());
expr call_unary_fusible_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_foramt0, const expr &in_format0,
        const expr &out_size, const expr &kernel);
expr call_binary_fusible_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &in1, const expr &out_format0,
        const expr &in_format0, const expr &in_format1, const expr &out_size,
        const expr &kernel);
// reorder need to query its impl alg.
expr call_reorder_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_foramt0, const expr &in_format0,
        const expr &out_size, const expr &kernel,
        const expr &impl = get_ir_null());
expr call_reduce_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_foramt0, const expr &in_format0,
        const expr &out_size, const expr &kernel);
expr call_tensor_view_op_query_format(const expr &tb, const expr &out0,
        const expr &in0, const expr &out_format0, const expr &in_format0,
        const expr &out_size, const expr &kernel);
expr call_fused_op_query_combined(const expr &tb, const expr &combined_keys,
        const expr &combined_algs, const expr &each_op_num_key,
        const expr &op_num, const expr &kernel);
expr call_cal_blocking_dims(const expr &placeholder, const expr &format);
// runtime op shape of tensor
func_t get_cal_shape_of_tensor_op_func();

// gets the IR func for get_thread_id. @see thread_pool_table::get_thread_id
func_t get_thread_id_func();

// gets the IR func for is_in_parallel. @see thread_pool_table::is_in_parallel
func_t get_is_in_parallel_func();

// gets the IR func for sc::runtime::enter_barrier
func_t get_barrier_arrive_func();

// gets the IR func for sc::runtime::init_barrier
func_t get_init_barrier_func();

// gets the IR func for sc_set_idle_func_managed
func_t get_set_idle_func_managed_func();

func_t get_brgemm_init_func();

} // namespace builtin
} // namespace sc

#endif
