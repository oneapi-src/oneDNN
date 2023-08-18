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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_BUILDER_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_BUILDER_HPP

#include <functional>
#include <vector>

#include <memory>
#include <string>
#include <utility>
#include "intrinsics.hpp"
#include "sc_data_type.hpp"
#include "sc_expr.hpp"
#include "sc_function.hpp"
#include "sc_stmt.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * The IR builder. It contains utility functions that helps
 * to build the IR. For easier IR building, you can use easy_build,
 * which is based on these function. Each thread has a
 * thread-local pointer which points to the current builder. This
 * feature is useful for easy_build to save the user from providing
 * the current builder.
 * */
namespace builder {
/**
 * The internal implementation of IR builder. Use ir_builder_t
 * in most cases instead
 * */
class SC_INTERNAL_API builder_impl_t {
public:
    /**
     * A basic block is a list of statements that is temporarily
     * stored in the builder. A basic block can be appended with
     * a new statement. The builder will finally compose the
     * statements in a basic-block into a stmts node.
     * */
    struct basic_block_t {
        std::vector<stmt> body;
        /**
         * Appends a stmt into the BB
         * @param stmt the statement
         * */
        void emit(const stmt &stmt);
        /**
         * Compose a stmts node with the stmt that are previously
         * pushed into this BB. Will clear the `body` field
         * @return the composed stmts
         * */
        stmt get();
        basic_block_t() = default;
        basic_block_t(basic_block_t &&other) : body(std::move(other.body)) {}
    };

    /**
     * A builder may have nested basic blockes. It uses a stack to track
     * the current basic block.
     * */
    std::vector<basic_block_t> scopes;

    /**
     * Gets the current basic block on the top of the scopes
     * */
    basic_block_t &get_current_scope();
    builder_impl_t() = default;

    /**
     * Pushs a statement into the current basic block
     * */
    void emit(const stmt &s);

    /**
     * Pops out the current basic block. Generates a stmts node
     * from it
     * @return the stmts node for the poped basic block
     * */
    stmt pop_scope();

    /**
     * Pushes a empty current basic block on the top of the scopes.
     * emit() after calling this function will push the statements
     * into this new basic block
     * */
    void push_scope();

    /**
     * Makes and pushes an assign statement
     * @param var the dest LValue
     * @param value the src RValue
     * @return the pushed node
     * */
    stmt push_assign(const expr &var, const expr &value);

    /**
     * Makes and pushes an if_else
     * @param condition_ the condition, should be boolean typed
     * @param then_case_ the `then` block
     * @param else_case_ the `else` block, nullable.
     * @return the pushed node
     * */
    stmt push_if_else(const expr &condition_, const stmt &then_case_,
            const stmt &else_case_);

    /**
     * Makes and pushes an evaluate statement
     * @param val the expression
     * @return the pushed node
     * */
    stmt push_evaluate(const expr &val);

    /**
     * Makes and pushes an return statement
     * @param val the expression, nullable if the current function returns
     *      void_t
     * @return the pushed node
     * */
    stmt push_returns(const expr &val = expr());

    /**
     * Makes and push a variable def statement
     * @param var defined var or tensor
     * @param linkage
     * @param init init value, nullable
     * @return the pushed node
     * */
    stmt push_var_tensor_def(const expr &var, linkage linkage = linkage::local,
            const expr &init = expr());

    /**
     * Makes and pushes a for_loop
     * @param var the iterate variable. The loop-var is expected to
     *  be used only within the scope of the loop. Should be an integer var
     * @param iter_begin the initial value of var_
     * @param iter_end the max bound of the loop-var var_. Can never be reached
     * @param step the step of var_ in each iteration.
     * @param body the body of the loop
     * @param incremental if the loop-var var_ is incremental. Not currently
     *  used.
     * @param kind the kind of the loop. @see for_type
     * @param num_threads the number of threads in parallel-for
     * @return the pushed node
     * */
    stmt push_for_loop(const expr &var, const expr &iter_begin,
            const expr &iter_end, const expr &step, const stmt &body,
            bool incremental, for_type kind, int num_threads = 0);

    /**
     * Makes and pushes blank stmts as an anchor
     * */
    stmts push_anchor();

    /**
     * Makes a brgemm-call node and evaluates it
     * */
    stmt brgemm(const expr_c &a, const expr_c &b, const expr_c &c,
            const expr_c &blocks, const expr_c &M, const expr_c &N,
            const expr_c &K, const expr_c &lda, const expr_c &ldb,
            const expr_c &ldc, const expr_c &a_stride, const expr_c &b_stride,
            const std::vector<expr> &postops_data, const expr_c &c_buf,
            const expr_c &bd_mask_idx, const brgemm_args::extra_args_t &extras);

    /**
     * Makes a brgemm-call node and evaluates it
     * */
    stmt list_brgemm(const expr_c &a, const expr_c &b, const expr_c &c,
            const expr_c &blocks, const expr_c &M, const expr_c &N,
            const expr_c &K, const expr_c &lda, const expr_c &ldb,
            const expr_c &ldc, const expr_c &a_stride, const expr_c &b_stride,
            const expr_c &len, const std::vector<expr> &postops_data,
            const expr_c &c_buf, const expr_c &bd_mask_idx,
            const brgemm_args::extra_args_t &extras);

    /**
     * Makes a string simulated by tensor node in the current location
     * For debug use only (e.g. in fmtprint). It may have performance overhead
     * @param name the function name
     * @return the created node, should be a tensor
     * */
    expr make_str(const std::string &str);
};

/**
 * Gets the current thread-local builder. If you have never set
 * a builder by `set_current()`, this function will return nullptr
 * */
SC_INTERNAL_API builder_impl_t *get_current_builder();

/**
 * Sets the current thread-local builder.
 * @param b the pointer to the builder
 * */
SC_INTERNAL_API void set_current_builder(builder_impl_t *b);

/**
 * Makes a binary/logic/cmp node of the same type of original, with new LHS=l
 * and RHS=r. Copies attrs and other info from original
 * @return the created node
 * */
expr remake_binary(const expr_c &l, const expr_c &r, const expr_c &original);

/**
 * Makes a constant node of f32
 * @param val the value
 * @return the created node
 * */
expr make_constant(float val);

/**
 * Makes a constant node of s32
 * @param val the value
 * @return the created node
 * */
expr make_constant(int32_t val);

/**
 * Makes a constant node of index type
 * @param val the value
 * @return the created node
 * */
expr make_constant(uint64_t val);

/**
 * Makes a constant node of std::vector<union_val>
 * @param val the value
 * @return the created node
 * */
expr make_constant(const std::vector<union_val> &val, sc_data_type_t dtype);

/**
 * Makes a var node
 * @param type the type of the variable
 * @param name the name of the variable
 * @return the created node
 * */
expr make_var(sc_data_type_t type, const std::string &name);

/**
 * Makes a cast node
 * @param type the destination type of the casting
 * @param in the expression to convert
 * @return the created node
 * */
SC_INTERNAL_API expr make_cast(sc_data_type_t type, const expr_c &in);

/**
 * Makes a add (+) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_add(const expr_c &left, const expr_c &right);

/**
 * Makes a sub (-) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_sub(const expr_c &left, const expr_c &right);

/**
 * Makes a mul (*) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
SC_INTERNAL_API expr make_mul(const expr_c &left, const expr_c &right);

/**
 * Makes a div (div) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_div(const expr_c &left, const expr_c &right);

/**
 * Makes a mod (%) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_mod(const expr_c &left, const expr_c &right);

/**
 * Makes a min node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_min(const expr_c &left, const expr_c &right);

/**
 * Makes a reinterpret_cast node
 * @param v the value
 * @param dtype the target type, must have the same size of v
 * @return the created node
 * */
expr make_reinterpret(const expr_c &v, sc_data_type_t dtype);

/**
 * Makes a isnan(float) node
 * @param v the value
 * @return the created node
 * */
expr make_isnan(const expr_c &v);

/**
 * Makes a saturated cast node
 * @param v the value
 * @param dtype the target dtype, must have the same size of v
 * @return the created node
 * */
expr make_saturated_cast(const expr_c &v, sc_data_type_t dtype);

/**
 * Makes a round-and-cast node. The input should be a FP32 vector or scalar
 * type. The output should be S32 vector or scalar type.
 * @param v the value
 * @param dtype the target dtype, must have the same size of v
 * @return the created node
 * */
expr make_round_and_cast(const expr_c &v, sc_data_type_t dtype);

/**
 * Makes a max node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_max(const expr_c &left, const expr_c &right);

/**
 * Makes a abs node (absolute value)
 * @param v the input value
 * @return the created node
 * */
expr make_abs(const expr_c &v);

/**
 * Makes a round node
 * @param v the input value
 * @return the created node
 * */
expr make_round(const expr_c &v);

/**
 * Makes a floor node
 * @param v the input value
 * @return the created node
 * */
expr make_floor(const expr_c &v);

/**
 * Makes a ceil node
 * @param v the input value
 * @return the created node
 * */
expr make_ceil(const expr_c &v);

/**
 * Makes an exp node
 * @param v the input value
 * @return the created node
 * */
expr make_exp(const expr_c &v);

/**
 * Makes a log node
 * @param v the input value
 * @return the created node
 * */
expr make_log(const expr_c &v);

/**
 * Makes a erf node
 * @param v the input value
 * @return the created node
 * */
expr make_erf(const expr_c &v);

/**
 * Makes an sqrt node
 * @param v the input value
 * @return the created node
 * */
expr make_sqrt(const expr_c &v);

/**
 * Makes an rsqrt node
 * @param v the input value
 * @return the created node
 * */
expr make_rsqrt(const expr_c &v);

/**
 * Makes an reduce add node
 * @param v the input value
 * @return the created node
 * */
expr make_reduce_add(const expr_c &v);

/**
 * Makes an reduce max node
 * @param v the input value
 * @return the created node
 * */
expr make_reduce_max(const expr_c &v);

/**
 * Makes an reduce min node
 * @param v the input value
 * @return the created node
 * */
expr make_reduce_min(const expr_c &v);

/**
 * Makes an reduce mul node
 * @param v the input value
 * @return the created node
 * */
expr make_reduce_mul(const expr_c &v);

/**
 * Makes a broadcast node
 * @param v the input value
 * @param lanes the lanes of output value
 * @return the created node
 * */
expr make_broadcast(const expr_c &v, int lanes);

/**
 * Makes an fmadd node
 * @param v_a the first input value
 * @param v_b the second input value
 * @param v_c the third input value
 * @return the created node
 * */
expr make_fmadd(const expr_c &v_a, const expr_c &v_b, const expr_c &v_c);

/**
 * Makes an unpack_low node
 * Unpack and interleave elements from the low half of each 128-bit lane
 * in v_a and v_b
 *
 * _mm256_unpacklo_ps
 * DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
 *	dst[31:0] := src1[31:0]
 *  dst[63:32] := src2[31:0]
 *  dst[95:64] := src1[63:32]
 *  dst[127:96] := src2[63:32]
 *  RETURN dst[127:0]
 * }
 * dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
 * dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
 * dst[MAX:256] := 0
 *
 * @param v_a the first input value
 * @param v_b the second input value
 * @param elem_bits the bits of unpack element, e.g. f32=>32, s64=>64
 * @return the created node
 * */
expr make_unpack_low(const expr_c &v_a, const expr_c &v_b, int elem_bits = 32);

/**
 * Makes an unpack_high node
 * Unpack and interleave elements from the high half of each 128-bit lane
 * in v_a and v_b
 *
 * _mm256_unpackhi_ps
 * DEFINE INTERLEAVE_HIGH_DWORDS(src1[127:0], src2[127:0]) {
 *	dst[31:0] := src1[95:64]
 *	dst[63:32] := src2[95:64]
 *	dst[95:64] := src1[127:96]
 *	dst[127:96] := src2[127:96]
 *	RETURN dst[127:0]
 * }
 * dst[127:0] := INTERLEAVE_HIGH_DWORDS(a[127:0], b[127:0])
 * dst[255:128] := INTERLEAVE_HIGH_DWORDS(a[255:128], b[255:128])
 * dst[MAX:256] := 0
 *
 * @param v_a the first input value
 * @param v_b the second input value
 * @param elem_bits the lanes of unpack element, e.g. f32=>32, s64=>64
 * @return the created node
 * */
expr make_unpack_high(const expr_c &v_a, const expr_c &v_b, int elem_bits = 32);

/**
 * Makes a shuffle node
 * Shuffle elements in v_a and v_b within 128-bit lanes using the control in v_c
 *
 * _mm256_shuffle_ps
 * DEFINE SELECT4(src, control) {
 *	CASE(control[1:0]) OF
 *	0:	tmp[31:0] := src[31:0]
 *	1:	tmp[31:0] := src[63:32]
 *	2:	tmp[31:0] := src[95:64]
 *	3:	tmp[31:0] := src[127:96]
 *	ESAC
 *	RETURN tmp[31:0]
 * }
 * dst[31:0] := SELECT4(a[127:0], imm8[1:0])
 * dst[63:32] := SELECT4(a[127:0], imm8[3:2])
 * dst[95:64] := SELECT4(b[127:0], imm8[5:4])
 * dst[127:96] := SELECT4(b[127:0], imm8[7:6])
 * dst[159:128] := SELECT4(a[255:128], imm8[1:0])
 * dst[191:160] := SELECT4(a[255:128], imm8[3:2])
 * dst[223:192] := SELECT4(b[255:128], imm8[5:4])
 * dst[255:224] := SELECT4(b[255:128], imm8[7:6])
 * dst[MAX:256] := 0
 *
 * @param v_a the first input value
 * @param v_b the second input value
 * @param v_c the third input value
 * @param type_bits the number of bits of the data type you want to shuffle
 * @return the created node
 * */
expr make_shuffle(const expr_c &v_a, const expr_c &v_b, const int &v_c,
        const int &type_bits);

/**
 * Makes an permute node
 * Shuffle each 128-bits selected by v_c from v_a and v_b.
 *
 * _mm256_permute2f128_ps
 * DEFINE SELECT4(src1, src2, control) {
 *	CASE(control[1:0]) OF
 *	0:	tmp[127:0] := src1[127:0]
 *	1:	tmp[127:0] := src1[255:128]
 *	2:	tmp[127:0] := src2[127:0]
 *	3:	tmp[127:0] := src2[255:128]
 *	ESAC
 *	IF control[3]
 *		tmp[127:0] := 0
 *	FI
 *	RETURN tmp[127:0]
 * }
 * dst[127:0] := SELECT4(a[255:0], b[255:0], imm8[3:0])
 * dst[255:128] := SELECT4(a[255:0], b[255:0], imm8[7:4])
 * dst[MAX:256] := 0

 * @param v_a the first input value
 * @param v_b the second input value
 * @param v_c the third input value
 * @param type_bits the number of bits of the data type you want to permute
 * @return the created node
 * */
expr make_permute(const expr_c &v_a, const expr_c &v_b, const int &v_c,
        const int &type_bits = 128);

/**
 * Makes an permutexvar node
 * Using the corresponding bit in idx to Shuffle v.
 * eg:

 * _mm512_permutexvar_epi8
 * FOR j := 0 to 63
 *  i := j*8
 *  id := idx[i+5:i]*8
 *   dst[i+7:i] := a[id+7:id]
 * ENDFOR

 * @param idx the correspoding index
 * @param v the input value
 * @param lanes specify the lanes for permutex data. For example: if datatype
 is u8 and specify lanes is 8, which means you want to permutex 64bit data in
 v.
 * @return the created node
 * */
expr make_permutexvar(const expr_c &idx, const expr_c &v, const int lanes = 1);

/**
 * Insert the value into dst at the location specified by imm. Note that if the
 * data is more than 128bit, the first parameter needs to be twice the number of
 * bits of the second parameter.
 *
 * ep: _mm512_inserti32x8
 * Operation
 * dst[511:0] := a[511:0]
 * CASE imm8[0] OF
 * 0: dst[255:0] := b[255:0]
 * 1: dst[511:256] := b[255:0]
 * ESAC
 * dst[MAX:512] := 0

 * @param v_a the first input value
 * @param v_b the second input value
 * @param imm the location specified value, 0 or 1
 * @return the created node
 * */
expr make_insert(const expr_c &v_a, const expr_c &v_b, const int imm);

/**
 * Extract the value from input specified by imm.
 *
 * dst[7:0] := (a[127:0] >> (imm[3:0] * 8))[7:0]
 * dst[31:8] := 0

 * @param v_a the input value
 * @param imm the location specified value, 0 or 1
 * @param lanes specify the lanes for extracting data. For example: if datatype
 is u8 and specify lanes is 8, which means you want to extract 64bit data from
 v_a.
 * @return the created node
 * */
expr make_extract(const expr_c &v_a, const int imm, const int lanes = 1);

/**
 * Makes an gather node
 * Gather elements from memory.
 * dst = __mm512{*(addr + indices[0]), *(addr + indices[1]), ...}
 * @param addr the base addr value, usually has pointer dtype.
 * @param indices the index list, usually use a simd var.
 * @return the created node
 * */
expr make_gather(const expr_c &addr, const expr_c &indices);

/**
 * Makes a permutex2var node
 * @param v_a the first input value
 * @param v_b the second input value
 * @param v_c the third input value
 * @return the created node
 * */
expr make_permutex2var(const expr_c &v_a, const expr_c &v_b, const expr_c &v_c);

/**
 * Makes a cmp_eq (==) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_eq(const expr_c &left, const expr_c &right);

/**
 * Makes a cmp_ne (!=) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_ne(const expr_c &left, const expr_c &right);

/**
 * Makes a cmp_lt (<) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_lt(const expr_c &left, const expr_c &right);

/**
 * Makes a cmp_le (<=) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_le(const expr_c &left, const expr_c &right);

/**
 * Makes a cmp_gt (>) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_gt(const expr_c &left, const expr_c &right);

/**
 * Makes a cmp_ge (>=) node
 * @param left left hand side
 * @param right right hand side
 * @return the created node
 * */
expr make_cmp_ge(const expr_c &left, const expr_c &right);

/**
 * Makes a logic_and (&&) node
 * @param left left hand side, should be of boolean type
 * @param right right hand side, should be of boolean type
 * @return the created node
 * */
expr make_logic_and(const expr_c &left, const expr_c &right);

/**
 * Makes a logic_or (||) node
 * @param left left hand side, should be of boolean type
 * @param right right hand side, should be of boolean type
 * @return the created node
 * */
expr make_logic_or(const expr_c &left, const expr_c &right);

/**
 * Makes a shift left (<<) node
 * @param left left hand side, the number to be shift
 * @param bits the number of bits to shift
 * @return the created node
 * */
expr make_shl(const expr_c &left, const expr_c &bits);

/**
 * Makes a shift right (>>) node
 * @param left left hand side, the number to be shift
 * @param bits the number of bits to shift
 * @return the created node
 * */
expr make_shr(const expr_c &left, const expr_c &bits);

/**
 * Makes a int_and (&) node
 * @param left left hand side, should be of boolean type
 * @param right right hand side, should be of boolean type
 * @return the created node
 * */
expr make_int_and(const expr_c &left, const expr_c &right);

/**
 * Makes a int_or (|) node
 * @param left left hand side, should be of boolean type
 * @param right right hand side, should be of boolean type
 * @return the created node
 * */
expr make_int_or(const expr_c &left, const expr_c &right);

/**
 * Makes a int_xor (^) node
 * @param left left hand side, should be of boolean type
 * @param right right hand side, should be of boolean type
 * @return the created node
 * */
expr make_int_xor(const expr_c &left, const expr_c &right);

/**
 * Makes a node to read field from a struct.
 * @param in dynamic tensor, should be a tensor with u8 pointer type
 * @param struct_name name string of struct
 * @param field_name name enum of field
 * @return the created node
 * */
expr make_read_struct(const expr_c &in, const std::string &struct_name,
        const int &field_name);

/**
 * Makes a node to write a field to struct.
 * @param dyn_tsr dynamic tensor, should be a tensor with u8 pointer type
 * @param in data tensor, should be a tensor with pointer type
 * @param struct_name name string of struct
 * @param field_name name enum of field
 * @return the created node
 * */
expr make_write_struct(const expr_c &dyn_tsr, const expr_c &in,
        const std::string &struct_name, const int &field_name);

/**
 * Makes a node to get thread group id
 * @param par_for_level_id the level of parallel for
 * @return the created node
 * */
expr make_get_group_id(uint64_t par_for_level_id);

/**
 * Makes a node to get thread id in group. Specifying the (-1) group will return
 * the global thread id
 * @param par_for_level_id the level of parallel for
 * @return the created node
 * */
expr make_get_group_thread_id(int par_for_level_id);

/**
 * Makes a logic_not (!) node
 * @param in the input expression, should be of boolean type
 * @return the created node
 * */
expr make_logic_not(const expr_c &in);

/**
 * Makes a select (? :) node
 * @param cond the condition expression, should be of boolean type
 * @param l obtained value when condition is true
 * @param r obtained value when condition is false
 * @return the created node
 * */
expr make_select(const expr_c &cond, const expr_c &l, const expr_c &r);

/**
 * Makes a clip(in, clip_min, clip_max) node
 * clip(in, clip_min, clip_max) = in when clip_min <= in <= clip_max
 * clip(in, clip_min, clip_max) = clip_min when in < clip_min
 * clip(in, clip_min, clip_max) = clip_max when in > clip_max
 * @param in the input value to be clipped
 * @param xmin min value for clip
 * @param xmax max value for clip
 * @return the created node
 * */
expr make_clip(
        const expr_c &in, const expr_c &clip_min, const expr_c &clip_max);

/**
 * Makes a indexing node of multiple dimemsions
 * @param ptr the buffer expression, should be a tensor
 * @param idx the indices, should be integers
 * @param mask the loading mask, currently unused, nullable
 * @param length the vector length to index, or 1 as a scalar
 * @return the created node
 * */
expr make_indexing(const expr &ptr, const std::vector<expr> &idx,
        uint16_t length = 1, const expr &mask = expr());
expr make_indexing(const expr &ptr, std::initializer_list<expr> idx,
        uint16_t length = 1, const expr &mask = expr());

expr make_indexing(const expr_c &ptr, const std::vector<expr_c> &idx,
        uint16_t length = 1, const expr_c &mask = expr_c());

/**
 * Makes a indexing node of single dimemsion
 * @param ptr the buffer expression, should be a tensor
 * @param idx the index, should be an integer
 * @param mask the loading mask, currently unused, nullable
 * @param length the vector length to index, or 1 as a scalar
 * @return the created node
 * */
expr make_indexing(const expr_c &ptr, const expr_c &idx, uint16_t length = 1,
        const expr_c &mask = expr_c());

/**
 * Makes a call node
 * @param func the callee
 * @param args the arguments
 * @return the created node
 * */
expr make_call(const func_t &func, const std::vector<expr> &args);
expr make_call(const func_c &func, const std::vector<expr_c> &args);

/**
 * Makes a call node with given callee and arguments. Other members are copied
 * from `old`
 * @param func_t the callee
 * @param args the arguments
 * @param old the old call_node
 * @return the created node
 * */
expr remake_call(
        const func_t &func, const std::vector<expr> &args, const call_c &old);
expr remake_call(
        const func_c &func, const std::vector<expr_c> &args, const call_c &old);

/**
 * Makes a tensor node
 * @param name the name of the tensor
 * @param dims the dimensions, should be integers
 * @param dtype the elemente type of the tensor
 * @param addr_space the address space: CPU/Device
 * @param strides stride info for each dim (optional)
 * @return the created node
 * */
SC_INTERNAL_API expr make_tensor(const std::string &name,
        const std::vector<expr> &dims, sc_data_type_t dtype,
        address_space addrspace = address_space::automatic,
        const std::shared_ptr<static_data_t> &init_value = nullptr,
        const std::vector<expr> &strides = {});
SC_INTERNAL_API expr make_tensor(const std::string &name,
        const std::vector<expr_c> &dims, sc_data_type_t dtype,
        address_space addrspace = address_space::automatic,
        const std::shared_ptr<static_data_t> &init_value = nullptr,
        const std::vector<expr_c> &strides = {});
SC_INTERNAL_API expr make_tensor(const std::string &name,
        std::initializer_list<expr> dims, sc_data_type_t dtype,
        address_space addrspace = address_space::automatic,
        const std::shared_ptr<static_data_t> &init_value = nullptr,
        std::initializer_list<expr> strides = std::initializer_list<expr>());

/**
 * Makes a tensor node with user-defined stride
 * @param name the name of the tensor
 * @param dims the dimensions, should be integers
 * @param strides stride info for each dim
 * @param dtype the elemente type of the tensor
 * @param addr_space the address space: CPU/Device
 * @return the created node
 * */
expr make_stensor(const std::string &name, const std::vector<expr> &dims,
        const std::vector<expr> &strides, sc_data_type_t dtype,
        address_space addrspace = address_space::automatic,
        const std::shared_ptr<static_data_t> &init_value = nullptr);
expr make_stensor(const std::string &name, const std::vector<expr_c> &dims,
        const std::vector<expr_c> &strides, sc_data_type_t dtype,
        address_space addrspace = address_space::automatic,
        const std::shared_ptr<static_data_t> &init_value = nullptr);

/**
 * Makes a function node
 * @param name the function name
 * @param params the parameters, should be tensors or vars
 * @param body the body of the function
 * @param ret_type the return type of the function
 * @return the created node
 * */
SC_INTERNAL_API func_t make_func(const std::string &name,
        const std::vector<expr> &params, stmt body, sc_data_type_t ret_type);
/**
 * @see make_func overloaded function
 * */
SC_INTERNAL_API func_t make_func(const std::string &name,
        const std::vector<expr_c> &params, const stmt_c &body,
        sc_data_type_t ret_type);

/**
 * Makes an assign statement, the statement is not attached to any builder
 * @param var the dest LValue
 * @param value the src RValue
 * @return the pushed node
 * */
stmt make_assign_unattached(const expr_c &var, const expr_c &value);

/**
 * Makes an stmts statement, the statement is not attached to any builder
 * @param seq the sequence of the stmt
 * @return the pushed node
 * */
stmt make_stmts_unattached(const std::vector<stmt_c> &seq);

/**
 * Makes an if_else, the statement is not attached to any builder
 * @param condition_ the condition, should be boolean typed
 * @param then_case_ the `then` block
 * @param else_case_ the `else` block, nullable.
 * @return the pushed node
 * */
stmt make_if_else_unattached(const expr_c &condition, const stmt_c &then_case,
        const stmt_c &else_case);

/**
 * Makes an evaluate statement, the statement is not attached to any
 * builder
 * @param val the expression
 * @return the pushed node
 * */
stmt make_evaluate_unattached(const expr_c &val);

/**
 * Makes a return statement, the statement is not attached to any builder
 * @param val the expression, nullable if the current function returns
 *      void_t
 * @return the pushed node
 * */
stmt make_returns_unattached(const expr_c &val = expr_c());

/**
 * Makes a variable def statement, the statement is not attached to any builder
 * @param var defined var or tensor
 * @param linkage
 * @param init init value, nullable
 * @return the pushed node
 * */
SC_INTERNAL_API stmt make_var_tensor_def_unattached(const expr_c &var,
        linkage linkage = linkage::local, const expr_c &init = expr_c());

/**
 * Makes a for_loop, the statement is not attached to any builder
 * @param var the iterate variable. The loop-var is expected to
 *  be used only within the scope of the loop. Should be an integer var
 * @param iter_begin the initial value of var_
 * @param iter_end the max bound of the loop-var var_. Can never be reached
 * @param step the step of var_ in each iteration.
 * @param body the body of the loop
 * @param incremental if the loop-var var_ is incremental. Not currently
 *  used.
 * @param kind the kind of the loop. @see for_type
 * @param num_threads the number of threads to use in parallel-for. 0 for using
 * all avaliable threads in current thread group. If the loop is not parallel,
 * it must be 0.
 * @return the pushed node
 * */
stmt make_for_loop_unattached(const expr_c &var, const expr_c &iter_begin,
        const expr_c &iter_end, const expr_c &step, const stmt_c &body,
        bool incremental, for_type kind, int num_threads = 0);

// makes a new intrin_call with type_ and intrin_attrs_ copied
intrin_call remake_intrin_call(
        const intrin_call_c &v, const std::vector<expr> &newargs);

// makes a func_ptr
expr make_func_addr(func_t v);

/**
 * Makes a phi node
 * */
expr make_phi(const std::vector<expr> &values, bool is_loop_phi = false);

/**
 * Makes a x86 gerenal intrinsic node
 * */
expr make_x86_intrin(x86_intrin_type::x86_intrin_type_t type,
        const std::vector<expr> &args, const any_map_t &attrs = any_map_t());

// makes a new low_level_intrin with newargs and type_ copied
low_level_intrin remake_low_level_intrin(
        const low_level_intrin_c &v, const std::vector<expr> &newargs);

/**
 * Gets the pointer of an element of the tensor as a view
 * @see tensorptr_node
 * @param tensor the tensor
 * @param idx the indices of the element within the tensor
 * @param shape the shape of the resulting view. It can be empty, when
 * there is no indexing on the resulting view. If there is an indexing
 * on this tensor_ptr, the shape cannot be empty.
 * @param is_slice How `index_flatten` pass flatten the indexing on this
 * tensor view. For example, we have a tensorptr
 * `ptr=&base[offset0,offset1,offset2]`.If `is_slice` is true, indexing
 * on this tensor view `ptr[i,j,k]` will be first mapped to indexing on
 * the base tensor: `base[i+offset0, j+offset1, k+offset2]`, where
 * offsetN is the offset of this tensorptr to the base tensor. Then it
 * will flatten the mapped indexing. If `is_slice` is false,
 * `index_flatten` pass will treat the view as an independent tensor,
 * which shares a part of the memory from the base tensor. So the
 * indexing on view `ptr[i,j,k]` will be first flattened to `ptr[i * A +
 * j * B + k]`, where `A` and `B` depends on the `shape` field of this
 * IR node. Then it will add the flattened index `i * A + j * B + k` to
 * the flattened base offset of this tensor ptr `offset0 * C + offset1 *
 * D + offset2`, to finally compute the index on the base tensor.
 * @return the address of the element in the tensor
 * */
SC_INTERNAL_API expr tensor_ptr(const expr &tensor,
        const std::vector<expr> &idx, const std::vector<expr> &shape = {},
        bool is_slice = false);
SC_INTERNAL_API expr tensor_ptr(const expr_c &tensor,
        const std::vector<expr_c> &idx, const std::vector<expr_c> &shape = {},
        bool is_slice = false);
SC_INTERNAL_API expr tensor_ptr(const expr &tensor,
        std::initializer_list<expr> idx, std::initializer_list<expr> shape = {},
        bool is_slice = false);
/**
 * Sets the attr of the input IR node
 * @tparam TNode should be expr(_c) or stmt(_c)
 * @tparam T the value of the attr
 *
 * @param node the IR node
 * @param key the key name of the attr
 * @param value the value of the attr
 * @return node
 * */
template <typename TNode, typename T>
TNode with_attr(TNode node, const std::string &key, const T &value) {
    node->attr()[key] = value;
    return node;
}

/**
 * A builder with scope guard to help set the builder of the current
 * thread. It will set the current builder to `this` and remember the old
 * builder. After this builder is destructed, it will set the current builder
 * back to the old one. Using this builder, we can ensure that
 * builder::get_current_builder is a valid pointer or null.
 * @see builder_impl_t
 * @note Common usage:
 * {
 *      guarded_builder builder;
 *      // build the IR with builder
 * }
 * */
class ir_builder_t : public builder_impl_t {
    builder_impl_t *old_;

public:
    ir_builder_t(const ir_builder_t &) = delete;
    ir_builder_t() {
        old_ = get_current_builder();
        set_current_builder(this);
    }
    ~ir_builder_t() { set_current_builder(old_); }
};
} // namespace builder

#define _BUILDER_MAKE_BIN_OP(OP, NAME) \
    inline expr operator OP(const expr_c &l, const expr_c &r) { \
        return builder::make_##NAME(l, r); \
    }

_BUILDER_MAKE_BIN_OP(+, add)
_BUILDER_MAKE_BIN_OP(-, sub)
_BUILDER_MAKE_BIN_OP(*, mul)
_BUILDER_MAKE_BIN_OP(/, div)
_BUILDER_MAKE_BIN_OP(%, mod)
_BUILDER_MAKE_BIN_OP(==, cmp_eq)
_BUILDER_MAKE_BIN_OP(!=, cmp_ne)
_BUILDER_MAKE_BIN_OP(<, cmp_lt)
_BUILDER_MAKE_BIN_OP(<=, cmp_le)
_BUILDER_MAKE_BIN_OP(>, cmp_gt)
_BUILDER_MAKE_BIN_OP(>=, cmp_ge)
_BUILDER_MAKE_BIN_OP(&&, logic_and)
_BUILDER_MAKE_BIN_OP(||, logic_or)
_BUILDER_MAKE_BIN_OP(&, int_and)
_BUILDER_MAKE_BIN_OP(|, int_or)
_BUILDER_MAKE_BIN_OP(^, int_xor)
_BUILDER_MAKE_BIN_OP(<<, shl)
_BUILDER_MAKE_BIN_OP(>>, shr)

inline expr operator!(const expr_c &l) {
    return builder::make_logic_not(l);
}
#undef _BUILDER_MAKE_BIN_OP

expr copy_attr(const expr_base &ths, expr &&newexpr);
stmt copy_attr(const stmt_base_t &ths, stmt &&newstmt);
func_t copy_attr(const func_base &ths, func_t &&newfunc);

stmt get_parent_node(const stmt &node);
// If buffer is tptr, return base tensor, If buffer is tensor, return itself
tensor get_real_tensor(const expr &buffer);
// set base tensor of `tptr` with `tsr`
void set_base_tensor(expr &tptr, const expr &tsr);
void add_parent_node(const stmt &s, const stmt &ret);
stmt get_common_parent_node(const stmt &node1, const stmt &node2);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
