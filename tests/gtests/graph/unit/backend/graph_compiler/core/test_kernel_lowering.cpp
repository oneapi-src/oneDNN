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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/cpu/kernel_lower.hpp>

#include <iostream>
#include "test_utils.hpp"
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;

brgemm::attrs_setting_t::attrs_map_t attr_0
        = {brgemm::attr_key::max_top_vpad, 2};
brgemm::attrs_setting_t::attrs_map_t attr_1
        = {brgemm::attr_key::hint_expected_A_size, 1024};
brgemm::attrs_setting_t::attrs_map_t attr_2
        = {brgemm::attr_key::hint_expected_B_size, 1024};
brgemm::attrs_setting_t::attrs_map_t attr_3
        = {brgemm::attr_key::bd_mask_level, 2};
brgemm::attrs_setting_t::attrs_map_t range_attr_0
        = {brgemm::attr_key::M_range_upper_bound, 64};
brgemm::attrs_setting_t::attrs_map_t range_attr_1
        = {brgemm::attr_key::N_range_upper_bound, 64};
brgemm::attrs_setting_t::attrs_map_t range_attr_2
        = {brgemm::attr_key::K_range_upper_bound, 64};
brgemm::attrs_setting_t::attrs_map_t range_attr_3
        = {brgemm::attr_key::M_range_tail_value, 2};
sc_brgemm_bd_mask_t bd_mask_0 {1, 0};
sc_brgemm_bd_mask_t bd_mask_1 {0, 1};

int shape_0[2] = {2, 4};
int shape_1[2] = {3, 4};
brgemm::postop_setting_t postop_0((brgemm::bias_op_t(sc_data_etype::F32)));
brgemm::postop_setting_t postop_1 = brgemm::scale_op_t();
brgemm::postop_setting_t postop_2(brgemm::bin_op_t(
        brgemm::alg_kind_t::binary_add, shape_0, sc_data_etype::F32));
brgemm::postop_setting_t postop_3(brgemm::bin_op_t(
        brgemm::alg_kind_t::binary_add, shape_1, sc_data_etype::F32));

#define DEFINE_ATTRS_BDMSK_TENSORS(mod) \
    size_t attrs_size = sizeof(brgemm::attrs_setting_t::attrs_map_t) * 3 \
            + sizeof(int64_t); \
    std::vector<char> attrs_data(attrs_size, 0); \
    brgemm::attrs_setting_t *attrs_ptr \
            = reinterpret_cast<brgemm::attrs_setting_t *>(attrs_data.data()); \
    attrs_ptr->num_ = 3; \
    attrs_ptr->map_[0] = attr_0; \
    attrs_ptr->map_[1] = attr_1; \
    attrs_ptr->map_[2] = attr_3; \
    _module_tensor_(mod, attrs_tsr, datatypes::u8, {attrs_size}); \
    attrs_tsr.get().checked_as<tensor>()->init_value_ \
            = std::make_shared<static_data_t>(attrs_ptr, attrs_size); \
    _module_tensor_(mod, bd_mask_tsr, datatypes::u8, {bd_mask.size()}); \
    bd_mask_tsr.get().checked_as<tensor>()->init_value_ \
            = std::make_shared<static_data_t>(bd_mask.data(), bd_mask.size());

#define DEFINE_ATTRS_MODULE_TENSORS(num) \
    _module_tensor_(m2, attrs_tsr_##num, datatypes::u8, {attrs_sz_##num}); \
    attrs_tsr_##num.get().checked_as<tensor>()->init_value_ \
            = std::make_shared<static_data_t>( \
                    attrs_ptr_##num, attrs_sz_##num); \
    _module_tensor_( \
            m2, bd_mask_tsr_##num, datatypes::u8, {bd_mask_##num.size()}); \
    bd_mask_tsr_##num.get().checked_as<tensor>()->init_value_ \
            = std::make_shared<static_data_t>(bd_mask_##num.data(), 2); \
    _module_tensor_(m2, bd_mask_arr_##num, datatypes::pointer, {1}); \
    builder::make_assign_unattached( \
            builder::make_indexing(bd_mask_arr_##num, {0}), \
            builder::tensor_ptr(bd_mask_tsr_##num, {0})); \
    _module_tensor_( \
            m2, postop_set_tsr_##num, datatypes::u8, {postop_set_sz_##num}); \
    postop_set_tsr_##num.get().checked_as<tensor>()->init_value_ \
            = std::make_shared<static_data_t>( \
                    postop_set_ptr_##num, postop_set_sz_##num);

#define DEFINE_REF_MODULE_TENSOR_0() \
    size_t attrs_sz_0 = sizeof(brgemm::attrs_setting_t::attrs_map_t) * 3 \
            + sizeof(int64_t); \
    std::vector<char> attrs_data_0(attrs_sz_0, 0); \
    brgemm::attrs_setting_t *attrs_ptr_0 \
            = reinterpret_cast<brgemm::attrs_setting_t *>( \
                    attrs_data_0.data()); \
    attrs_ptr_0->num_ = 3; \
    attrs_ptr_0->map_[0] = attr_0; \
    attrs_ptr_0->map_[1] = attr_1; \
    attrs_ptr_0->map_[2] = attr_3; \
    size_t postop_set_sz_0 \
            = sizeof(brgemm::postop_setting_t) * postop_set_0.size() \
            + sizeof(int64_t); \
    std::vector<char> postop_set_data_0(postop_set_sz_0, 0); \
    brgemm::postops_setting_t *postop_set_ptr_0 \
            = reinterpret_cast<brgemm::postops_setting_t *>( \
                    postop_set_data_0.data()); \
    postop_set_ptr_0->num_ = 3; \
    postop_set_ptr_0->ops_[0] = postop_0; \
    postop_set_ptr_0->ops_[1] = postop_1; \
    postop_set_ptr_0->ops_[2] = postop_2; \
    DEFINE_ATTRS_MODULE_TENSORS(0);

#define DEFINE_REF_MODULE_TENSOR_1() \
    size_t attrs_sz_1 = sizeof(brgemm::attrs_setting_t::attrs_map_t) * 3 \
            + sizeof(int64_t); \
    std::vector<char> attrs_data_1(attrs_sz_1, 0); \
    brgemm::attrs_setting_t *attrs_ptr_1 \
            = reinterpret_cast<brgemm::attrs_setting_t *>( \
                    attrs_data_1.data()); \
    attrs_ptr_1->num_ = 3; \
    attrs_ptr_1->map_[0] = attr_0; \
    attrs_ptr_1->map_[1] = attr_2; \
    attrs_ptr_1->map_[2] = attr_3; \
    size_t postop_set_sz_1 \
            = sizeof(brgemm::postop_setting_t) * postop_set_1.size() \
            + sizeof(int64_t); \
    std::vector<char> postop_set_data_1(postop_set_sz_1, 0); \
    brgemm::postops_setting_t *postop_set_ptr_1 \
            = reinterpret_cast<brgemm::postops_setting_t *>( \
                    postop_set_data_1.data()); \
    postop_set_ptr_1->num_ = 3; \
    postop_set_ptr_1->ops_[0] = postop_0; \
    postop_set_ptr_1->ops_[1] = postop_3; \
    postop_set_ptr_1->ops_[2] = postop_1; \
    DEFINE_ATTRS_MODULE_TENSORS(1);

#define DEFINE_POSTOP_DATA_TENSORS(num) \
    _tensor_(bias_##num, datatypes::f32, {100}); \
    _tensor_(scales_##num, datatypes::f32, {100}); \
    _tensor_(binary_tsr_##num, datatypes::f32, {50});
#define POSTOP_DATA_INIT(dst_num, src_num) \
    _tensor_(binary_rhs_ptr_##dst_num, datatypes::pointer, {1}); \
    binary_rhs_ptr_##dst_num[UINT64_C(0)] = binary_tsr_##src_num; \
    _tensor_(postop_data_##dst_num, datatypes::u8, \
            {brgemm::postops_data_size}); \
    _evaluate_call_(postop_data_init, postop_data_##dst_num, bias_##src_num, \
            scales_##src_num, binary_rhs_ptr_##dst_num, ir_zero, ir_zero, \
            ir_nullptr, ir_zero, ir_nullptr, ir_nullptr, ir_nullptr, ir_false, \
            ir_zero_s32, ir_false, ir_false);

TEST(GCCore_CPU_kernel_lowering_cpp, TestKernelLowering) {
    builder::ir_builder_t builder;
    sc_brgemm_attrs_t attrs_0 = {attr_0, attr_1, attr_3};
    sc_brgemm_postops_setting_t postop_set_0 = {postop_0, postop_1, postop_2};
    sc_brgemm_attrs_t attrs_1 = {attr_0, attr_2, attr_3};
    sc_brgemm_postops_setting_t postop_set_1 = {postop_0, postop_3, postop_1};
    sc_brgemm_attrs_t attrs_2 = {range_attr_0};
    _function_(datatypes::void_t, aaa, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        DEFINE_POSTOP_DATA_TENSORS(0);
        DEFINE_POSTOP_DATA_TENSORS(1);
        _tensor_(c_buf, datatypes::f32, {100});

        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        // same parameters as above, there should only be one kernel entry
        builtin::brgemm_init_update(A, B, C, 2, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        _var_(c, datatypes::s32);
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16, attrs_2);

        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);

        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);

        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        // different parameters from above, there should only be 2 different
        // entries
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 10,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);

        auto postop_data_0 = builtin::create_initialed_postops_data();
        postop_data_0[brgemm::postop_data_kind::bias] = bias_0;
        postop_data_0[brgemm::postop_data_kind::scales] = scales_0;
        postop_data_0[brgemm::postop_data_kind::binary_post_ops_rhs]
                = binary_tsr_0;
        auto postop_data_1 = builtin::create_initialed_postops_data();
        postop_data_1[brgemm::postop_data_kind::bias] = bias_1;
        postop_data_1[brgemm::postop_data_kind::scales] = scales_1;
        postop_data_1[brgemm::postop_data_kind::binary_post_ops_rhs]
                = binary_tsr_1;
        expr brg_c_buf = c_buf;
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16, attrs_0, bd_mask_0, 0, 1,
                postop_set_0, postop_data_0, brg_c_buf);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16, attrs_1, bd_mask_1, 0, 1,
                postop_set_1, postop_data_1);
        ///////////// list calls
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_list_update(A, B, C, 1, 2, c, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_list_update(A, B, C, 1, 2, c, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16, datatypes::bf16, attrs_0, bd_mask_0, 0, 1,
                postop_set_0, postop_data_0, brg_c_buf);
    }
    auto ctx = std::make_shared<context_t>(*get_default_context());
    ctx->flags_.brgemm_backend_ = scflags_t::brgemm_t::dnnl;
    auto m = ir_module_t::from_entry_func(ctx, aaa);
    auto res = kernel_lowering_cpu_t(true)(m);
    scflags_t::brgemm_t backend = scflags_t::brgemm_t::dnnl;
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);
    expr ir_zero = make_expr<constant_node>(0UL, datatypes::index);
    expr ir_zero_s32 = make_expr<constant_node>(0UL, datatypes::s32);
    expr ir_false = make_expr<constant_node>(0UL, datatypes::boolean);
    func_t strd_call, strd_create, strd_init_update, strd_update, ptr_call,
            ptr_create, ptr_init_update, ptr_update;
    func_t strd_call_postop, strd_create_postop, ptr_call_postop,
            ptr_create_postop;
    std::tie(strd_create, strd_call)
            = builtin::get_brgemm_creator_and_call_func(
                    builtin::brgemm_mode::stride, backend, false);
    std::tie(ptr_create, ptr_call) = builtin::get_brgemm_creator_and_call_func(
            builtin::brgemm_mode::addr_list, backend, false);

    std::tie(strd_create_postop, strd_call_postop)
            = builtin::get_brgemm_creator_and_call_func(
                    builtin::brgemm_mode::stride, backend, true);
    std::tie(ptr_create_postop, ptr_call_postop)
            = builtin::get_brgemm_creator_and_call_func(
                    builtin::brgemm_mode::addr_list, backend, true);

    std::tie(strd_update, strd_init_update) = builtin::get_brgemm_update_funcs(
            builtin::brgemm_mode::stride, backend);
    std::tie(ptr_update, ptr_init_update) = builtin::get_brgemm_update_funcs(
            builtin::brgemm_mode::addr_list, backend);
    /////////////////// expected
    auto m2 = std::make_shared<ir_module_t>(ctx);

    func_t postop_data_init = builtin::get_brgemm_postops_data_init_func();
    _module_var_(m2, kernel1, datatypes::pointer,
            strd_create(2, 3, 4, 5, 6, 7, 8, 9, 0.0f,
                    datatypes::bf16.as_etype_int(),
                    datatypes::bf16.as_etype_int(), ir_nullptr, ir_nullptr,
                    ir_nullptr));
    _module_var_(m2, kernel2, datatypes::pointer,
            strd_create(2, 3, 4, 5, 6, 7, 8, 9, 1.0f,
                    datatypes::bf16.as_etype_int(),
                    datatypes::bf16.as_etype_int(), ir_nullptr, ir_nullptr,
                    ir_nullptr));
    _module_var_(m2, kernel5, datatypes::pointer,
            strd_create(2, 3, 4, 5, 6, 7, 8, 10, 1.0f,
                    datatypes::bf16.as_etype_int(),
                    datatypes::bf16.as_etype_int(), ir_nullptr, ir_nullptr,
                    ir_nullptr));
    DEFINE_REF_MODULE_TENSOR_0();
    _module_tensor_(m2, kernel6, datatypes::pointer, {1});
    auto cachev = strd_create_postop(2, 3, 4, 5, 6, 7, 8, 9, 0.0f,
            datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
            attrs_tsr_0, bd_mask_arr_0[0], postop_set_tsr_0);
    builder::make_assign_unattached(
            builder::make_indexing(kernel6, {0}), cachev);

    DEFINE_REF_MODULE_TENSOR_1();
    _module_tensor_(m2, kernel7, datatypes::pointer, {1});
    cachev = strd_create_postop(2, 3, 4, 5, 6, 7, 8, 9, 1.0f,
            datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
            attrs_tsr_1, bd_mask_arr_1[0], postop_set_tsr_1);
    builder::make_assign_unattached(
            builder::make_indexing(kernel7, {0}), cachev);

    _module_var_(m2, list_k1, datatypes::pointer,
            ptr_create(2, 3, 4, 5, 6, 7, 1.0f, datatypes::bf16.as_etype_int(),
                    datatypes::bf16.as_etype_int(), ir_nullptr, ir_nullptr,
                    ir_nullptr));

    _module_tensor_(m2, list_k2, datatypes::pointer, {1});
    cachev = ptr_create_postop(2, 3, 4, 5, 6, 7, 1.0f,
            datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
            attrs_tsr_0, bd_mask_arr_0[0], postop_set_tsr_0);
    builder::make_assign_unattached(
            builder::make_indexing(list_k2, {0}), cachev);

    _function_(datatypes::void_t, expected, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        DEFINE_POSTOP_DATA_TENSORS(0);
        DEFINE_POSTOP_DATA_TENSORS(1);
        _tensor_(c_buf, datatypes::f32, {100});

        _evaluate_call_(strd_call, kernel1, A, B, C, 1, ir_nullptr);
        _evaluate_call_(strd_call, kernel1, A, B, C, 2, ir_nullptr);
        _var_(c, datatypes::s32);
        _evaluate_call_(strd_init_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);

        _evaluate_call_(strd_call, kernel2, A, B, C, 1, ir_nullptr);
        _evaluate_call_(strd_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);

        _evaluate_call_(strd_call, kernel1, A, B, C, 1, ir_nullptr);
        _evaluate_call_(strd_init_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);

        _evaluate_call_(strd_call, kernel2, A, B, C, 1, ir_nullptr);
        _evaluate_call_(strd_call, kernel5, A, B, C, 1, ir_nullptr);
        _evaluate_call_(strd_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);

        POSTOP_DATA_INIT(0, 0);
        _evaluate_call_(strd_call_postop, kernel6[0], A, B, C, 1, postop_data_0,
                c_buf, ir_nullptr);
        POSTOP_DATA_INIT(1, 1);
        _tensor_(c_buf_1, datatypes::f32, {expr(2) * 3});
        _evaluate_call_(strd_call_postop, kernel7[0], A, B, C, 1, postop_data_1,
                c_buf_1, ir_nullptr);

        _evaluate_call_(ptr_call, list_k1, A, B, C, 1, 8, 9, 10,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr);
        _evaluate_call_(ptr_update, A, B, C, 1, 2, c, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);
        _evaluate_call_(ptr_call, list_k1, A, B, C, 1, 8, 9, 10,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr);
        _evaluate_call_(ptr_update, A, B, C, 1, 2, c, 4, 5, 6, 7, 8, 9, 10,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);
        POSTOP_DATA_INIT(2, 0);
        _evaluate_call_(ptr_call_postop, list_k2[0], A, B, C, 1, 8, 9, 10,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                postop_data_2, c_buf, ir_nullptr);
    }
    m2->add_func({expected});
    ir_comparer cmp(true);

    ASSERT_TRUE(m2->get_module_vars().size() == res->get_module_vars().size());
    for (unsigned i = 0; i < m2->get_module_vars().size(); i++) {
        EXPECT_TRUE(cmp.compare(
                m2->get_module_vars()[i], res->get_module_vars()[i]));
    }
    EXPECT_TRUE(
            cmp.compare(m2->get_contents()[0], res->get_contents()[0], false));
}

TEST(GCCore_CPU_kernel_lowering_cpp, TestKernelLoweringNoOptim) {
    builder::ir_builder_t builder;
    sc_brgemm_attrs_t attrs_0 = {attr_0, attr_1, attr_3};
    sc_brgemm_postops_setting_t postop_set_0 = {postop_0, postop_1, postop_2};
    sc_brgemm_attrs_t attrs_1 = {attr_0, attr_2, attr_3};
    sc_brgemm_postops_setting_t postop_set_1 = {postop_0, postop_3, postop_1};
    _function_(datatypes::void_t, aaa, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        DEFINE_POSTOP_DATA_TENSORS(0);
        DEFINE_POSTOP_DATA_TENSORS(1);
        _tensor_(c_buf, datatypes::f32, {100});
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);
        _var_(c, datatypes::s32);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16);

        auto postop_data_0 = builtin::create_initialed_postops_data();
        postop_data_0[brgemm::postop_data_kind::bias] = bias_0;
        postop_data_0[brgemm::postop_data_kind::scales] = scales_0;
        postop_data_0[brgemm::postop_data_kind::binary_post_ops_rhs]
                = binary_tsr_0;
        auto postop_data_1 = builtin::create_initialed_postops_data();
        postop_data_1[brgemm::postop_data_kind::bias] = bias_1;
        postop_data_1[brgemm::postop_data_kind::scales] = scales_1;
        postop_data_1[brgemm::postop_data_kind::binary_post_ops_rhs]
                = binary_tsr_1;
        expr brg_c_buf = c_buf;
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16, datatypes::bf16, attrs_0, bd_mask_0, 0, 1,
                postop_set_0, postop_data_0, brg_c_buf);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16, datatypes::bf16, attrs_1, bd_mask_1, 0, 1,
                postop_set_1, postop_data_1);
        ///////////// list calls
        builtin::brgemm_list_update(A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::bf16, datatypes::bf16);
        builtin::brgemm_list_update(A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::bf16, datatypes::bf16, attrs_0, bd_mask_0, 0, 1,
                postop_set_0, postop_data_0, brg_c_buf);
    }
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);
    expr ir_zero = make_expr<constant_node>(0UL, datatypes::index);
    expr ir_zero_s32 = make_expr<constant_node>(0UL, datatypes::s32);
    expr ir_false = make_expr<constant_node>(0UL, datatypes::boolean);
    auto m = ir_module_t::from_entry_func(get_default_context(), aaa);
    auto res = kernel_lowering_cpu_t(false)(m);
    auto backend = get_default_context()->flags_.brgemm_backend_;
    func_t strd_init_update, strd_update, ptr_init_update, ptr_update;
    std::tie(strd_update, strd_init_update) = builtin::get_brgemm_update_funcs(
            builtin::brgemm_mode::stride, backend);
    std::tie(ptr_update, ptr_init_update) = builtin::get_brgemm_update_funcs(
            builtin::brgemm_mode::addr_list, backend);
    /////////////////// expected
    auto m2 = std::make_shared<ir_module_t>(get_default_context());

    func_t postop_data_init = builtin::get_brgemm_postops_data_init_func();
    DEFINE_REF_MODULE_TENSOR_0();
    DEFINE_REF_MODULE_TENSOR_1();
    _function_(datatypes::void_t, expected, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        DEFINE_POSTOP_DATA_TENSORS(0);
        DEFINE_POSTOP_DATA_TENSORS(1);
        _tensor_(c_buf, datatypes::f32, {100});
        _evaluate_call_(strd_init_update, A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);
        _var_(c, datatypes::s32);
        _evaluate_call_(strd_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);
        POSTOP_DATA_INIT(0, 0);
        _evaluate_call_(strd_init_update, A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                attrs_tsr_0, bd_mask_arr_0[0], postop_set_tsr_0, postop_data_0,
                c_buf, ir_nullptr);
        POSTOP_DATA_INIT(1, 1);
        _tensor_(c_buf_1, datatypes::f32, {expr(2) * 3});
        _evaluate_call_(strd_update, A, B, C, 1, 2, 3, 4, 5, c, 7, 8, 9,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                attrs_tsr_1, bd_mask_arr_1[0], postop_set_tsr_1, postop_data_1,
                c_buf_1, ir_nullptr);

        _evaluate_call_(ptr_update, A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr, ir_nullptr,
                ir_nullptr);
        POSTOP_DATA_INIT(2, 0);
        _evaluate_call_(ptr_update, A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::bf16.as_etype_int(), datatypes::bf16.as_etype_int(),
                attrs_tsr_0, bd_mask_arr_0[0], postop_set_tsr_0, postop_data_2,
                c_buf, ir_nullptr);
    }
    m2->add_func({expected});
    ir_comparer cmp(true);
    ASSERT_TRUE(m2->get_module_vars().size() == res->get_module_vars().size());
    for (unsigned i = 0; i < m2->get_module_vars().size(); i++) {
        EXPECT_TRUE(cmp.compare(
                m2->get_module_vars()[i], res->get_module_vars()[i]));
    }
    EXPECT_TRUE(
            cmp.compare(m2->get_contents()[0], res->get_contents()[0], false));
}

TEST(GCCore_CPU_kernel_lowering_cpp, TestBrgemmAttrs) {
    builder::ir_builder_t builder;
    sc_brgemm_attrs_t attrs = {attr_0, attr_1, attr_3};
    sc_brgemm_bd_mask_t bd_mask {1, 0};
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);

    _function_(datatypes::void_t, tested_func, {}) {
        _tensor_(A, datatypes::s8, {100});
        _tensor_(B, datatypes::s8, {100});
        _tensor_(C, datatypes::s32, {100});
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                datatypes::s8, datatypes::s8, attrs, bd_mask);
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                datatypes::s8, datatypes::s8);
    }
    auto ctx = std::make_shared<context_t>(*get_default_context());
    scflags_t::brgemm_t backend = scflags_t::brgemm_t::dnnl;
    ctx->flags_.brgemm_backend_ = backend;
    auto tested_mod = ir_module_t::from_entry_func(ctx, tested_func);
    auto tested = kernel_lowering_cpu_t(true)(tested_mod);

    auto m2 = std::make_shared<ir_module_t>(ctx);
    func_t ptr_create, ptr_call;
    std::tie(ptr_create, ptr_call) = builtin::get_brgemm_creator_and_call_func(
            builtin::brgemm_mode::addr_list, backend, false);

    DEFINE_ATTRS_BDMSK_TENSORS(m2);
    _module_tensor_(m2, bd_mask_arr, datatypes::pointer, {1});

    auto cachev2 = ptr_create(2, 3, 4, 5, 6, 7, 1.0f,
            datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
            attrs_tsr, bd_mask_arr[0], ir_nullptr);
    auto cachev1 = ptr_create(2, 3, 4, 5, 6, 7, 1.0f,
            datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
            ir_nullptr, ir_nullptr, ir_nullptr);
    _module_tensor_(m2, list_k, datatypes::pointer, {1});
    _module_var_(m2, k, datatypes::pointer, cachev1);

    _function_(datatypes::void_t, expected_func, {}) {
        _tensor_(A, datatypes::s8, {100});
        _tensor_(B, datatypes::s8, {100});
        _tensor_(C, datatypes::s32, {100});

        _evaluate_call_(ptr_call, list_k[0UL], A, B, C, 1, 8, 9, 10,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                ir_nullptr);
        _evaluate_call_(ptr_call, k, A, B, C, 1, 8, 9, 10,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                ir_nullptr);
    }

    _function_(datatypes::void_t, init_func, {}) {
        k = cachev1;
        bd_mask_arr[0] = builder::tensor_ptr(bd_mask_tsr, {0 * expr(2)});
        list_k[0] = cachev2;
    }
    m2->add_func({expected_func});
    m2->add_func({init_func});

    ir_comparer cmp(true);
    ASSERT_TRUE(
            m2->get_module_vars().size() == tested->get_module_vars().size());
    for (size_t i = 0; i < m2->get_module_vars().size(); i++) {
        EXPECT_TRUE(cmp.compare(
                m2->get_module_vars()[i], tested->get_module_vars()[i]));
    }

    for (size_t i = 0; i < m2->get_contents().size(); ++i) {
        EXPECT_TRUE(cmp.compare(
                m2->get_contents()[i], tested->get_contents()[i], false));
    }
}

TEST(GCCore_CPU_kernel_lowering_cpp, TestBrgemmSharedBdmask) {
    builder::ir_builder_t builder;
    sc_brgemm_attrs_t attrs = {attr_0, attr_1, attr_3};
    sc_brgemm_bd_mask_t bd_mask {1, 1, 0, 0};
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);

    _function_(datatypes::void_t, tested_func, {}) {
        _tensor_(A, datatypes::s8, {100});
        _tensor_(B, datatypes::s8, {100});
        _tensor_(C, datatypes::s32, {100});
        builtin::brgemm_list_update(A, B, C, 1, 4, 6, 4, 4, 6, 4, 8, 9, 2,
                datatypes::s8, datatypes::s8, attrs, bd_mask, 0, 1);
        builtin::brgemm_list_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2,
                datatypes::s8, datatypes::s8, attrs, bd_mask, 0, 2);
    }
    auto ctx = std::make_shared<context_t>(*get_default_context());
    scflags_t::brgemm_t backend = scflags_t::brgemm_t::dnnl;
    ctx->flags_.brgemm_backend_ = backend;
    auto tested_mod = ir_module_t::from_entry_func(ctx, tested_func);
    auto tested = kernel_lowering_cpu_t(true)(tested_mod);

    // expected
    auto m2 = std::make_shared<ir_module_t>(ctx);
    func_t ptr_create, ptr_call;
    std::tie(ptr_create, ptr_call) = builtin::get_brgemm_creator_and_call_func(
            builtin::brgemm_mode::addr_list, backend, false);

    DEFINE_ATTRS_BDMSK_TENSORS(m2);

    _module_tensor_(m2, bd_mask_arr_1, datatypes::pointer, {1});
    _module_tensor_(m2, list_k1, datatypes::pointer, {1});
    auto cachev1 = ptr_create(4, 6, 4, 4, 6, 4, 1.0f,
            datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
            attrs_tsr, bd_mask_arr_1[0], ir_nullptr);
    _module_tensor_(m2, bd_mask_arr_2, datatypes::pointer, {2});
    _module_tensor_(m2, list_k2, datatypes::pointer, {2});

    _function_(datatypes::void_t, expected_func, {}) {
        _tensor_(A, datatypes::s8, {100});
        _tensor_(B, datatypes::s8, {100});
        _tensor_(C, datatypes::s32, {100});

        _evaluate_call_(ptr_call, list_k1[0], A, B, C, 1, 8, 9, 2,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                ir_nullptr);
        _evaluate_call_(ptr_call, list_k2[0], A, B, C, 1, 8, 9, 2,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                ir_nullptr);
    }

    _function_(datatypes::void_t, init_func, {}) {
        bd_mask_arr_1[0] = builder::tensor_ptr(bd_mask_tsr, {0 * expr(4)});
        bd_mask_arr_2[0] = builder::tensor_ptr(bd_mask_tsr, {0 * expr(2)});
        bd_mask_arr_2[1] = builder::tensor_ptr(bd_mask_tsr, {1 * expr(2)});
        list_k1[0] = cachev1;
        list_k2[0] = ptr_create(2, 3, 4, 5, 6, 7, 1.0f,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                attrs_tsr, bd_mask_arr_2[0], ir_nullptr);
        list_k2[1] = ptr_create(2, 3, 4, 5, 6, 7, 1.0f,
                datatypes::s8.as_etype_int(), datatypes::s8.as_etype_int(),
                attrs_tsr, bd_mask_arr_2[1], ir_nullptr);
    }
    m2->add_func({expected_func});
    m2->add_func({init_func});

    ir_comparer cmp(true);
    ASSERT_TRUE(
            m2->get_module_vars().size() == tested->get_module_vars().size());
    for (size_t i = 0; i < m2->get_module_vars().size(); i++) {
        EXPECT_TRUE(cmp.compare(
                m2->get_module_vars()[i], tested->get_module_vars()[i]));
    }

    for (size_t i = 0; i < m2->get_contents().size(); ++i) {
        EXPECT_TRUE(cmp.compare(
                m2->get_contents()[i], tested->get_contents()[i], false));
    }
}

TEST(GCCore_CPU_kernel_lowering_cpp, TestRangeKernelLowering) {
    REQUIRE_AVX2();
    auto backend = get_default_context()->flags_.brgemm_backend_;
    if (backend != scflags_t::brgemm_t::dnnl) { GTEST_SKIP(); }
    builder::ir_builder_t builder;
    sc_brgemm_attrs_t attrs_0 = {range_attr_0, range_attr_3};
    sc_brgemm_attrs_t attrs_1 = {range_attr_0, range_attr_1, range_attr_2};
    _function_(datatypes::void_t, aaa, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        builtin::brgemm_init_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::f32, datatypes::f32, attrs_0);
        builtin::brgemm_update(A, B, C, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                datatypes::f32, datatypes::f32, attrs_1);
        ///////////// list calls
        builtin::brgemm_init_list_update(A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::f32, datatypes::f32, attrs_1);
        builtin::brgemm_list_update(A, B, C, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1,
                datatypes::f32, datatypes::f32, attrs_0);
    }
    expr ir_nullptr = make_expr<constant_node>(0UL, datatypes::pointer);
    auto m = ir_module_t::from_entry_func(get_default_context(), aaa);
    auto res = kernel_lowering_cpu_t(true)(m);
    func_t strd_range_func, list_range_func;
    strd_range_func
            = builtin::get_brgemm_call_range_func(builtin::brgemm_mode::stride);
    list_range_func = builtin::get_brgemm_call_range_func(
            builtin::brgemm_mode::addr_list);
    /////////////////// expected
    auto m2 = std::make_shared<ir_module_t>(get_default_context());

    _module_var_(
            m2, handle_0, datatypes::pointer, res->get_module_vars()[0]->init_);
    _module_var_(
            m2, handle_1, datatypes::pointer, res->get_module_vars()[1]->init_);
    _module_var_(
            m2, handle_2, datatypes::pointer, res->get_module_vars()[2]->init_);
    _module_var_(
            m2, handle_3, datatypes::pointer, res->get_module_vars()[3]->init_);

    _function_(datatypes::void_t, expected, {}) {
        _tensor_(A, datatypes::f32, {100});
        _tensor_(B, datatypes::f32, {100});
        _tensor_(C, datatypes::f32, {100});
        _evaluate_call_(
                strd_range_func, handle_0, 2, 3, 4, A, B, C, 1, ir_nullptr);
        _evaluate_call_(
                strd_range_func, handle_1, 2, 3, 4, A, B, C, 1, ir_nullptr);
        _evaluate_call_(list_range_func, handle_2, 3, 4, 5, A, B, C, 2, 9, 10,
                1, datatypes::f32.as_etype_int(), datatypes::f32.as_etype_int(),
                ir_nullptr);
        _evaluate_call_(list_range_func, handle_3, 3, 4, 5, A, B, C, 2, 9, 10,
                1, datatypes::f32.as_etype_int(), datatypes::f32.as_etype_int(),
                ir_nullptr);
    }
    m2->add_func({expected});
    ir_comparer cmp(true);
    ASSERT_TRUE(m2->get_module_vars().size() == res->get_module_vars().size());
    for (unsigned i = 0; i < m2->get_module_vars().size(); i++) {
        EXPECT_TRUE(cmp.compare(
                m2->get_module_vars()[i], res->get_module_vars()[i]));
    }
    EXPECT_TRUE(
            cmp.compare(m2->get_contents()[0], res->get_contents()[0], false));
}
