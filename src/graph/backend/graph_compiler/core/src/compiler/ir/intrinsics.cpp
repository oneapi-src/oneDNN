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

#include "intrinsics.hpp"
#include <memory>
#include <string>
#include <utility>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
intrinsic_handler_t::intrinsic_handler_t(const std::string &name)
    : name_(name) {}

x86_intrinsic_handler_t::x86_intrinsic_handler_t(const std::string &name)
    : intrinsic_handler_t(name) {}

struct binary_intrinsic_handler_t : public intrinsic_handler_t {
    binary_intrinsic_handler_t(const std::string &name)
        : intrinsic_handler_t(name) {}
    void on_initialize(intrin_call_node &node) override;
};

void binary_intrinsic_handler_t::on_initialize(intrin_call_node &node) {
    assert(node.args_.size() == 2);
    auto &l = node.args_[0];
    auto &r = node.args_[1];
    node.dtype_ = l->dtype_ == r->dtype_ ? l->dtype_ : datatypes::undef;
}

struct trinary_intrinsic_handler_t : public intrinsic_handler_t {
    trinary_intrinsic_handler_t(const std::string &name)
        : intrinsic_handler_t(name) {}
    void on_initialize(intrin_call_node &node) override;
};

void trinary_intrinsic_handler_t::on_initialize(intrin_call_node &node) {
    assert(node.args_.size() == 3);
    auto &a = node.args_[0];
    auto &b = node.args_[1];
    auto &c = node.args_[2];
    if (node.type_ == intrin_type::permute
            || node.type_ == intrin_type::shuffle) {
        node.dtype_ = a->dtype_ == b->dtype_ ? a->dtype_ : datatypes::undef;
    } else if (node.type_ == intrin_type::permutex2var) {
        node.dtype_ = a->dtype_ == c->dtype_ ? a->dtype_ : datatypes::undef;
    } else {
        node.dtype_ = a->dtype_ == b->dtype_ && a->dtype_ == c->dtype_
                ? a->dtype_
                : datatypes::undef;
    }
}

struct min_handler_t : public binary_intrinsic_handler_t {
    min_handler_t() : binary_intrinsic_handler_t("min") {}
};

struct max_handler_t : public binary_intrinsic_handler_t {
    max_handler_t() : binary_intrinsic_handler_t("max") {}
};

struct abs_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    abs_handler_t() : intrinsic_handler_t("abs") {}
};

struct round_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    round_handler_t() : intrinsic_handler_t("round") {}
};

struct ceil_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    ceil_handler_t() : intrinsic_handler_t("ceil") {}
};

struct floor_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    floor_handler_t() : intrinsic_handler_t("floor") {}
};

struct exp_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    exp_handler_t() : intrinsic_handler_t("exp") {}
};

struct log_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    log_handler_t() : intrinsic_handler_t("log") {}
};

struct erf_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    erf_handler_t() : intrinsic_handler_t("erf") {}
};

struct sqrt_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    sqrt_handler_t() : intrinsic_handler_t("sqrt") {}
};

struct rsqrt_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    rsqrt_handler_t() : intrinsic_handler_t("rsqrt") {}
};

struct reduce_add_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_add_handler_t() : intrinsic_handler_t("reduce_add") {}
};

struct reduce_mul_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_mul_handler_t() : intrinsic_handler_t("reduce_mul") {}
};

struct reduce_max_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_max_handler_t() : intrinsic_handler_t("reduce_max") {}
};

struct reduce_min_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = 1;
    }
    reduce_min_handler_t() : intrinsic_handler_t("reduce_min") {}
};

struct broadcast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        auto lanes = node.intrin_attrs_->get<int>("lanes");
        COMPILE_ASSERT(lanes <= 512, "Expecting lanes<=512");
        node.dtype_ = node.args_[0]->dtype_;
        node.dtype_.lanes_ = lanes;
    }
    broadcast_handler_t() : intrinsic_handler_t("broadcast") {}
};

struct fmadd_handler_t : public trinary_intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        node.dtype_ = node.args_[0]->dtype_;
    }
    fmadd_handler_t() : trinary_intrinsic_handler_t("fmadd") {}
};

struct unpack_low_handler_t : public binary_intrinsic_handler_t {
    unpack_low_handler_t() : binary_intrinsic_handler_t("unpack_low") {}
};

struct unpack_high_handler_t : public binary_intrinsic_handler_t {
    unpack_high_handler_t() : binary_intrinsic_handler_t("unpack_high") {}
};

struct shuffle_handler_t : public binary_intrinsic_handler_t {
    shuffle_handler_t() : binary_intrinsic_handler_t("shuffle") {}
};

struct permute_handler_t : public binary_intrinsic_handler_t {
    permute_handler_t() : binary_intrinsic_handler_t("permute") {}
};

struct reinterpret_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>(
                intrin_attr::out_dtype);
    }
    reinterpret_handler_t() : intrinsic_handler_t("reinterpret") {}
};

struct permutex2var_handler_t : public trinary_intrinsic_handler_t {
    permutex2var_handler_t() : trinary_intrinsic_handler_t("permutex2var") {}
};

struct permutexvar_handler_t : public binary_intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        COMPILE_ASSERT(node.args_.size() == 2, "Expecting 2 args.");
        node.dtype_ = node.args_[1]->dtype_;
    }
    permutexvar_handler_t() : binary_intrinsic_handler_t("permutexvar") {}
};

struct insert_handler_t : public trinary_intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        node.dtype_ = node.args_[0]->dtype_;
    }
    insert_handler_t() : trinary_intrinsic_handler_t("insert") {}
};

struct extract_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = sc_data_type_t(node.args_[0]->dtype_.type_code_);
        if (node.intrin_attrs_->get<int>("lanes") > 1) {
            node.dtype_.lanes_ = node.intrin_attrs_->get<int>("lanes");
        }
    }
    extract_handler_t() : intrinsic_handler_t("extract") {}
};

struct gather_handler_t : public binary_intrinsic_handler_t {
    gather_handler_t() : binary_intrinsic_handler_t("gather") {}
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_[1]->dtype_.is_etype(sc_data_etype::S32));
        node.dtype_ = node.args_[0]->dtype_.get_pointer_element();
        node.dtype_.lanes_ = node.args_[1]->dtype_.lanes_;
    }
};

struct round_and_cast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>(
                intrin_attr::out_dtype);
        COMPILE_ASSERT(node.dtype_.lanes_ == node.args_[0]->dtype_.lanes_
                        && node.dtype_.type_code_ == sc_data_etype::S32
                        && node.args_[0]->dtype_.type_code_
                                == sc_data_etype::F32,
                "round_and_cast cannot handle " << node.args_[0]->dtype_ << "->"
                                                << node.dtype_);
    }
    round_and_cast_handler_t() : intrinsic_handler_t("round_and_cast") {}
};

struct isnan_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = sc_data_type_t::boolean(node.dtype_.lanes_);
    }
    isnan_handler_t() : intrinsic_handler_t("isnan") {}
};

struct saturated_cast_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>("out_dtype");
    }
    saturated_cast_handler_t() : intrinsic_handler_t("saturated_cast") {}
};

struct shl_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        node.dtype_ = node.args_[0]->dtype_;
    }
    shl_handler_t() : intrinsic_handler_t("shl") {}
};

struct shr_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 2);
        node.dtype_ = node.args_[0]->dtype_;
    }
    shr_handler_t() : intrinsic_handler_t("shr") {}
};

struct int_and_handler_t : public binary_intrinsic_handler_t {
    int_and_handler_t() : binary_intrinsic_handler_t("int_and") {}
};

struct int_or_handler_t : public binary_intrinsic_handler_t {
    int_or_handler_t() : binary_intrinsic_handler_t("int_or") {}
};

struct int_xor_handler_t : public binary_intrinsic_handler_t {
    int_xor_handler_t() : binary_intrinsic_handler_t("int_xor") {}
};

sc_data_type_t get_dtype_from_struct_and_field(
        const std::string &in, int field) {
    if (in == dyn_tsr_struct_t::name) {
        return dyn_tsr_struct_t::dtypes[field];
    } else {
        COMPILE_ASSERT(false, "struct " << in << " has not been supported!");
    }
    return sc_data_type_t();
}

struct read_struct_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        assert(node.intrin_attrs_->has_key(intrin_attr::struct_name)
                && node.intrin_attrs_->has_key(intrin_attr::struct_field));
        node.dtype_ = get_dtype_from_struct_and_field(
                node.intrin_attrs_->get<std::string>(intrin_attr::struct_name),
                node.intrin_attrs_->get<int>(intrin_attr::struct_field));
    }
    read_struct_handler_t() : intrinsic_handler_t("read_struct") {}
};

struct write_struct_handler_t : public binary_intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.intrin_attrs_->has_key(intrin_attr::struct_name)
                && node.intrin_attrs_->has_key(intrin_attr::struct_field));
        node.dtype_ = datatypes::void_t;
    }
    write_struct_handler_t() : binary_intrinsic_handler_t("write_struct") {}
};

struct brgemm_handler_t : public intrinsic_handler_t {
    size_t arg_cnt_;
    void on_initialize(intrin_call_node &node) override {
        assert(node.check_brgemm_arg_size(arg_cnt_));
        node.dtype_ = datatypes::void_t;
    }
    brgemm_handler_t(size_t arg_cnt, const char *name)
        : intrinsic_handler_t(name), arg_cnt_(arg_cnt) {}
};

struct set_thread_idle_func_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        node.dtype_ = datatypes::void_t;
        COMPILE_ASSERT(node.args_.size() >= 2UL,
                "set_thread_idle_func requires more than 2 args");
        COMPILE_ASSERT(node.args_[0]->dtype_ == datatypes::pointer,
                "The first arg of set_thread_idle_func should be pointer");
    }
    set_thread_idle_func_handler_t()
        : intrinsic_handler_t("set_thread_idle_func") {}
};

struct prefetch_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        node.dtype_ = datatypes::void_t;
        COMPILE_ASSERT(node.args_.size() == 1, "prefetch requires 1 arg");
        COMPILE_ASSERT(node.args_[0]->dtype_.is_pointer(),
                "The first arg of prefetch should be pointer");
        auto locality = node.intrin_attrs_->get_or_else("locality", -1);
        COMPILE_ASSERT(locality >= 0 && locality <= 3,
                "locality attr of prefetch must be between 0 to 3");
    }
    prefetch_handler_t() : intrinsic_handler_t("prefetch") {}
};

struct load_const_mem_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.args_[0]->dtype_;
    }
    load_const_mem_handler_t() : intrinsic_handler_t("load_const_mem") {}
};

struct get_group_id_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = datatypes::u32;
    }
    get_group_id_handler_t() : intrinsic_handler_t("get_group_id") {}
};

struct get_group_thread_id_handler_t : public intrinsic_handler_t {
    void on_initialize(intrin_call_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = datatypes::s32;
    }
    get_group_thread_id_handler_t()
        : intrinsic_handler_t("get_group_thread_id") {}
};

struct avx_broadcast_idx_handler_t : public x86_intrinsic_handler_t {
    void on_initialize(low_level_intrin_node &node) override {
        assert(node.args_.size() == 3);
        assert(node.args_[0]->dtype_.is_pointer());
        auto lanes = node.intrin_attrs_->get<int>("lanes");
        node.dtype_ = sc_data_type_t(
                etypes::get_pointer_element(node.args_[0]->dtype_.type_code_),
                lanes);
    }
    avx_broadcast_idx_handler_t()
        : x86_intrinsic_handler_t("avx_broadcast_idx") {}
};

struct avx_mask_cast_handler_t : public x86_intrinsic_handler_t {
    void on_initialize(low_level_intrin_node &node) override {
        assert(node.args_.size() == 1);
        node.dtype_ = node.intrin_attrs_->get<sc_data_type_t>("dtype");
    }
    avx_mask_cast_handler_t() : x86_intrinsic_handler_t("avx_mask_cast") {}
};

struct avx_compare_handler_t : public x86_intrinsic_handler_t {
    void on_initialize(low_level_intrin_node &node) override {
        assert(node.args_.size() == 3);
        assert(node.args_[2].isa<constant>());
        assert(node.args_[0]->dtype_ == node.args_[1]->dtype_);
        node.dtype_ = node.args_[0]->dtype_;
    }
    avx_compare_handler_t() : x86_intrinsic_handler_t("avx_compare") {}
};

namespace brgemm_args {
sc_data_type_t arg_types[NUM_FULL_ARGS_STRIDE] = {
        datatypes::pointer, // A (overloaded)
        datatypes::pointer, // B
        datatypes::pointer, // C
        datatypes::s32, // num
        datatypes::s32, // M
        datatypes::s32, // N
        datatypes::s32, // K
        datatypes::s32, // LDA
        datatypes::s32, // LDB
        datatypes::s32, // LDC
        datatypes::s32, // stride_a
        datatypes::s32, // stride_b
        datatypes::pointer, // bias
        datatypes::pointer, // scales
        datatypes::pointer, // binary_post_ops_rhs
        datatypes::index, // oc_logical_off
        datatypes::index, // dst_row_logical_off
        datatypes::pointer, // data_C_ptr
        datatypes::index, // first_mb_matrix_addr_off
        datatypes::pointer, // a_zp_compensations
        datatypes::pointer, // b_zp_compensations
        datatypes::pointer, // c_zp_values
        datatypes::boolean, // skip_accumulation
        datatypes::s32, // zp_a_val
        datatypes::boolean, // do_only_comp
        datatypes::boolean, // do_only_zp_a_val
        datatypes::pointer, // c_buf
        datatypes::index // bdmask_idx
};

sc_data_type_t list_arg_types[NUM_FULL_ARGS_LIST] = {
        datatypes::pointer, // A
        datatypes::pointer, // B
        datatypes::pointer, // C
        datatypes::s32, // num
        datatypes::s32, // M
        datatypes::s32, // N
        datatypes::s32, // K
        datatypes::s32, // LDA
        datatypes::s32, // LDB
        datatypes::s32, // LDC
        datatypes::s32, // stride_a
        datatypes::s32, // stride_b
        datatypes::s32, // len
        datatypes::pointer, // bias
        datatypes::pointer, // scales
        datatypes::pointer, // binary_post_ops_rhs
        datatypes::index, // oc_logical_off
        datatypes::index, // dst_row_logical_off
        datatypes::pointer, // data_C_ptr
        datatypes::index, // first_mb_matrix_addr_off
        datatypes::pointer, // a_zp_compensations
        datatypes::pointer, // b_zp_compensations
        datatypes::pointer, // c_zp_values
        datatypes::boolean, // skip_accumulation
        datatypes::s32, // zp_a_val
        datatypes::boolean, // do_only_comp
        datatypes::boolean, // do_only_zp_a_val
        datatypes::pointer, // c_buf
        datatypes::index // bdmask_idx
};
} // namespace brgemm_args

static std::unique_ptr<intrinsic_handler_t> handlers[] = {
        utils::make_unique<min_handler_t>(),
        utils::make_unique<max_handler_t>(),
        utils::make_unique<abs_handler_t>(),
        utils::make_unique<round_handler_t>(),
        utils::make_unique<floor_handler_t>(),
        utils::make_unique<ceil_handler_t>(),
        utils::make_unique<exp_handler_t>(),
        utils::make_unique<log_handler_t>(),
        utils::make_unique<erf_handler_t>(),
        utils::make_unique<sqrt_handler_t>(),
        utils::make_unique<rsqrt_handler_t>(),
        utils::make_unique<reduce_add_handler_t>(),
        utils::make_unique<reduce_mul_handler_t>(),
        utils::make_unique<reduce_max_handler_t>(),
        utils::make_unique<reduce_min_handler_t>(),
        utils::make_unique<fmadd_handler_t>(),
        utils::make_unique<unpack_low_handler_t>(),
        utils::make_unique<unpack_high_handler_t>(),
        utils::make_unique<shuffle_handler_t>(),
        utils::make_unique<permute_handler_t>(),
        utils::make_unique<int_and_handler_t>(),
        utils::make_unique<int_or_handler_t>(),
        utils::make_unique<int_xor_handler_t>(),
        utils::make_unique<reinterpret_handler_t>(),
        utils::make_unique<broadcast_handler_t>(),
        utils::make_unique<isnan_handler_t>(),
        utils::make_unique<saturated_cast_handler_t>(),
        utils::make_unique<round_and_cast_handler_t>(),
        utils::make_unique<shl_handler_t>(),
        utils::make_unique<shr_handler_t>(),
        utils::make_unique<permutex2var_handler_t>(),
        utils::make_unique<permutexvar_handler_t>(),
        utils::make_unique<insert_handler_t>(),
        utils::make_unique<extract_handler_t>(),
        utils::make_unique<gather_handler_t>(),
        utils::make_unique<read_struct_handler_t>(),
        utils::make_unique<write_struct_handler_t>(),
        utils::make_unique<set_thread_idle_func_handler_t>(),
        utils::make_unique<prefetch_handler_t>(),
        utils::make_unique<load_const_mem_handler_t>(),
        utils::make_unique<get_group_id_handler_t>(),
        utils::make_unique<get_group_thread_id_handler_t>(),
        utils::make_unique<brgemm_handler_t>(
                brgemm_args::NUM_FULL_ARGS_STRIDE, "brgemm"),
        utils::make_unique<brgemm_handler_t>(
                brgemm_args::NUM_FULL_ARGS_LIST, "list_brgemm"),
};

static_assert(sizeof(handlers) / sizeof(handlers[0])
                == int(intrin_type::NUM_INTRINSICS),
        "Not all intrinsics are filled in handlers");

intrinsic_handler_t &get_intrinsic_handler(intrin_type intrin) {
    return *handlers[static_cast<int>(intrin)];
}

static std::unique_ptr<x86_intrinsic_handler_t> x86_handlers[] = {
        utils::make_unique<avx_broadcast_idx_handler_t>(),
        utils::make_unique<avx_mask_cast_handler_t>(),
        utils::make_unique<avx_compare_handler_t>(),
};
x86_intrinsic_handler_t &get_x86_intrinsic_handler(int64_t intrin) {
    return *x86_handlers[intrin];
}

static_assert(sizeof(x86_handlers) / sizeof(x86_handlers[0])
                == int(x86_intrin_type::NUM_INTRINSICS),
        "Not all intrinsics are filled in x86 handlers");

#define OFFSET(struct_type, field) \
    (size_t)(&(((struct_type *)nullptr)->field)) // NOLINT

const sc_data_type_t dyn_tsr_struct_t::dtypes[5] = {datatypes::pointer,
        datatypes::pointer, datatypes::s32, datatypes::u32, datatypes::u8};
const size_t dyn_tsr_struct_t::offsets[5]
        = {OFFSET(runtime::dynamic_tensor_t, data_),
                OFFSET(runtime::dynamic_tensor_t, dims_),
                OFFSET(runtime::dynamic_tensor_t, ndims_),
                OFFSET(runtime::dynamic_tensor_t, dtype_),
                OFFSET(runtime::dynamic_tensor_t, dyn_mask_)};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
