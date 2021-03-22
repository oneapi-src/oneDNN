/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef COMMON_IMPL_LIST_ITEM_HPP
#define COMMON_IMPL_LIST_ITEM_HPP

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct impl_list_item_t {
    impl_list_item_t() = delete;
    impl_list_item_t(const impl_list_item_t &other) = default;
    impl_list_item_t(impl_list_item_t &&other) = default;
    impl_list_item_t &operator=(const impl_list_item_t &other) = default;
    impl_list_item_t &operator=(impl_list_item_t &&other) = default;

    impl_list_item_t(std::nullptr_t) {}

    template <typename pd_t>
    struct type_deduction_helper_t {
        type_deduction_helper_t() : pd(nullptr) {
            static_assert(std::is_base_of<primitive_desc_t, pd_t>::value,
                    "type_deduction_helper_t is expected to be used for "
                    "primitive descriptor classes only.");
        }
        pd_t *pd;
    };

    template <typename pd_t>
    struct concat_type_deduction_helper_t
        : public type_deduction_helper_t<pd_t> {};

    template <typename pd_t>
    struct sum_type_deduction_helper_t : public type_deduction_helper_t<pd_t> {
    };

    template <typename pd_t>
    impl_list_item_t(type_deduction_helper_t<pd_t> helper) {
        using deduced_pd_t =
                typename std::remove_pointer<decltype(helper.pd)>::type;
        create_pd_func_ = &primitive_desc_t::create<deduced_pd_t>;
    }

    template <typename pd_t>
    impl_list_item_t(concat_type_deduction_helper_t<pd_t> helper) {
        using deduced_pd_t =
                typename std::remove_pointer<decltype(helper.pd)>::type;
        create_concat_pd_func_ = deduced_pd_t::create;
    }

    template <typename pd_t>
    impl_list_item_t(sum_type_deduction_helper_t<pd_t> helper) {
        using deduced_pd_t =
                typename std::remove_pointer<decltype(helper.pd)>::type;
        create_sum_pd_func_ = deduced_pd_t::create;
    }

    explicit operator bool() const {
        return !utils::everyone_is(nullptr, create_pd_func_,
                create_concat_pd_func_, create_sum_pd_func_);
    }

    status_t operator()(primitive_desc_t **pd, const op_desc_t *adesc,
            const primitive_attr_t *attr, engine_t *engine,
            const primitive_desc_t *hint_fwd) const {
        assert(create_pd_func_);
        if (!create_pd_func_) return status::runtime_error;
        return create_pd_func_(pd, adesc, attr, engine, hint_fwd);
    }

    status_t operator()(concat_pd_t **concat_pd, engine_t *engine,
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
            int concat_dim, const memory_desc_t *src_mds) const {
        assert(create_concat_pd_func_);
        if (!create_concat_pd_func_) return status::runtime_error;
        return create_concat_pd_func_(
                concat_pd, engine, attr, dst_md, n, concat_dim, src_mds);
    }

    status_t operator()(sum_pd_t **sum_pd, engine_t *engine,
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
            const float *scales, const memory_desc_t *src_mds) const {
        assert(create_sum_pd_func_);
        if (!create_sum_pd_func_) return status::runtime_error;
        return create_sum_pd_func_(
                sum_pd, engine, attr, dst_md, n, scales, src_mds);
    }

    using create_pd_func_t = status_t (*)(primitive_desc_t **,
            const op_desc_t *, const primitive_attr_t *, engine_t *,
            const primitive_desc_t *);

    using create_concat_pd_func_t
            = status_t (*)(concat_pd_t **, engine_t *, const primitive_attr_t *,
                    const memory_desc_t *, int, int, const memory_desc_t *);

    using create_sum_pd_func_t = status_t (*)(sum_pd_t **, engine_t *,
            const primitive_attr_t *, const memory_desc_t *, int, const float *,
            const memory_desc_t *);

    create_pd_func_t create_pd_func_ = nullptr;
    create_concat_pd_func_t create_concat_pd_func_ = nullptr;
    create_sum_pd_func_t create_sum_pd_func_ = nullptr;
};

} // namespace impl
} // namespace dnnl

#endif
