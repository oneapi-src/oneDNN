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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_PASSLET_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_PASSLET_HPP

#include <compiler/ir/sc_function.hpp>
#include <util/any_map.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace passlet {
enum pass_phase { PRE_VISIT, POST_VISIT };

#define SC_PASSLET_METHODS_IMPL(node_type, ...) \
    virtual void view(const node_type##_c &v, pass_phase phase);

#define SC_PASSLET_METHODS() \
    FOR_EACH_EXPR_IR_TYPE(SC_PASSLET_METHODS_IMPL) \
    FOR_EACH_STMT_IR_TYPE(SC_PASSLET_METHODS_IMPL) \
    FOR_EACH_BASE_EXPR_IR_TYPE(SC_PASSLET_METHODS_IMPL)

/**
 * Passlet base class. A passlet is a small plugin-able analysis pass which can
 * be inserted into an ir_viewer. ir_viewer can compose a list of passlets into
 * a big analysis pass. An analysis work written into a passlet can improve:
 * 1) reusablity, as a passlet can be easily reused by other passes
 * 2) performance, as a passlet don't need to dispatch the whole IR DAG
 *
 * A passlet uses user-provided result_addresser_t to return the analysis
 * result.
 * @see temp_data_addresser
 * @see map_addresser
 * */
struct passlet_t {
    using result_addresser_t
            = std::function<void *(passlet_t *ths, const node_base *v)>;

    SC_PASSLET_METHODS()

    virtual void view(const func_c &v, pass_phase phase);
    virtual void view(const expr_c &v, pass_phase phase);
    virtual void view(const stmt_c &v, pass_phase phase);

    result_addresser_t expr_result_func_;
    result_addresser_t stmt_result_func_;
    passlet_t(const result_addresser_t &expr_result_func,
            const result_addresser_t &stmt_result_func)
        : expr_result_func_(expr_result_func)
        , stmt_result_func_(stmt_result_func) {}
    virtual ~passlet_t() = default;
};

/**
 * The typed passlet.
 * @tparam T the analysis result
 * */
template <typename T>
struct typed_passlet : public passlet_t {
    using typed_addresser_t
            = std::function<T *(passlet_t *ths, const node_base *v)>;
    T *get_result(const expr_base *p) {
        return reinterpret_cast<T *>(expr_result_func_(this, p));
    }

    T *get_result(const stmt_base_t *p) {
        return reinterpret_cast<T *>(stmt_result_func_(this, p));
    }

    typed_passlet(const typed_addresser_t &expr_result_func,
            const typed_addresser_t &stmt_result_func)
        : passlet_t(expr_result_func, stmt_result_func) {}
};

/**
 * The passlet ananlysis result addresser. It will insert the result to the
 * expr/stmt's temp_data_.
 * @tparam T the type of the temp_data_
 * */
template <typename T>
struct temp_data_inserter {
    T *operator()(passlet_t *ths, const node_base *v) {
        auto &data = v->temp_data();
        if (!data.isa<T>()) { data = T(); }
        auto &ret = data.get<T>();
        return &ret;
    }
};

/**
 * The passlet ananlysis result addresser. It will insert the result to the
 * expr/stmt's temp_data_ as a field.
 * @tparam T the type of the temp_data_
 * @tparam TObj the analysis result type of the passlet
 * @tparam ptr the member pointer of TObj in struct T
 * */
template <typename T, typename TObj, TObj T::*ptr>
struct temp_data_addresser {
    TObj *operator()(passlet_t *ths, const node_base *v) {
        auto &data = v->temp_data();
        if (!data.isa<T>()) { data = T(); }
        auto &ret = data.get<T>();
        return &(ret.*ptr);
    }
};

namespace helper {

template <typename T, typename TObj>
struct temp_data_addresser_helper {
    using Base = T;
    using Obj = TObj;
};

template <typename T, typename TObj>
constexpr temp_data_addresser_helper<T, TObj> mk_helper(TObj T::*ptr) {
    return temp_data_addresser_helper<T, TObj> {};
};

#define sc_make_temp_data_addresser(PTR) \
    (temp_data_addresser<decltype(helper::mk_helper(PTR))::Base, \
            decltype(helper::mk_helper(PTR))::Obj, (PTR)>())

} // namespace helper

template <typename T>
struct key_converter {
    static T convert(const node_base *v) { return static_cast<T>(v); }
};

template <typename T1, typename Base>
struct key_converter<node_ptr<T1, Base>> {
    static node_ptr<T1, Base> convert(const node_base *v) {
        return static_cast<const Base *>(v)->node_ptr_from_this();
    }
};

/**
 * The passlet ananlysis result addresser. It will insert the result to an STL
 * map.
 * @tparam M the map type. Key can be `const expr_node*`,`const stmt_node*`,
 * `stmt_c`, `expr_c`
 * */
template <typename M>
struct map_addresser {
    using key_type = typename M::key_type;
    using value_type = typename M::mapped_type;
    M &map_;
    map_addresser(M &map) : map_(map) {}
    value_type *operator()(passlet_t *ths, const node_base *v) {
        auto &data = map_[key_converter<key_type>::convert(v)];
        return &data;
    }
};
} // namespace passlet

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
