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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_DETAIL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_TENSOR_DETAIL_HPP

#include <algorithm>
#include <functional>

#include <iosfwd>
#include <vector>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/sc_data_type.hpp>
#include <compiler/ir/sc_expr.hpp>
#include <unordered_set>
#include <util/assert.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace reflection {
template <typename T, typename Dummy>
struct type_registry;
}
class sc_graph_t;
struct SC_INTERNAL_API logical_tensor_t {
public:
    sc_data_type_t dtype_;
    logical_tensor_t() = default;

    bool operator==(const logical_tensor_t &other) const {
        return format_ == other.format_ && dtype_ == other.dtype_
                && plain_dims_ == other.plain_dims_
                && strides_ == other.strides_;
    }

    bool operator!=(const logical_tensor_t &other) const {
        return !(*this == other);
    }

    logical_tensor_t(const sc_data_format_t &format, const sc_dims &plain_dims,
            const sc_data_type_t &type, const sc_dims &strides = {})
        : dtype_(type)
        , format_(format)
        , plain_dims_(plain_dims)
        , strides_(strides) {
        internal_update();
    }

    // gets the dims, taking blocking into consideration, using cache
    // If the shape is dynamic, this function should not be used in lowering.
    // Use `sc_data_format_t::get_blocking_shapes_expr(plain_shape, format)`
    // instead.
    const sc_dims &get_blocking_dims() const;
    // gets the blocking dims exprs, we could not just use
    // graph.dims_to_expr(dims_) because of may generate block and num of block
    // of dynamic var, we should calculate by
    // sc_data_format_t::get_blocking_dims_expr();
    std::vector<expr> get_blocking_dims_expr(sc_graph_t &g) const;
    // sets the logical dims in blocking format
    void set_blocking_dims(const sc_dims &blocking_dims);
    // gets the logical dims in plain format
    const sc_dims &get_plain_dims() const { return plain_dims_; }
    // gets strides corresponding to each blocking dim
    const sc_dims &get_strides() const { return strides_; }
    // In dynamic, strides should be compute by compute_dense_stride_expr()
    std::vector<expr> get_strides_expr(sc_graph_t &) const;
    // sets the logical dims in plain format
    void set_plain_dims(const sc_dims &plain_dims);
    // gets the data format
    const sc_data_format_t &get_format() const { return format_; }
    // sets the data format and invalidate the cached blocking_dims
    void set_format(const sc_data_format_t &newv);
    const std::unordered_set<sc_data_format_t> &get_format_candidates() const {
        return format_candidates_;
    }
    // add a format to candidates, if exists in candidates, do nothing.
    void add_format_candidate(const sc_data_format_t &newv);
    // remove format candidate by format.
    void remove_format_candidate(const sc_data_format_t &v);
    // sets the format candidates and update the format_ if number of candidates
    // == 1
    void set_format_candidates(const std::vector<sc_data_format_t> &newf);
    // gets the size of the tensor in bytes
    size_t get_blocking_byte_size() const;
    // if the tensor is dynamic
    bool is_dynamic() const;
    // sets the strides
    void set_strides(const sc_dims &strides);
    // sets the data format and stride
    void set_format_and_stride(
            const sc_data_format_t &newv, const sc_dims &strides);
    // judge whether the current logical tensor is dense
    bool is_dense();
    // print the tensor detail to string
    void to_string(std::ostream &os);
    // used to compute dense stride based on dims
    static sc_dims compute_dense_stride(const sc_dims &dims);
    static std::vector<expr> compute_dense_stride_expr(
            sc_graph_t &graph, const std::vector<expr> &dims);
    size_t hash() const;

private:
    template <typename T, typename Dummy>
    friend struct reflection::type_registry;
    friend struct std::hash<dnnl::impl::graph::gc::logical_tensor_t>;
    // definite format for internal sync.
    sc_data_format_t format_;
    // The real dims, which may be blocking.
    sc_dims dims_;
    // the logical dims in plain format
    sc_dims plain_dims_;
    // strides corresponding to each dim of `dims_`, always dense in dynamic
    sc_dims strides_;
    // dynamic shape may has several format/stride pair in graph.
    std::unordered_set<sc_data_format_t> format_candidates_;
    // sync real dims based on plain dims and format
    void internal_update();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::gc::logical_tensor_t> {
    std::size_t operator()(
            const dnnl::impl::graph::gc::logical_tensor_t &k) const;
};
} // namespace std
#endif
