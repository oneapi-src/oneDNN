/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef BENCHDNN_GRAPH_REF_PRIMITIVE_HPP
#define BENCHDNN_GRAPH_REF_PRIMITIVE_HPP

#include "deserialize.hpp"

namespace graph {

// `prb_wrapper_base_t` and `prb_wrapper_t` defined to wrap `prb_t` object
// because C++11 doesn't support templated member variables, and there is no
// common base type for `prb_t` types, thus, it's impossible to put a shared
// pointer of `prb_t` or its base object directly into `ref_prims_` member of
// `ref_partition_t` object. Shared pointer of wrapper base object will be put
// into `ref_prims_`. These wrappers could be removed after moving to C++14.
class prb_wrapper_base_t {
public:
    virtual ~prb_wrapper_base_t() = default;
    template <typename prb_t>
    const prb_t *get() const;
};

// A template class to wrap shared pointer of prb obj
template <typename prb_t>
class prb_wrapper_t : public prb_wrapper_base_t {
public:
    prb_wrapper_t(const std::shared_ptr<prb_t> prb) { prb_ = prb; }
    // get raw pointer of prb object
    const prb_t *get() const { return prb_.get(); }

private:
    std::shared_ptr<prb_t> prb_;
};

// A template function in base wrapper, which dynamic cast from base object to
// derived object and return raw pointer of prb obj
template <typename prb_t>
inline const prb_t *prb_wrapper_base_t::get() const {
    return dynamic_cast<const prb_wrapper_t<prb_t> &>(*this).get();
}

// `ref_primitive_t` is an abstraction to connect a graph op and a primitive
// driver. Its purpose is to translate a graph op into a primitive and execute
// it. Any primitive driver with template programming work should be done
// through this class.
// Note: non-templated functions are exposed to simplify the logic.
class ref_primitive_t {
public:
    ref_primitive_t() = default;
    ref_primitive_t(const deserialized_op &op);

    int init_prb(res_t *res);
    // By default, the reference primitives are created with f32 data type.
    // However, there's a displacer that relies on the logic that would fill
    // memories with int8 data. `force_override` flag restricts forcing f32
    // data type primarily for this use case.
    int init_prim(const engine_t &eng, res_t *res, bool force_override = false);
    void init_memory_args(const engine_t &eng);
    int init_ref_memory_args(const engine_t &eng, res_t *res);
    int execute_prim(res_t *res) const;
    void check_correctness(const args_t &args, bool has_eltwise, bool has_nans,
            res_t *res) const;
    // some util function for ref_partition_t to link args
    void replace_arg(const int arg, const dnn_mem_t &mem) {
        // Only compatible memory objects can be replaced.
        const auto &orig_mem = args_.find(arg);
        if (orig_mem.size() != mem.size()) {
            BENCHDNN_PRINT(0,
                    "Error: can't replace mem_%s (%zu) with mem_%s (%zu) for "
                    "%s op.\n",
                    dt2str(orig_mem.dt()), orig_mem.size(), dt2str(mem.dt()),
                    mem.size(), op_.kind_.c_str());
            SAFE_V(FAIL);
        }

        args_.replace(arg, &mem);
    }
    const dnn_mem_t &get_arg(const int arg) const { return args_.find(arg); }
    ::dnnl::graph::op::kind get_kind() const { return kind_; }
    // Displaces scale values in a memory object with scale values from `op`.
    int displace_scales() const;
    dnnl_data_type_t get_lt_dt(size_t id) const;
    const_dnnl_primitive_desc_t get_pd() const { return query_pd(prim_); }

private:
    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(ref_primitive_t);

    deserialized_op op_;
    ::dnnl::graph::op::kind kind_;
    dnnl_driver_t driver_;
    bool is_special_backward_op_;
    ::std::shared_ptr<prb_wrapper_base_t> prb_wrapper_;
    benchdnn_dnnl_wrapper_t<dnnl_primitive_t> fwd_prim_, prim_;
    dnn_mem_map_t mems_;
    args_t args_;
};

} // namespace graph

#endif
