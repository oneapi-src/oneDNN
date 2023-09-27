/*******************************************************************************
* Copyright 2023 Intel Corporation
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

// prb_wrapper_base_t & prb_wrapper_t are defined to wrap prb objection because
// C++ 11 does not support template member variable and there is no common base type
// for all prb_t types, thus we cannot put shared ptr of prb or its base object
// directly into ref_prims_ member of ref_partition_t object. now shared pointer of
// wrapper base object will be put into ref_prims_.
// These wrappers could be removed after moving to C++ 14
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

// ref_primitive_t is a bridge between graph op and primitive driver
// translate graph op into primitive and run
// all primitive driver with template programming work should be handled in this class
// expose non-template function to the caller to simplify the logic
class ref_primitive_t {
public:
    ref_primitive_t() = default;
    ref_primitive_t(const deserialized_op &op);

    void init_prb(::std::unordered_set<size_t> &bf16_rewrite, res_t *res);
    int init_prim(const engine_t &eng, res_t *res);
    void init_memory_args(const engine_t &eng);
    int init_ref_memory_args(const engine_t &eng, res_t *res);
    int execute_prim(res_t *res) const;
    void check_correctness(const args_t &args, bool has_eltwise, bool has_nans,
            res_t *res) const;
    // some util function for ref_partition_t to link args
    void replace_arg(const int arg, const dnn_mem_t &mem) {
        args_.replace(arg, &mem);
    }
    const dnn_mem_t &get_arg(const int arg) const { return args_.find(arg); }
    ::dnnl::graph::op::kind get_kind() const { return kind_; }

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
