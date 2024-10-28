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

#ifndef GPU_INTEL_JIT_V2_CONV_PROBLEM_HPP
#define GPU_INTEL_JIT_V2_CONV_PROBLEM_HPP

#include "gpu/intel/jit/conv/problem.hpp"
#include "gpu/intel/jit/ir/tensor_config.hpp"
#include "gpu/intel/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

class problem_t {
public:
    problem_t() = default;
    problem_t(const std::string &line);
    const hw_t &hw() const { return hw_; }
    prop_kind_t prop() const { return prop_; }
    const layout_tag_t &src_tag() const { return src_tag_; }
    const layout_tag_t &wei_tag() const { return wei_tag_; }
    const layout_tag_t &dst_tag() const { return dst_tag_; }
    const type_t &bias_type() const { return bias_type_; }
    const layout_tag_t &layout_tag(tensor_kind_t kind) const {
        switch (kind) {
            case tensor_kind_t::a:
                return pick_a(prop_, src_tag_, wei_tag_, dst_tag_);
            case tensor_kind_t::b:
                return pick_b(prop_, src_tag_, wei_tag_, dst_tag_);
            case tensor_kind_t::c:
                return pick_c(prop_, src_tag_, wei_tag_, dst_tag_);
            default: ir_error_not_expected();
        }
        return src_tag_;
    }
    const pvar_tile_t &shape() const { return shape_; }
    pvar_map_t<dim_t> vars() const;
    bool is_depthwise() const {
        dim_t g = shape_.at(pvars::g);
        dim_t ic = shape_.at(pvars::ic);
        dim_t oc = shape_.at(pvars::oc);
        return (g > 1) && (ic == 1) && (oc == 1);
    }
    const type_t &out_type() const;
    void set_hw(const hw_t &hw) { hw_ = hw; }
    void set_prop(prop_kind_t prop) {
        prop_ = prop;
        if (prop_ == prop_kind::forward_inference) prop_ = prop_kind::forward;
    }
    void set_src_tag(const layout_tag_t &tag) { src_tag_ = tag; }
    void set_wei_tag(const layout_tag_t &tag) { wei_tag_ = tag; }
    void set_dst_tag(const layout_tag_t &tag) { dst_tag_ = tag; }
    void set_bias_type(const type_t &bias_type) { bias_type_ = bias_type; }
    void set_shape(const pvar_tile_t &shape) { shape_ = shape; }
    bool with_bias_fwd() const {
        return prop_ == prop_kind::forward && !bias_type_.is_undef();
    }
    bool with_bias_bwd_w() const {
        return prop_ == prop_kind::backward_weights && !bias_type_.is_undef();
    }
    double ops() const;

    void set_shape(const std::string &s);
    void normalize();
    std::string desc_str() const;
    std::string str() const;
    std::string csv_str() const;

    IR_DEFINE_DUMP()

    static pvar_tile_t default_shape();
    static double ops(prop_kind_t prop, const pvar_tile_t &shape);

private:
    hw_t hw_;
    prop_kind_t prop_ = prop_kind::undef;
    layout_tag_t src_tag_;
    layout_tag_t wei_tag_;
    layout_tag_t dst_tag_;
    type_t bias_type_;
    pvar_tile_t shape_;
    std::array<int, 3> dhw_map_;
};

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
