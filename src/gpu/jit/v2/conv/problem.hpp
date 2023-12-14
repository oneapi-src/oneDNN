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

#ifndef GPU_JIT_V2_CONV_PROBLEM_HPP
#define GPU_JIT_V2_CONV_PROBLEM_HPP

#include "gpu/jit/conv/problem.hpp"
#include "gpu/jit/ir/tensor_config.hpp"
#include "gpu/jit/v2/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
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
    const prb_tile_t &shape() const { return shape_; }

    bool is_depthwise() const {
        int g = shape_.at(prb_dims::g);
        int ic = shape_.at(prb_dims::ic);
        int oc = shape_.at(prb_dims::oc);
        return (g > 1) && (ic == 1) && (oc == 1);
    }

    void set_hw(const hw_t &hw) { hw_ = hw; }
    void set_prop(prop_kind_t prop) {
        prop_ = prop;
        if (prop_ == prop_kind::forward_inference) prop_ = prop_kind::forward;
    }
    void set_src_tag(const layout_tag_t &tag) { src_tag_ = tag; }
    void set_wei_tag(const layout_tag_t &tag) { wei_tag_ = tag; }
    void set_dst_tag(const layout_tag_t &tag) { dst_tag_ = tag; }
    void set_shape(const prb_tile_t &shape) { shape_ = shape; }

    void set_shape(const std::string &s);
    std::string desc_str() const;
    void serialize(std::ostream &out) const;
    void deserialize(std::istream &in);
    std::string str() const;
    std::string csv_str() const;

    IR_DEFINE_DUMP()

    static prb_tile_t default_shape();

private:
    hw_t hw_;
    prop_kind_t prop_ = prop_kind::undef;
    layout_tag_t src_tag_;
    layout_tag_t wei_tag_;
    layout_tag_t dst_tag_;
    prb_tile_t shape_;
};

tensor_config_t get_tensor_config(prop_kind_t prop);

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
