/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_IR_TENSOR_CONFIG_HPP
#define GPU_INTEL_JIT_IR_TENSOR_CONFIG_HPP

#include <vector>

#include "gpu/intel/jit/ir/post_ops.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct tensor_info_t {
    std::string name;
    int arg_key;
    bool is_input;
    bool is_output;
    layout_t compute_layout;
    layout_t user_layout;

    bool needs_reorder;
    bool needs_zero_out;
};

class tensor_config_t {
public:
    const std::vector<tensor_info_t> &tensors() const { return tensors_; }

    void add_tensor(const std::string &name, int arg_key, bool is_input,
            bool is_output, const layout_t &user_layout) {
        tensors_.emplace_back();
        auto &t = tensors_.back();
        t.name = name;
        t.arg_key = arg_key;
        t.is_input = is_input;
        t.is_output = is_output;
        t.compute_layout = user_layout;
        t.user_layout = user_layout;
        t.needs_reorder = false;
        t.needs_zero_out = false;
    }

    void add_tensor(const std::string &name, int arg_key, bool is_input,
            bool is_output, const layout_t &compute_layout,
            const layout_t &user_layout) {
        tensors_.emplace_back();
        auto &t = tensors_.back();
        t.name = name;
        t.arg_key = arg_key;
        t.is_input = is_input;
        t.is_output = is_output;
        t.compute_layout = compute_layout;
        t.user_layout = user_layout;
        t.needs_reorder = (t.compute_layout != t.user_layout);
        t.needs_zero_out = false;
    }

    void set_compute_layout(
            const std::string &name, const layout_t &compute_layout) {
        auto &t = find_tensor(name);
        t.compute_layout = compute_layout;
        t.needs_reorder = (t.compute_layout != t.user_layout);
    }

    const layout_t &compute_layout(const std::string &name) const {
        return find_tensor(name).compute_layout;
    }

    const layout_t &user_layout(const std::string &name) const {
        return find_tensor(name).user_layout;
    }

    void require_zero_out(const std::string &name) {
        auto &t = find_tensor(name);
        t.needs_zero_out = true;
    }

private:
    const tensor_info_t &find_tensor(const std::string &name) const {
        for (auto &t : tensors_) {
            if (t.name == name) return t;
        }
        ir_error_not_expected() << "Can't find tensor " << name;
        return tensors_.front();
    }

    tensor_info_t &find_tensor(const std::string &name) {
        auto *const_this = const_cast<const tensor_config_t *>(this);
        return const_cast<tensor_info_t &>(const_this->find_tensor(name));
    }

    std::vector<tensor_info_t> tensors_;
};

inline layout_t make_layout(const memory_desc_t &md) {
    if (md.format_kind == format_kind::any) return layout_t();
    return layout_t(md, /*do_normalize=*/false);
}

inline layout_t make_layout(const memory_desc_t &md, const std::string &tag) {
    if (tag == "user") return layout_t(md);
    return layout_t(md, tag, /*do_normalize=*/false);
}

inline layout_t make_layout(const type_t &type, const std::vector<dim_t> &dims,
        const std::string &tag) {
    return layout_t(type, 0, tag, dims, /*do_normalize=*/false);
}

inline void set_default_format(memory_desc_t &md, const std::string &tag) {
    if (md.format_kind != format_kind::any) return;
    md = make_layout(md, tag).to_dnnl(md.dims);
}

inline std::vector<std::pair<const char *, int>> get_scale_args() {
    std::vector<std::pair<const char *, int>> ret = {
            {"src_scales", DNNL_ARG_SRC},
            {"wei_scales", DNNL_ARG_WEIGHTS},
            {"dst_scales", DNNL_ARG_DST},
    };
    return ret;
}

inline std::vector<dim_t> get_prelu_weights_dims(
        uint32_t mask, const memory_desc_t &md) {
    std::vector<dim_t> dims(md.dims, md.dims + md.ndims);
    for (int i = 0; i < md.ndims; ++i)
        dims[i] = (mask & (1 << i)) ? dims[i] : 1;
    return dims;
}

void init_extra_tensors(const zero_points_config_t &zp_cfg,
        const primitive_attr_t &attr, const memory_desc_t *zp_src,
        const memory_desc_t &dst_md, dim_t ic, dim_t oc,
        tensor_config_t &tensor_cfg);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
