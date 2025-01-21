/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
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

#include <string>

#include "gpu/intel/jit/ir/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

std::string to_string(tensor_kind_t tensor) {
    switch (tensor) {
#define CASE(name) \
    case tensor_kind_t::name: return #name
        CASE(src);
        CASE(wei);
        CASE(dst);
        CASE(a);
        CASE(b);
        CASE(c);
#undef CASE
        default: gpu_error_not_expected();
    }
    return {};
}

const expr_t &pvar_t::index_var() const {
    static thread_local pvar_map_t<expr_t> vars;
    if (!vars.has(*this)) {
        auto var = var_t::make(type_t::s32(), name_ + "_idx");
        vars[*this] = var;
    }
    return vars[*this];
}

const expr_t &pvar_t::var() const {
    static thread_local pvar_map_t<expr_t> vars;
    if (!vars.has(*this)) {
        auto var = const_var_t::make(type_t::s32(), name_);
        vars[*this] = var;
    }
    return vars[*this];
}

pvar_t pvar_t::from_var(const expr_t &var) {
    auto *ptr = var.as_ptr<const_var_t>();
    if (!ptr) return pvar_t();
    return pvar_t(ptr->name);
}

pvar_t pvar_t::from_index_var(const expr_t &index_var) {
    auto *ptr = index_var.as_ptr<var_t>();
    if (!ptr) return pvar_t();
    const char *suffix = "_idx";
    const size_t suffix_len = std::strlen(suffix);
    auto &name = ptr->name;
    auto pos = name.find(suffix);
    if (pos == std::string::npos || pos + suffix_len != name.length())
        return pvar_t();
    return pvar_t(name.substr(0, name.length() - suffix_len));
}

char pvar_t::to_spatial() const {
    if (name_.size() != 2) return ' ';
    char c0 = name_[0];
    char c1 = name_[1];
    if (!std::strchr("dikops", c0)) return ' ';
    if (!std::strchr("dhw", c1)) return ' ';
    return c1;
}

int pvar_t::spatial_index() const {
    char sp = to_spatial();
    switch (sp) {
        case 'd': return 0;
        case 'h': return 1;
        case 'w': return 2;
        default: return -1;
    }
    return -1;
}

namespace pvars {
pvar_t g("g");
pvar_t ic("ic");
pvar_t id("id");
pvar_t ih("ih");
pvar_t iw("iw");
pvar_t kd("kd");
pvar_t kh("kh");
pvar_t kw("kw");
pvar_t mb("mb");
pvar_t oc("oc");
pvar_t od("od");
pvar_t oh("oh");
pvar_t ow("ow");
pvar_t sd("sd");
pvar_t sh("sh");
pvar_t sw("sw");
pvar_t dd("dd");
pvar_t dh("dh");
pvar_t dw("dw");
pvar_t pd("pd");
pvar_t ph("ph");
pvar_t pw("pw");
pvar_t b("b");
pvar_t m("m");
pvar_t n("n");
pvar_t k("k");
} // namespace pvars

bool is_spatial(const pvar_t &pvar, char prefix) {
    if (pvar.name().size() != 2) return false;
    char c0 = pvar.name()[0];
    char c1 = pvar.name()[1];
    return (c0 == prefix) && utils::one_of(c1, 'd', 'h', 'w');
}
bool is_input_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'i');
}
bool is_output_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'o');
}
bool is_kernel_spatial(const pvar_t &pvar) {
    return is_spatial(pvar, 'k');
}
bool is_dilation(const pvar_t &pvar) {
    return is_spatial(pvar, 'd');
}
bool is_padding(const pvar_t &pvar) {
    return is_spatial(pvar, 'p');
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
