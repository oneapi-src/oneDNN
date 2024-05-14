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
        default: ir_error_not_expected();
    }
    return {};
}

std::string to_string(prb_dim_kind_t kind) {
    switch (kind) {
#define CASE(name) \
    case prb_dim_kind_t::name: return #name
        CASE(undef);
        CASE(g);
        CASE(ic);
        CASE(id);
        CASE(ih);
        CASE(iw);
        CASE(kd);
        CASE(kh);
        CASE(kw);
        CASE(mb);
        CASE(oc);
        CASE(od);
        CASE(oh);
        CASE(ow);
        CASE(sd);
        CASE(sh);
        CASE(sw);
        CASE(dd);
        CASE(dh);
        CASE(dw);
        CASE(pd);
        CASE(ph);
        CASE(pw);
        CASE(b);
        CASE(m);
        CASE(n);
        CASE(k);
#undef CASE
        default: ir_error_not_expected();
    }
    return {};
}

prb_dim_spatial_kind_t to_spatial(prb_dim_kind_t kind) {
    switch (kind) {
        case prb_dim_kind_t::id:
        case prb_dim_kind_t::od:
        case prb_dim_kind_t::kd:
        case prb_dim_kind_t::sd:
        case prb_dim_kind_t::dd:
        case prb_dim_kind_t::pd: return prb_dim_spatial_kind_t::d;
        case prb_dim_kind_t::ih:
        case prb_dim_kind_t::oh:
        case prb_dim_kind_t::kh:
        case prb_dim_kind_t::sh:
        case prb_dim_kind_t::dh:
        case prb_dim_kind_t::ph: return prb_dim_spatial_kind_t::h;
        case prb_dim_kind_t::iw:
        case prb_dim_kind_t::ow:
        case prb_dim_kind_t::kw:
        case prb_dim_kind_t::sw:
        case prb_dim_kind_t::dw:
        case prb_dim_kind_t::pw: return prb_dim_spatial_kind_t::w;
        default: return prb_dim_spatial_kind_t::undef;
    }
}

namespace prb_dims {
prb_dim_t undef(prb_dim_kind_t::undef);
prb_dim_t g(prb_dim_kind_t::g);
prb_dim_t ic(prb_dim_kind_t::ic);
prb_dim_t id(prb_dim_kind_t::id);
prb_dim_t ih(prb_dim_kind_t::ih);
prb_dim_t iw(prb_dim_kind_t::iw);
prb_dim_t kd(prb_dim_kind_t::kd);
prb_dim_t kh(prb_dim_kind_t::kh);
prb_dim_t kw(prb_dim_kind_t::kw);
prb_dim_t mb(prb_dim_kind_t::mb);
prb_dim_t oc(prb_dim_kind_t::oc);
prb_dim_t od(prb_dim_kind_t::od);
prb_dim_t oh(prb_dim_kind_t::oh);
prb_dim_t ow(prb_dim_kind_t::ow);
prb_dim_t sd(prb_dim_kind_t::sd);
prb_dim_t sh(prb_dim_kind_t::sh);
prb_dim_t sw(prb_dim_kind_t::sw);
prb_dim_t dd(prb_dim_kind_t::dd);
prb_dim_t dh(prb_dim_kind_t::dh);
prb_dim_t dw(prb_dim_kind_t::dw);
prb_dim_t pd(prb_dim_kind_t::pd);
prb_dim_t ph(prb_dim_kind_t::ph);
prb_dim_t pw(prb_dim_kind_t::pw);
prb_dim_t b(prb_dim_kind_t::b);
prb_dim_t m(prb_dim_kind_t::m);
prb_dim_t n(prb_dim_kind_t::n);
prb_dim_t k(prb_dim_kind_t::k);
} // namespace prb_dims

int spatial_index(const prb_dim_t &dim) {
    switch (to_spatial(dim.kind())) {
        case prb_dim_spatial_kind_t::d: return 0;
        case prb_dim_spatial_kind_t::h: return 1;
        case prb_dim_spatial_kind_t::w: return 2;
        default: return -1;
    }
}

const expr_t &index_var(const prb_dim_t &prb_dim) {
    static thread_local dim_map_t<prb_dim_t, expr_t> index_vars = []() {
        dim_map_t<prb_dim_t, expr_t> ret;
        for (auto &d : prb_dim_t::all()) {
            ret[d] = var_t::make(type_t::s32(), d.str() + "_idx");
        }
        return ret;
    }();
    return index_vars.at(prb_dim);
}

const expr_t &size_var(const prb_dim_t &prb_dim) {
    static thread_local dim_map_t<prb_dim_t, expr_t> size_vars = []() {
        dim_map_t<prb_dim_t, expr_t> ret;
        for (auto &d : prb_dim_t::all()) {
            ret[d] = const_var_t::make(type_t::s32(), d.str());
        }
        return ret;
    }();
    return size_vars.at(prb_dim);
}

prb_dim_t index_to_prb_dim(const expr_t &var) {
    for (auto &d : prb_dim_t::all()) {
        if (index_var(d).is_same(var)) return d;
    }
    return prb_dims::undef;
}

prb_dim_t size_to_prb_dim(const expr_t &var) {
    for (auto &d : prb_dim_t::all()) {
        if (size_var(d).is_same(var)) return d;
    }
    return prb_dims::undef;
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
