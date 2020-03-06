/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef PRIMITIVE_DESC_HPP
#define PRIMITIVE_DESC_HPP

#include <typeindex>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "verbose.hpp"

struct dnnl_primitive_desc : public dnnl::impl::c_compatible {
    using md_t = dnnl::impl::memory_desc_t;

    dnnl_primitive_desc(dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_attr_t *attr,
            dnnl::impl::primitive_kind_t kind)
        : engine_(engine), attr_(*attr), kind_(kind) {}

    dnnl_primitive_desc(
            dnnl::impl::engine_t *engine, dnnl::impl::primitive_kind_t kind)
        : engine_(engine), kind_(kind) {}

    virtual dnnl_primitive_desc *clone() const = 0;
    virtual ~dnnl_primitive_desc() {}

    const dnnl::impl::primitive_attr_t *attr() const { return &attr_; }
    dnnl::impl::engine_t *engine() const { return engine_; }
    dnnl::impl::primitive_kind_t kind() const { return kind_; }

    const char *info() const {
        if (!info_.is_initialized()) info_.init(this);
        return info_.c_str();
    }

    dnnl::impl::memory_tracking::registry_t &scratchpad_registry() {
        return scratchpad_registry_;
    }
    const dnnl::impl::memory_tracking::registry_t &scratchpad_registry() const {
        return scratchpad_registry_;
    }
    virtual dnnl::impl::engine_t *scratchpad_engine() const { return engine_; }

    virtual const dnnl::impl::op_desc_t *op_desc() const { return nullptr; }

    enum class arg_usage_t { unused, input, output };
    virtual arg_usage_t arg_usage(int arg) const {
        using dnnl::impl::types::is_zero_md;
        if (arg == DNNL_ARG_ATTR_OUTPUT_SCALES
                && !attr()->output_scales_.defined())
            return arg_usage_t::input;
        if ((arg & DNNL_ARG_ATTR_ZERO_POINTS)
                && !attr()->zero_points_.defined(arg))
            return arg_usage_t::input;
        if (arg == DNNL_ARG_SCRATCHPAD && !is_zero_md(scratchpad_md()))
            return arg_usage_t::output;
        return arg_usage_t::unused;
    }

    virtual const dnnl::impl::memory_desc_t *arg_md(int arg) const {
        switch (arg) {
            case DNNL_ARG_WORKSPACE: return workspace_md(0);
            case DNNL_ARG_SCRATCHPAD: return scratchpad_md(0);
            default: return &dnnl::impl::glob_zero_md;
        }
    }

#define DECLARE_MD_STUB(stub) \
    virtual const dnnl::impl::memory_desc_t *stub(int idx = 0) const { \
        return &dnnl::impl::glob_zero_md; \
    }

    DECLARE_MD_STUB(input_md);
    DECLARE_MD_STUB(output_md);
    DECLARE_MD_STUB(src_md);
    DECLARE_MD_STUB(diff_src_md);
    DECLARE_MD_STUB(dst_md);
    DECLARE_MD_STUB(diff_dst_md);
    DECLARE_MD_STUB(weights_md);
    DECLARE_MD_STUB(diff_weights_md);
    DECLARE_MD_STUB(workspace_md);
#undef DECLARE_MD_STUB

    const dnnl::impl::memory_desc_t *scratchpad_md(int idx = 0) const {
        return idx == 0 ? &scratchpad_md_ : &dnnl::impl::glob_zero_md;
    }

    virtual void init_scratchpad_md() {
        auto size = scratchpad_size(dnnl::impl::scratchpad_mode::user);
        dnnl::impl::dims_t dims = {size};
        dnnl_memory_desc_init_by_tag(&scratchpad_md_, size ? 1 : 0, dims,
                dnnl::impl::data_type::u8, dnnl_x);
    }

    virtual std::type_index impl_id() const {
        assert(!"dnnl_primitive_desc doesn't have impl_id");
        return typeid(dnnl_primitive_desc);
    }

    /** returns the scratchpad size for the given scratchpad mode. */
    dnnl::impl::dim_t scratchpad_size(
            dnnl::impl::scratchpad_mode_t mode) const {
        if (mode != attr_.scratchpad_mode_) return 0;
        return scratchpad_registry().size();
    }

    virtual int n_inputs() const { return 0; }
    virtual int n_outputs() const { return 0; }

    virtual dnnl::impl::status_t query(
            dnnl::impl::query_t what, int idx, void *result) const;

    virtual dnnl::impl::status_t create_primitive_iface(
            dnnl::impl::primitive_iface_t **p_iface) const = 0;

    virtual const char *name() const { return "dnnl_primitive_desc"; }

    /* static magic */

    template <typename pd_t>
    static dnnl::impl::status_t create(dnnl::impl::primitive_desc_t **pd,
            const dnnl::impl::op_desc_t *adesc,
            const dnnl::impl::primitive_attr_t *attr,
            dnnl::impl::engine_t *engine,
            const dnnl::impl::primitive_desc_t *hint_fwd) {
        using namespace dnnl::impl;
        using namespace dnnl::impl::status;
        using pd_op_desc_t = typename pkind_traits<pd_t::base_pkind>::desc_type;
        // A hack to reuse softmax code using logsoftmax primitive.
        // TODO: consider removing it in v2.0 by introducing alg_kind in softmax
        bool valid_logsoftmax = pd_t::base_pkind == primitive_kind::softmax
                && adesc->kind == primitive_kind::logsoftmax;
        if (adesc->kind != pd_t::base_pkind && !valid_logsoftmax)
            return invalid_arguments;
        assert(hint_fwd ? hint_fwd->kind() == pd_t::base_pkind : true);
        auto hint
                = reinterpret_cast<const typename pd_t::hint_class *>(hint_fwd);
        auto _pd = new pd_t(engine, (const pd_op_desc_t *)adesc, attr, hint);
        if (_pd == nullptr) return out_of_memory;
        if (_pd->init() != success) {
            delete _pd;
            return unimplemented;
        }

        _pd->init_scratchpad_md();
        *pd = _pd;
        return success;
    }

protected:
    dnnl::impl::engine_t *engine_;
    dnnl::impl::primitive_attr_t attr_;
    dnnl::impl::primitive_kind_t kind_;

    dnnl::impl::memory_desc_t scratchpad_md_;

    mutable dnnl::impl::pd_info_t info_;

    dnnl::impl::memory_tracking::registry_t scratchpad_registry_;

protected:
    /** compares ws between fwd_pd and this (make sense to use for bwd_pd)
     * Expectation: this already set workspace, and this workspace should
     *              exactly match the one from fwd_pd */
    bool compare_ws(const dnnl_primitive_desc *fwd_pd) const {
        using namespace dnnl::impl;
        if (!workspace_md()) return true; // the impl lives fine w/o workspace
        return fwd_pd && fwd_pd->workspace_md()
                && *fwd_pd->workspace_md() == *workspace_md();
    }
};

#define DECLARE_COMMON_PD_t(impl_name, impl_type, use_global_scratchpad) \
    virtual pd_t *clone() const override { return new pd_t(*this); } \
    virtual status_t create_primitive_iface(primitive_iface_t **p_iface) \
            const override { \
        auto status = this->engine()->get_primitive_iface( \
                p_iface, this, \
                [=] { return std::make_shared<impl_type>(this); }, \
                use_global_scratchpad); \
        return status; \
    } \
    virtual const char *name() const override { return impl_name; } \
    virtual std::type_index impl_id() const override { return typeid(pd_t); }

#define DECLARE_COMMON_PD_T_USE_GLOBAL_SCRATCHPAD(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, true)

#define DECLARE_COMMON_PD_T_(impl_name, impl_type) \
    DECLARE_COMMON_PD_t(impl_name, impl_type, false)

#define DECLARE_COMMON_PD_T(impl_name, impl_type, ...) \
    DECLARE_COMMON_PD_T_##__VA_ARGS__(impl_name, impl_type)

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
