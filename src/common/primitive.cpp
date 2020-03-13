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

#include <assert.h>

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive.hpp"
#include "primitive_desc.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const exec_args_t &args) {
    for (const auto &arg : args) {
        if (arg.second.is_const) continue;
        auto *mem = arg.second.mem;
        void *p;
        mem->get_data_handle(&p);
        size_t s = memory_desc_wrapper(*mem->md()).size();
        msan_unpoison(p, s);
    }
}
} // namespace

namespace dnnl {
namespace impl {

nested_scratchpad_t::nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
        const std::shared_ptr<primitive_t> &nested_p) {
    auto scratchpad = master_ctx.get_scratchpad_grantor();
    scratchpad_mem_storage_ = scratchpad.get_memory_storage(key);
    grantor_ = utils::make_unique<memory_tracking::grantor_t>(
            nested_p->pd()->scratchpad_registry().grantor(
                    scratchpad_mem_storage_.get()));
}

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_primitive_desc_destroy(
        primitive_desc_iface_t *primitive_desc_iface) {
    if (primitive_desc_iface) delete primitive_desc_iface;
    return success;
}

status_t dnnl_primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return primitive_desc_iface->create_primitive_iface(primitive_iface);
}

status_t dnnl_primitive_execute(const primitive_iface_t *primitive_iface,
        stream_t *stream, int nargs, const dnnl_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primtive_args(
            primitive_iface->pd()->impl().get(), nargs, c_args, args);
    if (status != status::success) return status;

    stream->before_exec_hook();

    exec_ctx_t ctx(stream, std::move(args));

    if (get_verbose()) {
        double ms = get_msec();
        status = primitive_iface->execute(ctx);
        stream->wait();
        ms = get_msec() - ms;
        printf("dnnl_verbose,exec,%s,%g\n", primitive_iface->pd()->info(), ms);
        fflush(0);
    } else {
        status = primitive_iface->execute(ctx);
    }

    stream->after_exec_hook();

    if (msan_enabled) unpoison_outputs(ctx.args());

    return status;
}

status_t dnnl_primitive_get_primitive_desc(
        const primitive_iface_t *primitive_iface,
        const primitive_desc_iface_t **primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return safe_ptr_assign<const primitive_desc_iface_t>(
            *primitive_desc_iface, primitive_iface->pd());
}

status_t dnnl_primitive_destroy(primitive_iface_t *primitive_iface) {
    if (primitive_iface != nullptr) delete primitive_iface;
    return success;
}

// primitive_t implementation
dnnl_primitive::dnnl_primitive(const std::shared_ptr<primitive_t> &primitive,
        engine_t *engine, bool use_global_scratchpad)
    : primitive_(primitive)
    , scratchpad_(nullptr)
    , pd_(utils::make_unique<primitive_desc_iface_t>(
              primitive_->pd(), engine)) {

    const size_t scratchpad_size
            = primitive_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        auto *scratchpad_ptr = create_scratchpad(
                pd_->engine(), scratchpad_size, use_global_scratchpad);
        scratchpad_.reset(scratchpad_ptr);
    }
}

status_t dnnl_primitive::init() {
    // TODO: move scratchpad creation here from ctor
    return primitive_->create_resource(pd()->engine(), resource_mapper_);
}

engine_t *dnnl_primitive::engine() const {
    return pd_->engine();
}

const primitive_desc_iface_t *dnnl_primitive::pd() const {
    return pd_.get();
}

const std::shared_ptr<primitive_t> &dnnl_primitive::get_primitive() const {
    return primitive_;
}

status_t dnnl_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_->pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(DNNL_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                                        : nullptr;
    } else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    auto scratchpad_grantor
            = primitive_->pd()->scratchpad_registry().grantor(mem_storage);
    ctx.set_scratchpad_grantor(&scratchpad_grantor);
    ctx.set_resource_mapper(&resource_mapper_);

    auto status = primitive_->execute(ctx);
    return status;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
