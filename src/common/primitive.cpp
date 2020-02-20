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

// API
status_t dnnl_primitive_desc_destroy(primitive_desc_t *primitive_desc) {
    if (primitive_desc) delete primitive_desc;
    return success;
}

status_t dnnl_primitive_create(
        primitive_t **primitive, const primitive_desc_t *primitive_desc) {
    if (utils::any_null(primitive, primitive_desc)) return invalid_arguments;
    return primitive_desc->create_primitive(primitive);
}

status_t dnnl_primitive_execute(const primitive_t *primitive, stream_t *stream,
        int nargs, const dnnl_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive, stream)
            && primitive->engine() == stream->engine()
            && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primtive_args(primitive->pd(), nargs, c_args, args);
    if (status != status::success) return status;

    exec_ctx_t ctx(stream, std::move(args));

    if (get_verbose()) {
        double ms = get_msec();
        status = primitive->execute(ctx);
        stream->wait();
        ms = get_msec() - ms;
        printf("dnnl_verbose,exec,%s,%g\n", primitive->pd()->info(), ms);
        fflush(0);
    } else {
        status = primitive->execute(ctx);
    }

    if (msan_enabled) unpoison_outputs(ctx.args());

    return status;
}

status_t dnnl_primitive_get_primitive_desc(
        const primitive_t *primitive, const primitive_desc_t **primitive_desc) {
    if (utils::any_null(primitive, primitive_desc)) return invalid_arguments;
    return safe_ptr_assign<const primitive_desc_t>(
            *primitive_desc, primitive->pd());
}

status_t dnnl_primitive_destroy(primitive_t *primitive) {
    if (primitive != nullptr) delete primitive;
    return success;
}

// primitive_t implementation
dnnl_primitive::dnnl_primitive(
        const std::shared_ptr<primitive_impl_t> &primitive_impl,
        bool use_global_scratchpad = false)
    : primitive_impl_(primitive_impl), scratchpad_(nullptr) {

    const size_t scratchpad_size
            = primitive_impl_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        auto *scratchpad_ptr = create_scratchpad(
                engine(), scratchpad_size, use_global_scratchpad);
        scratchpad_.reset(scratchpad_ptr);
    }
}

status_t dnnl_primitive::init() {
    return primitive_impl_->init();
}

engine_t *dnnl_primitive::engine() const {
    return primitive_impl_->engine();
}

const primitive_desc_t *dnnl_primitive::pd() const {
    return primitive_impl_->pd();
}

const std::shared_ptr<primitive_impl_t> &
dnnl_primitive::get_primitive_impl() const {
    return primitive_impl_;
}

status_t dnnl_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_impl_->pd()->attr()->scratchpad_mode_
            == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(DNNL_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                                        : nullptr;
    } else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    ctx.set_scratchpad_grantor(
            primitive_impl_->pd()->scratchpad_registry().grantor(mem_storage));

    auto status = primitive_impl_->execute(ctx);
    return status;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
