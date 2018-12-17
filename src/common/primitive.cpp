/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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
#include "memory_pd.hpp"
#include "primitive_desc.hpp"
#include "primitive.hpp"
#include "type_helpers.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace mkldnn::impl;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const primitive_t *p) {
    if (p->engine()->kind() != engine_kind::cpu) return;
    for(auto o: p->outputs()) {
        assert(o->kind() == primitive_kind::memory);
        void *p;
        o->get_data_handle(&p);
        size_t s = ((memory_pd_t *)o->pd())->get_size();
        msan_unpoison(p, s);
    }
}
}

status_t mkldnn_primitive_desc_destroy(primitive_desc_t *primitive_desc) {
    if (primitive_desc) delete primitive_desc;
    return success;
}

status_t mkldnn_primitive_create(primitive_t **primitive,
        const primitive_desc_t *primitive_desc) {
    if (utils::any_null(primitive, primitive_desc))
        return invalid_arguments;
    return primitive_desc->create_primitive(primitive, nullptr, nullptr);
}

status_t mkldnn_primitive_execute(const primitive_t *primitive,
        stream_t *stream, int nargs, const mkldnn_exec_arg_t *c_args) {
    bool ok = true
        && !utils::any_null(primitive, stream)
        && primitive->engine() == stream->engine()
        && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primtive_args(primitive->pd(), nargs, c_args, args);
    if (status != status::success) return status;

    exec_ctx_t ctx(stream, std::move(args));

    if (mkldnn_verbose()->level) {
        double ms = get_msec();
        status = primitive->execute(ctx);
        ms = get_msec() - ms;
        printf("mkldnn_verbose,exec,%s,%g\n", primitive->pd()->info(), ms);
        fflush(0);
    } else {
        status = primitive->execute(ctx);
    }

    if (msan_enabled) unpoison_outputs(primitive);

    return status;
}

status_t mkldnn_primitive_get_primitive_desc(const primitive_t *primitive,
        const primitive_desc_t **primitive_desc) {
    if (utils::any_null(primitive, primitive_desc))
        return invalid_arguments;
    return safe_ptr_assign<const primitive_desc_t>(*primitive_desc,
            primitive->pd());
}

status_t mkldnn_primitive_get_input_at(const primitive_t *primitive,
        size_t index, primitive_at_t *input) {
    if (utils::any_null(primitive, input)
            || index >= primitive->inputs().size())
        return invalid_arguments;
    *input = primitive->inputs()[index];
    return success;
}

status_t mkldnn_primitive_get_output(const primitive_t *primitive,
        size_t index, const primitive_t **output) {
    if (utils::any_null(primitive, output)
            || index >= primitive->outputs().size())
        return invalid_arguments;
    *output = primitive->outputs()[index];
    return success;
}

status_t mkldnn_primitive_destroy(primitive_t *primitive) {
    if (primitive != nullptr)
        delete primitive;
    return success;
}

primitive_at_t mkldnn_primitive_at(const primitive_t *primitive,
        size_t output_index) {
    primitive_at_t result = {primitive, output_index};
    return result;
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
