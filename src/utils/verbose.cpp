/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <climits>
#include <cstring>
#include <stdlib.h>
#include <sys/time.h>

#include "oneapi/dnnl/dnnl_graph.h"

#include "debug.hpp"
#include "verbose.hpp"

#include "interface/backend.hpp"
#include "interface/c_types_map.hpp"
#include "interface/partition.hpp"
#include "interface/utils.hpp"

#ifndef DNNL_GRAPH_VERSION_MAJOR
#define DNNL_GRAPH_VERSION_MAJOR INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_MINOR
#define DNNL_GRAPH_VERSION_MINOR INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_PATCH
#define DNNL_GRAPH_VERSION_PATCH INT_MAX
#endif

#ifndef DNNL_GRAPH_VERSION_HASH
#define DNNL_GRAPH_VERSION_HASH "N/A"
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace utils {

// The following code is derived from oneDNN/src/common/verbose.cpp
double get_msec() {
    struct timeval time;
    gettimeofday(&time, nullptr);
    return 1e+3 * static_cast<double>(time.tv_sec)
            + 1e-3 * static_cast<double>(time.tv_usec);
}

int getenv(const char *name, char *buffer, int buffer_size) {
    if (name == nullptr || buffer_size < 0
            || (buffer == nullptr && buffer_size > 0))
        return INT_MIN;

    int result = 0;
    int term_zero_idx = 0;
    size_t value_length = 0;

    const char *value = ::getenv(name);
    value_length = value == nullptr ? 0 : strlen(value);

    if (value_length > INT_MAX)
        result = INT_MIN;
    else {
        int int_value_length = (int)value_length;
        if (int_value_length >= buffer_size) {
            result = -int_value_length;
        } else {
            term_zero_idx = int_value_length;
            result = int_value_length;
            if (value) strncpy(buffer, value, (size_t)buffer_size - 1);
        }
    }

    if (buffer != nullptr) buffer[term_zero_idx] = '\0';
    return result;
}

static setting_t<int> verbose {0};
int get_verbose() {
#if !defined(DNNL_GRAPH_DISABLE_VERBOSE)
    if (!verbose.initialized()) {
        const int len = 2;
        char var[len] = {0};
        if (getenv("DNNL_GRAPH_VERBOSE", var, len) == 1) verbose.set(atoi(var));
        if (!verbose.initialized()) verbose.set(0);
    }
    static bool version_printed = false;
    if (!version_printed && verbose.get() > 0) {
        printf("dnnl_graph_verbose,info,oneDNN Graph v%d.%d.%d (commit %s)\n",
                dnnl_graph_version()->major, dnnl_graph_version()->minor,
                dnnl_graph_version()->patch, dnnl_graph_version()->hash);
        version_printed = true;
    }
#endif
    return verbose.get();
}

#if defined(DNNL_GRAPH_DISABLE_VERBOSE)
void partition_info_t::init(
        const engine_t *engine, const compiled_partition_t *partition) {
    UNUSED(engine);
    UNUSED(partition);
}

#else

namespace {
#define DNNL_GRAPH_VERBOSE_DAT_LEN 512
#define DNNL_GRAPH_VERBOSE_FMT_LEN 32

#define DECL_DAT_STRS() \
    int dat_written = 0; \
    char dat_str[DNNL_GRAPH_VERBOSE_DAT_LEN] = {'\0'}; \
    int fmt_written = 0; \
    char fmt_str[DNNL_GRAPH_VERBOSE_FMT_LEN] = {'\0'}

// The following code is derived from oneDNN/src/common/verbose.cpp
void clear_buf(char *buf, int &written) {
    /* TODO: do it better */
    buf[0] = '#';
    buf[1] = '\0';
    written = 1;
}

#define CHECK_WRITTEN(buf, buf_len, written_now, written_total) \
    do { \
        if ((written_now) < 0 \
                || (written_total) + (written_now) > (buf_len)) { \
            clear_buf(buf, written_total); \
        } else { \
            (written_total) += (written_now); \
        } \
    } while (0)

#define PUTS(...) \
    do { \
        int l = snprintf(str + written_len, str_len, __VA_ARGS__); \
        if (l < 0) return l; \
        if ((size_t)l >= str_len) return -1; \
        written_len += l; \
        str_len -= (size_t)l; \
    } while (0);

#define DPRINT(buf, buf_len, written, ...) \
    do { \
        int l = snprintf((buf) + (written), (size_t)((buf_len) - (written)), \
                __VA_ARGS__); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0)

#define LOGICAL_TENSOR2STR(buf, buf_len, written, logical_tensor) \
    do { \
        int l = logical_tensor2str((buf) + (written), \
                (size_t)((buf_len) - (written)), logical_tensor); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0)

#define DIM2STR(buf, buf_len, written, logical_tensor) \
    do { \
        int l = logical_tensor2dim_str((buf) + (written), \
                (size_t)((buf_len) - (written)), logical_tensor); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0);

#define LAYOUT2STR(buf, buf_len, written, logical_tensor) \
    do { \
        int l = logical_tensor2layout_str((buf) + (written), \
                (size_t)((buf_len) - (written)), logical_tensor); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0);

#define FMT2STR(buf, buf_len, written, partition) \
    do { \
        int l = partition2fmt_str((buf) + (written), \
                (size_t)((buf_len) - (written)), partition); \
        CHECK_WRITTEN(buf, buf_len, l, written); \
    } while (0);

void verbose_templ_no_engine_kind(char *buffer, size_t parition_id,
        const char *op_name_str, const char *fmt_str, const char *data_str,
        const char *backend_str, int written = 0) {
    DPRINT(buffer, DNNL_GRAPH_VERBOSE_BUF_LEN, written, "%ld,%s,%s,%s,%s",
            parition_id, op_name_str, fmt_str, data_str, backend_str);
}

int logical_tensor2dim_str(char *str, size_t str_len,
        const impl::logical_tensor_t &logical_tenosr) {
    if (str == nullptr || str_len <= 1u) return -1;

    int written_len = 0;

    auto lt = impl::logical_tensor_wrapper(logical_tenosr);
    const int32_t ndim = lt.ndims();
    const auto dims = lt.dims();

    PUTS(":");
    for (int i = 0; i < ndim - 1; ++i) {
        PUTS("%ldx", dims[i]);
    }
    PUTS("%ld", dims[ndim - 1]);

    return written_len;
}

int logical_tensor2layout_str(char *str, size_t str_len,
        const impl::logical_tensor_t &logical_tensor) {
    if (str == nullptr || str_len <= 1u) return -1;

    int written_len = 0;

    auto lt = impl::logical_tensor_wrapper(logical_tensor);
    const int32_t ndim = lt.ndims();

    PUTS(":");
    if (lt.layout_type() == impl::layout_type::strided) {
        const auto strides = lt.strides();
        for (int i = 0; i < ndim - 1; ++i) {
            PUTS("%lds", strides[i]);
        }
        PUTS("%ld", strides[ndim - 1]);
    } else if (lt.layout_type() == impl::layout_type::opaque) {
        PUTS("%ld", lt.layout_id());
    } else {
        assert(!"layout type must be any, strided or opaque.");
    }

    return written_len;
}

int logical_tensor2str(char *str, size_t str_len,
        const impl::logical_tensor_t &logical_tensor) {
    if (str == nullptr || str_len <= 1u) return -1;

    int written = 0;
    DPRINT(str, DNNL_GRAPH_VERBOSE_DAT_LEN, written, "%s:%ld:%s",
            data_type2str(logical_tensor.data_type), logical_tensor.id,
            layout_type2str(logical_tensor.layout_type));

    return written;
}

int partition2fmt_str(
        char *str, size_t str_len, const impl::partition_t &partition) {
    if (str == nullptr || str_len <= 1u) return -1;
    // FIXME(qun) Disable for now. Need some re-design after new
    // backend API is ready
    // const auto &op = partition.node();
    UNUSED(partition);

    int written = 0;
    /*
    if (op->has_attr("data_format")) {
        const auto data_format = op->get_attr<std::string>("data_format");
        DPRINT(str, DNNL_GRAPH_VERBOSE_FMT_LEN, written, "data:%s",
                data_format.c_str());
    }
    if (op->has_attr("filter_format")) {
        const auto filter_format = op->get_attr<std::string>("filter_format");
        DPRINT(str, DNNL_GRAPH_VERBOSE_FMT_LEN, written, " filter:%s",
                filter_format.c_str());
    }
    */

    return written;
}

void verbose_templ(char *buffer, const impl::engine_t *engine,
        size_t partition_id, const char *op_name_str, const char *fmt_str,
        const char *data_str, const char *backend_str) {
    int written = 0;
    DPRINT(buffer, DNNL_GRAPH_VERBOSE_BUF_LEN, written, "%s,",
            engine_kind2str(engine->kind()));
    verbose_templ_no_engine_kind(buffer, partition_id, op_name_str, fmt_str,
            data_str, backend_str, written);
}

} // namespace

void partition_info_t::init(const engine_t *engine,
        const compiled_partition_t *compiled_partition) {
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
        str_.resize(DNNL_GRAPH_VERBOSE_BUF_LEN, '\0');
        init_info_partition(engine, compiled_partition, &str_[0]);
        is_initialized_ = true;
    });
}

#endif

} // namespace utils
} // namespace impl
} // namespace graph
} // namespace dnnl

namespace graph = dnnl::graph;

void init_info_partition(const graph::impl::engine_t *engine,
        const graph::impl::compiled_partition_t *compiled_partition,
        char *buffer) {
#if defined(DNNL_GRAPH_DISABLE_VERBOSE)
    UNUSED(engine);
    UNUSED(compiled_partition);
    UNUSED(buffer);
#else
    using namespace graph::impl::utils;
    DECL_DAT_STRS();

    const auto &partition = compiled_partition->src_partition_;
    FMT2STR(fmt_str, DNNL_GRAPH_VERBOSE_FMT_LEN, fmt_written, partition);
    {
        const auto &inputs = compiled_partition->get_inputs();
        const size_t inputs_size = inputs.size();
        for (size_t i = 0; i < inputs_size; ++i) {
            DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, "%s",
                    "in");
            DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, "%d_",
                    (int)i);
            LOGICAL_TENSOR2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    inputs[i]);
            DIM2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    inputs[i]);
            LAYOUT2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    inputs[i]);
            DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, " ");
        }
    }

    {
        const auto &outputs = compiled_partition->get_outputs();
        const size_t outputs_size = outputs.size();
        for (size_t i = 0; i < outputs_size; ++i) {
            DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, "%s",
                    "out");
            DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, "%c_",
                    '0' + static_cast<char>(i));
            LOGICAL_TENSOR2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    outputs[i]);
            DIM2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    outputs[i]);
            LAYOUT2STR(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written,
                    outputs[i]);
            if (i < outputs_size - 1)
                DPRINT(dat_str, DNNL_GRAPH_VERBOSE_DAT_LEN, dat_written, " ");
        }
    }

    verbose_templ(buffer, engine, partition.id(),
            partition.get_ops()[0]->get_name().c_str(), fmt_str, dat_str,
            partition.get_assigned_backend()->get_name().c_str());
#endif
}

const dnnl_graph_version_t *dnnl_graph_version(void) {
    static const dnnl_graph_version_t ver
            = {DNNL_GRAPH_VERSION_MAJOR, DNNL_GRAPH_VERSION_MINOR,
                    DNNL_GRAPH_VERSION_PATCH, DNNL_GRAPH_VERSION_HASH};
    return &ver;
}
