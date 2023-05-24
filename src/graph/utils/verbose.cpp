/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <memory>
#include <stdlib.h>
#include <vector>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl_graph.h"

#include "common/utils.hpp"
#include "common/verbose.hpp"

#include "graph/interface/backend.hpp"
#include "graph/interface/c_types_map.hpp"
#include "graph/interface/partition.hpp"

#include "graph/utils/debug.hpp"
#include "graph/utils/utils.hpp"
#include "graph/utils/verbose.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "common/dnnl_thread.hpp"
#include "cpu/platform.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

static dnnl::impl::setting_t<uint32_t> verbose {0};

void print_header(int verbosity_flag_hint = verbose_t::none) {
    static std::atomic_flag version_printed = ATOMIC_FLAG_INIT;
    if ((verbose.get() & verbosity_flag_hint)
            && !version_printed.test_and_set()) {
        printf("onednn_graph_verbose,info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        printf("onednn_graph_verbose,info,cpu,runtime:%s,nthr:%d\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime),
                dnnl_get_max_threads());
        printf("onednn_graph_verbose,info,cpu,isa:%s\n",
                cpu::platform::get_isa_info());
#endif
        printf("onednn_graph_verbose,info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
        std::vector<const backend_t *> &backends
                = backend_registry_t::get_singleton().get_registered_backends();
        for (size_t i = 0; i < backends.size() - 1; ++i) {
            backend_t *bkd = const_cast<backend_t *>(backends[i]);
            printf("onednn_graph_verbose,info,backend,%zu:%s\n", i,
                    bkd->get_name().c_str());
        }
    }
}

// verbosity flag is a hint on when to print header so that we print
// header only when something will effectively be logged
uint32_t get_verbose(int verbosity_flag_hint = verbose_t::none) {
#if defined(DISABLE_VERBOSE)
    return verbose_t::none;
#else
    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        static std::string user_opt = getenv_string_user("GRAPH_VERBOSE");

        auto update_kind = [&](const std::string &s, int &k) -> int {
            // Legacy: we accept values 0,1,2
            // 0 and none erase previously set flags, including error
            if (s == "0" || s == "none") return k = impl::verbose_t::none;
            if (s == "1") return k |= impl::verbose_t::exec_profile;
            if (s == "2")
                return k |= impl::verbose_t::exec_profile
                        | impl::verbose_t::create_profile;
            if (s == "all" || s == "-1") return k |= impl::verbose_t::all;
            if (s == "error") return k |= impl::verbose_t::error;
            if (s == "check")
                return k |= impl::verbose_t::create_check
                        | impl::verbose_t::exec_check;
            if (s == "profile")
                return k |= impl::verbose_t::create_profile
                        | impl::verbose_t::exec_profile;
            if (s == "profile_compile")
                return k |= impl::verbose_t::create_profile;
            if (s == "profile_exec") return k |= impl::verbose_t::exec_profile;

            // Unknown option is ignored
            // TODO: exit on unsupported or print a message?
            return k;
        };

        // we always enable error by default
        int val = impl::verbose_t::error;
        for (auto &tok : utils::str_split(user_opt, ','))
            update_kind(tok, val);

        // We parse for explicit flags
        verbose.set(val);
    }

    print_header(verbosity_flag_hint);

    return verbose.get();
#endif
}

bool verbose_has_error() {
    return get_verbose(impl::verbose_t::error) & impl::verbose_t::error;
};
bool verbose_has_create_check() {
    return get_verbose(impl::verbose_t::create_check)
            & impl::verbose_t::create_check;
};
bool verbose_has_create_profile() {
    return get_verbose(impl::verbose_t::create_profile)
            & impl::verbose_t::create_profile;
};
bool verbose_has_exec_profile() {
    return get_verbose(impl::verbose_t::exec_profile)
            & impl::verbose_t::exec_profile;
};

#if defined(DISABLE_VERBOSE)
void partition_info_t::init(
        const engine_t *engine, const compiled_partition_t *partition) {
    UNUSED(engine);
    UNUSED(partition);
}

#else

namespace {

std::string logical_tensor2dim_str(const logical_tensor_t &logical_tensor) {
    std::string s;

    auto lt = logical_tensor_wrapper_t(logical_tensor);

    s += ":";
    s += std::to_string(lt.dims()[0]);
    for (int d = 1; d < lt.ndims(); ++d)
        s += ("x" + std::to_string(lt.dims()[d]));

    return s;
}

std::string logical_tensor2layout_str(const logical_tensor_t &logical_tensor) {
    std::string s;

    auto lt = logical_tensor_wrapper_t(logical_tensor);

    s += ":";
    if (lt.layout_type() == layout_type::strided) {
        const auto strides = lt.strides();
        for (int i = 0; i < lt.ndims() - 1; ++i) {
            s += std::to_string(strides[i]);
            s += "s";
        }
        s += std::to_string(strides[lt.ndims() - 1]);
    } else if (lt.layout_type() == layout_type::opaque) {
        s += std::to_string(lt.layout_id());
    } else if (lt.layout_type() == layout_type::any) {
        s += "any";
    } else {
        assert(!"layout type must be any, strided or opaque.");
    }

    return s;
}

std::string logical_tensor2str(const logical_tensor_t &logical_tensor) {
    std::string s;

    s += std::string(data_type2str(logical_tensor.data_type));
    s += ":";
    s += std::to_string(logical_tensor.id);
    s += ":";
    s += std::string(layout_type2str(logical_tensor.layout_type));
    s += ":";
    s += std::string(property_type2str(logical_tensor.property));

    return s;
}

std::string partition2fmt_str(const partition_t &partition) {
    std::string s;

    const std::vector<std::shared_ptr<graph::op_t>> &operators
            = partition.get_ops();
    const size_t num_operator = operators.size();
    if (num_operator == 0) return s;

    bool data_filled = false;
    bool filter_filled = false;
    for (size_t i = 0; i < num_operator; ++i) {
        const std::shared_ptr<op_t> &op = operators[i];
        if (op->has_attr(op_attr::data_format)) {
            // If the first i ops have no data_format, empty string with suffix
            // `;` should be printed out for each of them.
            if (!data_filled) {
                s += "data:";
                for (size_t ii = 0; ii < i; ++ii)
                    s += ";";
                // Indicates that at least one op in the list have data format
                // spec.
                data_filled = true;
            }
            const auto data_format
                    = op->get_attr<std::string>(op_attr::data_format);
            if (i == num_operator - 1) {
                s += data_format;
                s += " ";
            } else {
                s += data_format;
                s += ";";
            }
        } else if (data_filled) {
            // If at least one op have data format, op without format spec
            // should give `;` except the last one of data which should give
            // ` `.
            if (i == num_operator - 1) {
                s += " ";
            } else {
                s += ";";
            }
        }
    }
    for (size_t i = 0; i < num_operator; ++i) {
        const std::shared_ptr<op_t> &op = operators[i];
        if (op->has_attr(op_attr::weights_format)) {
            if (!filter_filled) {
                s += "filter:";
                for (size_t ii = 0; ii < i; ++ii)
                    s += ";";
                filter_filled = true;
            }
            const auto filter_format
                    = op->get_attr<std::string>(op_attr::weights_format);
            if (i == num_operator - 1) {
                s += filter_format;
                s += " ";
            } else {
                s += filter_format;
                s += ";";
            }
        } else if (filter_filled) {
            s += ";";
        }
    }

    return s;
}

std::string init_info_partition(const engine_t *engine,
        const compiled_partition_t *compiled_partition) {
    std::stringstream ss;

    const auto &partition = compiled_partition->src_partition();

    ss << std::string(engine_kind2str(engine->kind())) << "," << partition.id()
       << "," << partition_kind2str(partition.get_kind()) << ",";

    const std::vector<std::shared_ptr<graph::op_t>> &operators
            = partition.get_ops();
    const size_t num_operators = operators.size();
    for (size_t i = 0; i < num_operators; ++i) {
        ss << operators[i]->get_name()
           << ((i == num_operators - 1) ? "," : ";");
    }

    ss << partition2fmt_str(partition) << ",";
    {
        const auto &inputs = compiled_partition->get_inputs();
        const size_t inputs_size = inputs.size();
        for (size_t i = 0; i < inputs_size; ++i) {
            ss << "in" << i << "_" << logical_tensor2str(inputs[i])
               << logical_tensor2dim_str(inputs[i])
               << logical_tensor2layout_str(inputs[i]) << " ";
        }
    }

    {
        const auto &outputs = compiled_partition->get_outputs();
        const size_t outputs_size = outputs.size();
        for (size_t i = 0; i < outputs_size; ++i) {
            ss << "out" << i << "_" << logical_tensor2str(outputs[i])
               << logical_tensor2dim_str(outputs[i])
               << logical_tensor2layout_str(outputs[i]);
            if (i < outputs_size - 1) ss << " ";
        }
    }

    ss << ",fpm:" << fpmath_mode2str(partition.get_pimpl()->get_fpmath_mode());

    ss << "," << partition.get_assigned_backend()->get_name();

    return ss.str();
}

} // namespace

void partition_info_t::init(const engine_t *engine,
        const compiled_partition_t *compiled_partition) {
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
        str_ = init_info_partition(engine, compiled_partition);
        is_initialized_ = true;
    });
}

#endif

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
