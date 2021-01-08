/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

#include "oneapi/dnnl/dnnl_graph.hpp"

#ifdef DNNL_GRAPH_WITH_SYCL
#include <CL/sycl.hpp>
using namespace cl::sycl;
#endif

using namespace dnnl::graph;

engine::kind parse_engine_kind(int argc, char **argv) {
    // Returns default engine kind, i.e. CPU, if none given
    if (argc == 1) {
        return engine::kind::cpu;
    } else if (argc == 2) {
        // Checking the engine type, i.e. CPU or GPU
        std::string engine_kind_str = argv[1];
        if (engine_kind_str == "cpu") {
            return engine::kind::cpu;
        } else if (engine_kind_str == "gpu") {
            return engine::kind::gpu;
        } else {
            throw std::runtime_error("only support cpu and gpu engine\n");
        }
    }
    // If all above fails, the example should be ran properly
    std::cout << "Inappropriate engine kind." << std::endl
              << "Please run the example like this: " << argv[0] << " [cpu|gpu]"
              << "." << std::endl;
    exit(1);
}

inline dnnl_graph_dim_t product(const std::vector<int64_t> &dims) {
    return dims.empty()
            ? 0
            : std::accumulate(dims.begin(), dims.end(), (dnnl_graph_dim_t)1,
                    std::multiplies<dnnl_graph_dim_t>());
}

void *allocate(size_t n, allocator::attribute attr) {
    (void)attr;
    return malloc(n);
}

void deallocate(void *ptr) {
    free(ptr);
}

struct logical_id_manager {
    static logical_id_manager &get() {
        static logical_id_manager id_mgr;
        return id_mgr;
    }

    size_t operator[](const std::string &arg) {
        const auto &it = knots_.find(arg);
        if (it != knots_.end()) return it->second;
        if (frozen_) {
            std::cout << "Unrecognized argument [" << arg << "]!\n";
            std::abort();
        }
        const auto &new_it = knots_.emplace(arg, knots_.size());
        if (new_it.second) {
            return new_it.first->second;
        } else {
            std::cout << "New argument [" << arg
                      << "] is failed to be added to knots.\n";
            std::abort();
        }
    }

    void freeze() { frozen_ = false; }

private:
    logical_id_manager() : frozen_(false) {};

    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

#ifdef DNNL_GRAPH_WITH_SYCL
template <typename dtype>
void fill_buffer(
        sycl::queue &q, float *usm_buffer, size_t length, dtype value) {
    q.parallel_for(range<1>(length), [=](id<1> i) {
         int idx = (int)i[0];
         usm_buffer[idx] = value;
     }).wait();
}

void *sycl_malloc_wrapper(
        size_t n, const void *dev, const void *ctx, allocator::attribute attr) {
    return malloc_device(n, *static_cast<const cl::sycl::device *>(dev),
            *static_cast<const cl::sycl::context *>(ctx));
}

void sycl_free_wrapper(void *ptr, const void *context) {
    free(ptr, *static_cast<const cl::sycl::context *>(context));
}
#endif
#endif
