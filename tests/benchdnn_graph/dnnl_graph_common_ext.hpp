/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#ifndef DNNL_GRAPH_COMMON_EXT_HPP
#define DNNL_GRAPH_COMMON_EXT_HPP

#include "dnnl_graph_common.hpp"

namespace benchdnnext {

struct id_manager_t {
    id_manager_t() : frozen_(false) {};

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

    void freeze() { frozen_ = true; }

private:
    std::map<std::string, size_t> knots_;
    // indicates that the graph is frozen
    bool frozen_;
};

struct tensor_descs_t {
    using property_type = dnnl::graph::logical_tensor::property_type;
    tensor_descs_t() = default;

    template <typename... Args>
    void emplace(std::string str, Args... args) {
        dnnl::graph::logical_tensor t(idmgr_[str], std::forward<Args>(args)...);
        map_.emplace(str, t);
    }

    void emplace(std::string str, dt dtype, const dims_t &adims,
            const std::string &atag,
            property_type ptype = property_type::undef) {
        size_t ndims = adims.size();
        const std::string dnnl_fmt_tag_str
                = normalize_tag(atag, static_cast<int>(ndims));
        const dnnl_format_tag_t fmt_tag = dnnl_fmt_str2tag(dnnl_fmt_tag_str);
        if (fmt_tag == dnnl_format_tag_undef) {
            []() {
                SAFE(FAIL, CRIT);
                return 0;
            }();
            return;
        }

        if (fmt_tag == dnnl_format_tag_any) {
            emplace(str, dtype, adims, lt::strided, ptype);
            return;
        }

#ifndef DNNL_GRAPH_LAYOUT_DEBUG
        if (fmt_tag > dnnl_abcdefghijlk) {
            []() {
                BENCHDNN_PRINT(0, "error: %s\n",
                        "to use dnnl opaque blocked formats, please build the "
                        "library with \"DNNL_GRAPH_LAYOUT_DEBUG=ON\"");
                SAFE(FAIL, CRIT);
                return 0;
            }();
        }
#endif // DNNL_GRAPH_LAYOUT_DEBUG

        const std::string ou_fmt_str = get_ou_format(dnnl_fmt_tag_str);

        static_assert(DNNL_GRAPH_MAX_NDIMS == DNNL_MAX_NDIMS,
                "Maximum number of dimensions of primitive and graph is not "
                "the same.");

        if (fmt_tag <= dnnl_abcdefghijlk) {
            dims_t strides(ndims, 0);
            dim_t acc = 1;
            // if the layout can be described with strides, let's calculate
            // the strides according to the given tag.
            for (int d = static_cast<int>(ndims) - 1; d >= 0; --d) {
                const size_t coord = static_cast<size_t>(ou_fmt_str[d] - 'a');
                strides[coord] = acc;
                acc *= adims[coord];
            }
            emplace(str, dtype, adims, strides, ptype);
        } else {
            const size_t dnnl_layout_id
                    = encode_dnnl_layout(static_cast<size_t>(fmt_tag));
            dnnl::graph::logical_tensor t(
                    idmgr_[str], dtype, adims, dnnl_layout_id, ptype);
            map_.emplace(str, t);
        }
    }

    dnnl::graph::logical_tensor operator[](const std::string &str) {
        return map_.at(str);
    }

private:
    std::map<std::string, dnnl::graph::logical_tensor> map_;
    id_manager_t idmgr_;
};

struct graph_prb_t {
    using dt = dnnl::graph::logical_tensor::data_type;
    using lt = dnnl::graph::logical_tensor::layout_type;
    using fpmath_mode = dnnl::graph::graph::fpmath_mode;

    dnnl::graph::graph to_graph(fpmath_mode mode = fpmath_mode::strict) {
        const dnnl::graph::engine &engine = benchdnnext::get_test_engine();
        dnnl::graph::graph graph(engine.get_kind(), mode);
        for (auto &&op : ops_)
            graph.add_op(op);
        return graph;
    }

    bool has_post_bia() const noexcept { return has_post_bia_; }
    bool has_post_bin() const noexcept { return has_post_bin_; }
    bool has_post_dw() const noexcept { return has_post_dw_; }
    bool has_post_sum() const noexcept { return has_post_sum_; }
    bool has_post_eltwise() const noexcept { return has_post_eltwise_; }
    bool with_quantization() const noexcept { return with_quantization_; }

    fill_status_t ctor_status;

protected:
    std::vector<dnnl::graph::op> ops_;
    tensor_descs_t tensor_descs_;
    std::unordered_map<std::string, std::vector<std::string>> tensor_id;

    std::vector<std::string> curr_out_map_ids_;

    bool has_post_bia_ {false};
    bool has_post_bin_ {false};
    bool has_post_dw_ {false};
    bool has_post_sum_ {false};
    bool has_post_eltwise_ {false};
    bool with_quantization_ {false};
};

} // namespace benchdnnext
#endif