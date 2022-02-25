/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GRAPH_MATMUL_HPP
#define GRAPH_MATMUL_HPP

#include "dnnl_graph_common.hpp"
#include "matmul/matmul.hpp"

namespace benchdnnext {
namespace matmul {

struct matmul_graph_prb_t : public graph_prb_t {
    matmul_graph_prb_t(const ::matmul::prb_t *prb) : spec_(prb) {
        const auto stop_work = [](const fill_status_t s) {
            return s != fill_status::DONE
                    && s != fill_status::UNHANDLED_CONFIG_OPTIONS;
        };

        if (spec_.bia_dt != dt::undef) {
            // bias will become 3rd input (optionall) for MatMul
            // this case is handled in handle_main_op_()
            has_post_bia_ = true;
        }

        ctor_status = handle_main_op_();
        if (stop_work(ctor_status)) return;

        if (benchdnnext::is_low_precision(get_dtypes()))
            // needs to be set before call of post-op handlers
            with_quantization_ = true;

        for (const auto &po : prb->attr.post_ops.entry) {
            if (po.is_eltwise_kind()) {
                has_post_eltwise_ = true;
                ctor_status = handle_elt_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_binary_kind()) {
                has_post_bin_ = true;
                ctor_status = handle_bin_(po);
                if (stop_work(ctor_status)) return;
            } else if (po.is_sum_kind()) {
                has_post_sum_ = true;
                ctor_status = handle_sum_();
                if (stop_work(ctor_status)) return;
            }
        }

        // x8x8bf16 cases
        if (with_typecast(get_dtypes())) {
            ctor_status = handle_typecast_(prb);
            if (stop_work(ctor_status)) return;
        }
        if (with_quantization()) {
            ctor_status = handle_low_precision_(prb);
            if (stop_work(ctor_status)) return;
        }

        ctor_status = fill_status::DONE;
    };

    const std::vector<float> &get_oscales() const noexcept { return oscales_; }

    std::vector<dt> get_dtypes() const {
        return {spec_.src_dt, spec_.wei_dt, spec_.dst_dt};
    }

private:
    struct spec_t {
        spec_t(const ::matmul::prb_t *prb) noexcept;

        bool transpose_a {false};
        bool transpose_b {false};

        dims_t src_dims;
        dims_t wei_dims;
        dims_t dst_dims;
        dims_t bia_dims;

        dt src_dt;
        dt wei_dt;
        dt dst_dt;
        dt bia_dt;

        std::string data_format;

        std::string raw_src_tag;
        std::string raw_wei_tag;
        std::string raw_dst_tag;
    };

    spec_t spec_;
    std::vector<float> oscales_;
    std::vector<int64_t> src_zero_points;
    std::vector<int64_t> wei_zero_points;
    std::vector<int64_t> dst_zero_points;

    po_handlers_t po_handler;

    fill_status_t handle_main_op_();
    fill_status_t handle_sum_();
    fill_status_t handle_typecast_(const ::matmul::prb_t *prb_);
    fill_status_t handle_low_precision_(const ::matmul::prb_t *prb_);
    fill_status_t handle_elt_(const attr_t::post_ops_t::entry_t &po_entry);
    fill_status_t handle_bin_(const attr_t::post_ops_t::entry_t &po_entry);

    dnnl::graph::op::kind get_main_op_kind() const noexcept override {
        return dnnl::graph::op::kind::MatMul;
    }
};

dims_t get_runtime_dims(
        const dims_t &dims, const ::matmul::dims_mask_t &mask) noexcept;
int doit(const ::matmul::prb_t *prb, res_t *res);

} // namespace matmul
} // namespace benchdnnext

#endif
