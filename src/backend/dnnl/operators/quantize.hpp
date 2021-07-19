/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef BACKEND_DNNL_OPERATORS_QUANTIZE_HPP
#define BACKEND_DNNL_OPERATORS_QUANTIZE_HPP

#include <algorithm>
#include <string>
#include <vector>

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/legacy.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

struct quantize_dequantize : public dnnl::reorder, public kernel_base {
    using super = dnnl::reorder;

private:
    dnnl::memory::desc cvt_src_desc_ {};
    dnnl::memory::desc cvt_dst_desc_ {};

    primitive_desc pd_;
    dnnl::reorder prim_;
    dnnl::engine p_engine_;

    static std::vector<float> inverse_scales(const std::vector<float> &scales) {
        std::vector<float> inv_scales;
        inv_scales.reserve(scales.size());
        for (auto &s : scales) {
            // add epsilon to avoid divide zero
            inv_scales.emplace_back(1.f / (s + 1e-9f));
        }
        return inv_scales;
    };

    static std::vector<int32_t> cast_zps(const std::vector<int64_t> &zps) {
        std::vector<int32_t> int32_zps;
        int32_zps.reserve(zps.size());
        for (auto &zp : zps) {
            int32_zps.emplace_back(static_cast<int32_t>(zp));
        }
        return int32_zps;
    }

public:
    impl::status_t compile_impl(const op_t *op, const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using ltw = impl::logical_tensor_wrapper;
        p_engine_ = make_dnnl_engine(*g_engine);
        memory::desc src = make_dnnl_memory_desc(inputs[0]);
        memory::desc dst;

        // the dst layout of reorder primitive can't be any.
        // so, if user set any, we will use the src layout
        if (ltw(outputs[0]).is_any()) {
            dst = src;
            dst.data.data_type = static_cast<dnnl_data_type_t>(
                    ltw(outputs[0]).data_type());
        } else {
            dst = make_dnnl_memory_desc(outputs[0]);
        }

        cvt_src_desc_ = src;
        cvt_dst_desc_ = dst;

        std::string qtype = op->get_attr<std::string>("qtype");
        int64_t axis = op->get_attr<int64_t>("axis");
        const auto &scales = op->get_attr<std::vector<float>>("scales");
        const auto &zps = op->get_attr<std::vector<int64_t>>("zps");

        if ((qtype == "per_tensor" && scales.size() != 1)
                || (qtype == "per_channel" && src.dims()[axis] != scales.size())
                || (zps.size() != scales.size()))
            return status::invalid_argument;

        int mask = 0;
        if (qtype == "per_channel") { mask = 1 << axis; }

        primitive_attr attr;
        if (op->get_kind() == op_kind::Quantize) {
            // inverse the scales, since dnnl multiply the scales to dst
            attr.set_output_scales(mask, inverse_scales(scales));
        } else {
            attr.set_output_scales(mask, scales);
        }

        // If zps are all zero, we don't need to set this attr
        auto pos = std::find_if(zps.begin(), zps.end(),
                [](const int64_t &zp) -> bool { return zp != 0; });

        if (pos != zps.end()) {
            if (qtype == "per_channel") {
                // TODO(qun) reorder doesn't support per_channel zps attr
                // now, we need convert these attrs to tensors and pass them
                // in primitive runtime
                return status::invalid_argument;
            }

            if (op->get_kind() == op_kind::Quantize) {
                attr.set_zero_points(DNNL_ARG_TO, mask, cast_zps(zps));
            } else {
                attr.set_zero_points(DNNL_ARG_FROM, mask, cast_zps(zps));
            }
        }

        pd_ = primitive_desc(p_engine_, src, p_engine_, dst, attr);
        prim_ = super(pd_);
        dnnl::memory::desc opt_dst_desc(pd_.dst_desc());
        auto *ori_dst_lt = const_cast<impl::logical_tensor_t *>(&outputs[0]);
        fill_layout_info(ori_dst_lt, opt_dst_desc);

        return status::success;
    }

    impl::status_t execute_impl(const op_t *op, const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);
        memory src = make_dnnl_memory(
                cvt_src_desc_, p_engine_, inputs[0].get_data_handle());
        memory dst = make_dnnl_memory(
                cvt_dst_desc_, p_engine_, outputs[0].get_data_handle());
        prim_.execute(p_stream, src, dst);
        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
