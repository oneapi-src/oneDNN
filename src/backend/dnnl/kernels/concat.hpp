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

#ifndef BACKEND_DNNL_KERNELS_CONCAT_HPP
#define BACKEND_DNNL_KERNELS_CONCAT_HPP

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "interface/backend.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/f32_kernel_resource.hpp"
#include "backend/dnnl/scratchpad.hpp"
#include "backend/dnnl/thread_local_cache.hpp"
#include "backend/dnnl/utils.hpp"

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

namespace cat {
// Since concat accepts variadic inputs, so we use `kBase` as the key for dst
// memory while using `kBase + (i + 1)` for variadic inputs.
enum mem_keys { kBase };
} // namespace cat

struct concat : public dnnl::concat, public kernel_base {
    using super = dnnl::concat;

private:
    primitive_desc pd_;
    dnnl::concat prim_;

    dnnl::engine p_engine_;

    std::vector<dnnl::reorder::primitive_desc> src_reorder_pds_;
    dnnl::reorder::primitive_desc dst_reorder_pd_;

    registry_t registry_;
    std::function<std::shared_ptr<f32_concat_resource_t>()> resource_ctor_;
    f32_concat_resource_t::desc_t res_desc_;

public:
    ~concat() override {
        thread_local_cache_t<f32_concat_resource_t> res_cache;
        res_cache.remove_if_exist(reinterpret_cast<size_t>(this));
    }

    impl::status_t compile_impl(const op_t *op, const impl::engine_t *g_engine,
            const std::vector<impl::logical_tensor_t> &inputs,
            const std::vector<impl::logical_tensor_t> &outputs) override {
        using lt = impl::logical_tensor_t;

        p_engine_ = make_dnnl_engine(*g_engine);

        const size_t expected_outs = 1;
        if (inputs.empty() || expected_outs != outputs.size())
            return status::invalid_argument;

        const auto rank = outputs.front().ndims;
        const auto res
                = utils::try_reverse_axis(op->get_attr<int64_t>("axis"), rank);
        if (!res.first) return status::invalid_argument;
        const auto axis = res.second;

        // Here we force to use plain-in-plain-out (acdb) for 4D case to make
        // sure good performance of DensenNet121 (reducing reorder overhead).
        // But for other cases like 2D/3D (e.g. DLRM), we just use default
        // format since there may be followed by a non-DNNL op which requires an
        // input with default format. Anyway it looks like a bit tricky.
        auto get_forced_format_tag = [](const dims &in_dims) -> format_tag {
            if (in_dims.size() == 4)
                return format_tag::acdb;
            else
                return get_default_format(in_dims);
        };

        std::vector<memory::desc> src_mds;
        src_mds.reserve(inputs.size());
        res_desc_.cvt_src_.reserve(inputs.size());
        std::for_each(inputs.cbegin(), inputs.cend(),
                [this, &src_mds, &get_forced_format_tag](const lt &in) {
                    auto tmp_desc = make_dnnl_memory_desc(in);
                    this->res_desc_.cvt_src_.push_back(tmp_desc);
                    src_mds.push_back(
                            memory::desc {tmp_desc.dims(), tmp_desc.data_type(),
                                    get_forced_format_tag(tmp_desc.dims())});
                });

        res_desc_.cvt_dst_ = make_dnnl_memory_desc(outputs.front());
        memory::desc dst_permute_md {res_desc_.cvt_dst_.dims(),
                res_desc_.cvt_dst_.data_type(),
                get_forced_format_tag(res_desc_.cvt_dst_.dims())};

        pd_ = primitive_desc(
                dst_permute_md, static_cast<int>(axis), src_mds, p_engine_);
        prim_ = super(pd_);

        registrar_t registrar = registry_.registrar();
        res_desc_.opt_src_.reserve(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto tmp_opt_src_desc = pd_.src_desc(static_cast<int>(i));
            res_desc_.opt_src_.push_back(tmp_opt_src_desc);
            if (res_desc_.cvt_src_[i] != tmp_opt_src_desc) {
                registrar.book(static_cast<size_t>(cat::kBase + i + 1),
                        tmp_opt_src_desc.get_size());
                src_reorder_pds_.emplace_back(dnnl::reorder::primitive_desc(
                        p_engine_, res_desc_.cvt_src_[i], p_engine_,
                        tmp_opt_src_desc));
            }
        }

        res_desc_.opt_dst_ = pd_.dst_desc();
        if (impl::logical_tensor_wrapper_t(outputs.at(0)).is_any()) {
            res_desc_.cvt_dst_ = res_desc_.opt_dst_;
        }

        if (res_desc_.opt_dst_ != res_desc_.cvt_dst_) {
            registrar.book(cat::kBase, res_desc_.opt_dst_.get_size());
            dst_reorder_pd_ = dnnl::reorder::primitive_desc(p_engine_,
                    res_desc_.opt_dst_, p_engine_, res_desc_.cvt_dst_);
        }

        lt *orig_dst_lt = const_cast<lt *>(&outputs.front());
        fill_layout_info(orig_dst_lt, pd_.dst_desc());

        resource_ctor_ = [this]() {
            return std::make_shared<f32_concat_resource_t>(
                    this->res_desc_, this->p_engine_);
        };

        return status::success;
    }

    impl::status_t execute_impl(const op_t *op, const impl::stream_t *g_stream,
            const std::vector<impl::tensor_t> &inputs,
            const std::vector<impl::tensor_t> &outputs) override {
        UNUSED(op);
        dnnl::stream p_stream = make_dnnl_stream(p_engine_, *g_stream);

        // each thread's own local resource
        thread_local_cache_t<f32_concat_resource_t> res_cache;
        f32_concat_resource_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);

        impl::allocator_t *g_alloc_ = g_stream->get_engine()->get_allocator();
        temporary_scratchpad_t scratchpad(
                registry_.size(), p_engine_, *g_alloc_);
        grantor_t grantor = registry_.grantor(scratchpad.get_buffer());

        size_t idx = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            res->cvt_src_mems_[i].set_data_handle(
                    inputs.at(i).get_data_handle());
            if (res_desc_.cvt_src_[i] != res_desc_.opt_src_[i]) {
                res->opt_src_mems_[i].set_data_handle(
                        grantor.get(static_cast<size_t>(cat::kBase + i + 1)));
                dnnl::reorder(src_reorder_pds_[idx++])
                        .execute(p_stream, res->cvt_src_mems_[i],
                                res->opt_src_mems_[i]);
            } else {
                res->opt_src_mems_[i].set_data_handle(
                        res->cvt_src_mems_[i].get_data_handle());
            }
        }

        res->cvt_dst_mem_.set_data_handle(outputs.at(0).get_data_handle());
        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            res->opt_dst_mem_.set_data_handle(grantor.get(cat::kBase));
        } else {
            res->opt_dst_mem_.set_data_handle(
                    res->cvt_dst_mem_.get_data_handle());
        }

        prim_.execute(p_stream, res->exec_args_);

        if (res_desc_.cvt_dst_ != res_desc_.opt_dst_) {
            dnnl::reorder(dst_reorder_pd_)
                    .execute(p_stream, res->opt_dst_mem_, res->cvt_dst_mem_);
        }

        return status::success;
    }
};

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
