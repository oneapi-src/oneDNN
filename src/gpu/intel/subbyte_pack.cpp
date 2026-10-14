/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "gpu/intel/subbyte_pack.hpp"

#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

status_t subbyte_pack_desc_t::init(const memory_desc_t &dst_md) {
    const memory_desc_wrapper mdw(dst_md);
    const dim_t elems_per_byte
            = into<dim_t>(types::bytes_to_elements(dst_md.data_type, 1));
    if (elems_per_byte == 1) return status::success;
    if (dst_md.format_kind != format_kind::blocked || dst_md.offset0 != 0)
        return status::unimplemented;
    if (mdw.has_runtime_dims_or_strides()) return status::unimplemented;

    auto layout = block_layout_t(mdw);
    for (auto &block : layout)
        block.dim_idx = 0;
    layout = layout.normalized();

    if (layout.empty()) return status::success;

    conf_.bits = into<int>(8 / elems_per_byte);
    span_ = mdw.span();
    conf_.use_int32_offset = span_ <= INT32_MAX;
    conf_.require_stateless_addressing = span_ > UINT32_MAX;

    if (layout.size() > max_ndims) return status::unimplemented;
    for (const auto &block : layout) {
        dims_.array[conf_.ndims] = block.block;
        strides_.array[conf_.ndims] = static_cast<dim_t>(block.stride);
        conf_.ndims++;
    }

    if (strides_.array[0] != 1) return status::unimplemented;
    for (int d = 1; d < conf_.ndims; d++)
        if (strides_.array[d] % elems_per_byte) return status::unimplemented;

    gws_[0] = utils::div_up(dims_.array[0], elems_per_byte);
    for (int d = 1; d < conf_.ndims; d++)
        gws_[d == 1 ? 1 : 2] *= dims_.array[d];

    return status::success;
}

const std::vector<const char *> &
subbyte_pack_desc_t::conf_t::get_kernel_names() const {
    static const std::vector<const char *> names {"subbyte_pack"};
    return names;
}

status_t subbyte_pack_desc_t::conf_t::create_generator(
        const engine_t &engine, compute::kernel_bundle_t &bundle) const {
    compute::kernel_ctx_t kernel_ctx;
    kernel_ctx.define_int("SUBBYTE_PACK_BITS", bits);
    kernel_ctx.define_int("SUBBYTE_PACK_NDIMS", ndims);
    kernel_ctx.use_int32_offset(use_int32_offset);
    kernel_ctx.require_stateless_addressing(require_stateless_addressing);
    return engine.create_kernel_bundle(bundle, get_kernel_names(), kernel_ctx);
}

status_t subbyte_pack_t::create(const subbyte_pack_desc_t &desc,
        primitive_t &primitive, impl::engine_t *engine) {
    CHECK(primitive.create_kernel(
            engine, kernel_, "subbyte_pack", desc.conf()));
    if (!kernel_) return status::runtime_error;

    nd_range_ = desc.nd_range();
    dims_ = desc.dims();
    strides_ = desc.strides();
    return status::success;
}

status_t subbyte_pack_t::operator()(const exec_ctx_t &ctx,
        const memory_storage_t &src, const memory_storage_t &dst) const {
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);
    arg_list.set(2, dims_);
    arg_list.set(3, strides_);

    return primitive_t::large_parallel_for(
            ctx, nd_range_, kernel_, arg_list, 4);
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
