/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/intel/ocl/ref_reorder.hpp"

#include "common/utils.hpp"
#include "gpu/intel/ocl/utils.hpp"
namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

using namespace dnnl::impl::memory_tracking::names;

status_t ref_reorder_t::pd_t::init_conf(impl::engine_t *engine) {
    using namespace format_tag;
    using namespace data_type;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    status_t status = status::success;

    const auto &padded_dims = dst_mdw.padded_dims();
    conf.src_quant = {attr(), src_mdw, DNNL_ARG_SRC};
    conf.dst_quant = {attr(), dst_mdw, DNNL_ARG_DST};
    conf.sum_quant = {attr()};
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(padded_dims, conf.ndims);

    conf.sub_group_size = 1;

    if (conf.nelems == 0) return status::success;

    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);
    conf.dispatch = compute_engine->create_dispatch(dst_mdw.md_);
    conf.subbyte_pack
            = utils::one_of(dst_mdw.data_type(), u4, s4, f4_e2m1, f4_e3m0);

    dim_t blocks[MAX_NDIMS] = {1, 1, 0, 0, 0, 0};
    for (int i = 0; i < MAX_NDIMS; ++i) {
        auto dim_str = utils::format("D%d", i);
        if (i < dst_mdw.ndims()) {
            dim_t dim = padded_dims[i];
            // if needed to align vectorized dim with vector size, pad that dim again
            conf.dispatch.define_dim(dim_str, i, dim, blocks[i]);
        } else {
            conf.dispatch.define_dim(dim_str, 1);
        }
    }

    conf.dispatch.generate();
    return status;
}

status_t ref_reorder_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace format_tag;
    using namespace data_type;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    if (conf.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.add_option("-cl-std=CL2.0");

    conf.src_quant.define_macros(kernel_ctx, "SRC");
    conf.dst_quant.define_macros(kernel_ctx, "DST");
    conf.sum_quant.define_macros(kernel_ctx, "SUM");

    def_dispatch(kernel_ctx, conf.dispatch);

    kernel_ctx.define_int("REF_REORDER", 1);
    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);

    auto dst_rnd_mode = attr()->rounding_mode_.get(DNNL_ARG_DST);
    kernel_ctx.define_int(
            "WITH_SROUND", dst_rnd_mode == rounding_mode::stochastic);
    kernel_ctx.define_int("DST_DT_DIGITS",
            dnnl::impl::types::digits<uint32_t>(conf.dst_md_info.data_type));

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    return status::success;
}

void ref_reorder_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    if (conf.subbyte_pack) {
        scratchpad.book(memory_tracking::names::key_reorder_space, conf.nelems,
                sizeof(char), OCL_BUFFER_ALIGNMENT);
    }
    if (conf.src_quant.with_scale()) {
        scratchpad.book(memory_tracking::names::key_reorder_src_scales,
                conf.src_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
    if (conf.dst_quant.with_scale()) {
        scratchpad.book(memory_tracking::names::key_reorder_dst_scales,
                conf.dst_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
}

status_t ref_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);
    auto tmp = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_reorder_space);

    const auto &conf = pd()->conf;
    if (conf.nelems == 0) return status::success;

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(conf.subbyte_pack ? *tmp : dst);

    arg_list.append(conf.src_quant.scales(ctx));
    arg_list.append(conf.src_quant.zero_points(ctx));
    arg_list.append(conf.dst_quant.scales(ctx));
    arg_list.append(conf.dst_quant.zero_points(ctx));

    arg_list.append(conf.sum_quant.scales());
    arg_list.append(conf.sum_quant.zero_points());

    if (!pd()->attr()->rounding_mode_.has_default_values())
        arg_list.append(CTX_IN_STORAGE(DNNL_ARG_ATTR_ROUNDING_SEED));

    auto nd_range = conf.dispatch.nd_range();
    CHECK(large_parallel_for(
            ctx, nd_range, kernels_[0], arg_list, arg_list.nargs()));

    if (conf.subbyte_pack) {
        compute::kernel_arg_list_t repack_arg_list;
        repack_arg_list.set(0, *tmp);
        repack_arg_list.set(1, dst);
        repack_arg_list.set(2, into<dim_t>(conf.nelems));
        repack_arg_list.set(3, 4);
        compute::range_t repack_gws((conf.nelems * 4 + 7) / 8);
        compute::nd_range_t repack_nd_range(repack_gws);
        CHECK(large_parallel_for(
                ctx, repack_nd_range, kernels_[1], repack_arg_list, 4));
    }
    CHECK(pd()->maybe_exec_zp_precompute_conv(ctx, zp_precomp_conv_));
    return status::success;
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
