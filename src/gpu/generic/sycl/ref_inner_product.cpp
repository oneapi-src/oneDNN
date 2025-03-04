/*******************************************************************************
* Copyright 2024 Intel Corporation
* Copyright 2024 Codeplay Software Limited
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

#include "gpu/generic/sycl/ref_inner_product.hpp"

namespace dnnl::impl::gpu::generic::sycl {

namespace detail {

// TODO: this seems like a function generic enough to go a common utils file.
status_t get_primitive_descriptor(op_desc_t *op_desc,
        const primitive_attr_t *attributes, impl::engine_t *engine,
        std::shared_ptr<primitive_desc_t> &pd) {

    primitive_desc_iterator_t it(engine, op_desc, attributes, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    while (++it != it.end()) {
        if (*it) {
            pd = *it;
            return status::success;
            ;
        }
    }

    return status::out_of_memory;
}

status_t init_matmul_pd(impl::engine_t *engine,
        const primitive_attr_t *attributes, const memory_desc_t *src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *bias_desc,
        const memory_desc_t *dst_desc,
        std::shared_ptr<primitive_desc_t> &matmul_pd) {

    matmul_desc_t matmul_desc;
    CHECK(matmul_desc_init(
            &matmul_desc, src_desc, weights_desc, bias_desc, dst_desc));

    CHECK(get_primitive_descriptor(reinterpret_cast<op_desc_t *>(&matmul_desc),
            attributes, engine, matmul_pd));
    return status::success;
}

status_t init_reorder_pd(impl::engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *dst_md,
        std::shared_ptr<primitive_desc_t> &reorder_pd) {
    // This will always be a gpu-gpu copy in our case.
    CHECK(reorder_primitive_desc_create(reorder_pd, engine, src_md, dst_md));
    return status::success;
}

void get_flattened_dimension(const dims_t &dims, dims_t &squished_dims,
        dim_t ndims, bool swap_dimensions) {
    int64_t accum = 1;
    for (dim_t i = 1; i < ndims; i++) {
        accum *= dims[i];
    }
    if (swap_dimensions) {
        squished_dims[0] = accum;
        squished_dims[1] = dims[0];
    } else {
        squished_dims[0] = dims[0];
        squished_dims[1] = accum;
    }
}

std::vector<int> get_dim_order(int ndims, const dims_t strides) {
    std::vector<int> order(ndims);
    for (int i = 0; i < ndims; ++i) {
        order[i] = i;
    }

    std::sort(order.begin(), order.end(),
            [&strides](size_t i, size_t j) { return strides[i] < strides[j]; });

    return order;
}

bool strides_in_desc_order(const dims_t &strides, dim_t ndims) {
    bool are_descending = true;
    for (int i = 1; i < ndims; i++) {
        are_descending = are_descending & (strides[i] < strides[i - 1]);
    }
    return are_descending;
}

} // namespace detail

bool ref_inner_product_fwd_t::pd_t::check_if_dtypes_valid(
        const data_type_t &src_dt, const data_type_t &dst_dt,
        const data_type_t &bias_dt, const data_type_t &weight_dt) const {
    using namespace data_type;
    return (utils::one_of(src_dt, f32) && utils::one_of(weight_dt, f32)
                   && utils::one_of(dst_dt, f32)
                   && utils::one_of(bias_dt, f32, undef))
            || (utils::one_of(src_dt, f16) && utils::one_of(weight_dt, f16)
                    && utils::one_of(dst_dt, f16, f32, s8, u8)
                    && utils::one_of(bias_dt, f16, f32, undef))
            || (utils::one_of(src_dt, u8, s8) && utils::one_of(weight_dt, s8)
                    && utils::one_of(dst_dt, u8, s8, s32, bf16, f32)
                    && utils::one_of(bias_dt, u8, s8, s32, bf16, f32, undef))
            || (utils::one_of(src_dt, bf16) && utils::one_of(weight_dt, bf16)
                    && utils::one_of(dst_dt, f32, bf16)
                    && utils::one_of(bias_dt, f32, bf16, undef));
}

bool ref_inner_product_bwd_data_t::pd_t::check_bwd_data_dtypes(
        const data_type_t &src_dt, const data_type_t &dst_dt,
        const data_type_t &weight_dt) const {
    using namespace data_type;
    return (utils::one_of(src_dt, f32) && utils::one_of(dst_dt, f32, f16, bf16)
                   && utils::one_of(weight_dt, f32, bf16, f16))
            || (utils::one_of(src_dt, bf16) && utils::one_of(dst_dt, bf16)
                    && utils::one_of(weight_dt, bf16))
            || (utils::one_of(src_dt, f16) && utils::one_of(dst_dt, f16)
                    && utils::one_of(weight_dt, f16));
}

bool ref_inner_product_bwd_weights_t::pd_t::check_bwd_weights_dtypes(
        const data_type_t &src_dt, const data_type_t &dst_dt,
        const data_type_t &weight_dt, const data_type_t &bias_dt) const {
    using namespace data_type;
    return (utils::one_of(src_dt, f32) && utils::one_of(dst_dt, f32)
                   && utils::one_of(weight_dt, f32)
                   && utils::one_of(bias_dt, f32, undef))
            || (utils::one_of(src_dt, bf16) && utils::one_of(dst_dt, bf16)
                    && utils::one_of(weight_dt, f32, bf16)
                    && utils::one_of(bias_dt, f32, bf16, undef))
            || (utils::one_of(src_dt, f16) && utils::one_of(dst_dt, f16)
                    && utils::one_of(weight_dt, f32, f16)
                    && utils::one_of(bias_dt, f32, f16, undef));
}

status_t ref_inner_product_bwd_weights_t::pd_t::init_reduction_pd(
        impl::engine_t *engine, const memory_desc_t *src_desc,
        const memory_desc_t *dest_desc) {
    reduction_desc_t reduction_descriptor;
    //diff_bias is 1D, diff_dst will be 2D, reshape diff_bias to 1xOC
    dims_t diff_bias_reshaped_dims {1, dest_desc->dims[0]};
    memory_desc_t diff_bias_reshaped;
    CHECK(memory_desc_init_by_tag(diff_bias_reshaped, 2,
            diff_bias_reshaped_dims, dest_desc->data_type, format_tag::ab));
    CHECK(reduction_desc_init(&reduction_descriptor, alg_kind::reduction_sum,
            src_desc, &diff_bias_reshaped, 0.0f, 0.0f));
    CHECK(detail::get_primitive_descriptor(
            reinterpret_cast<op_desc_t *>(&reduction_descriptor), attr(),
            engine, reduction_pd));
    return status::success;
}

status_t ref_inner_product_fwd_t::pd_t::init(impl::engine_t *engine) {

    const bool ok = (set_default_params() == status::success);
    VDISPATCH_INNER_PRODUCT(ok, VERBOSE_UNSUPPORTED_TAG);

    auto bias_dt
            = with_bias() ? arg_md(DNNL_ARG_BIAS)->data_type : data_type::undef;

    auto src_wrapper = memory_desc_wrapper(src_md());
    auto wei_wrapper = memory_desc_wrapper(weights_md());
    auto dst_wrapper = memory_desc_wrapper(dst_md());

    VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(
            check_if_dtypes_valid(src_wrapper.data_type(),
                    dst_wrapper.data_type(), bias_dt, wei_wrapper.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_INNER_PRODUCT(
            (attr_.set_default_formats(dst_md()) == status::success),
            "Failed to set default formats");
    VDISPATCH_INNER_PRODUCT(
            sycl_post_ops_t::post_ops_ok(attr()), "Unsupported Post Ops");
    VDISPATCH_INNER_PRODUCT(src_wrapper.is_plain(),
            "source memory descriptor is not a plain memory format");
    VDISPATCH_INNER_PRODUCT(wei_wrapper.is_plain(),
            "weight memory descriptor is not a plain memory format");
    VDISPATCH_INNER_PRODUCT(dst_wrapper.is_plain(),
            "destination memory descriptor is not a plain memory format");

    // if anything contains a zero dimension, return success as this will be converted
    // to a no-op

    if (src_wrapper.has_zero_dim() || wei_wrapper.has_zero_dim()
            || dst_wrapper.has_zero_dim()) {
        has_zero_dim = true;
        return status::success;
    }

    memory_desc_t src_reshaped;
    memory_desc_t weights_reshaped;
    memory_desc_t dst_reshaped;
    format_tag_t src_format_tag = format_tag::ab;
    format_tag_t wei_format_tag = format_tag::ba;
    format_tag_t dst_format_tag = format_tag::ab;
    memory_desc_t bias_reshaped = types::zero_md();

    auto src_strides = src_wrapper.strides();
    auto ndims = src_wrapper.ndims();

    // If it's two dimensional, and not properly ordered, rely on TT/TN/NT GEMMs rather than reorders
    if (ndims == 2) {
        if (src_strides[1] != 1) { src_format_tag = format_tag::ba; }

        if (wei_wrapper.strides()[1] != 1) { wei_format_tag = format_tag::ab; }
    }

    if (dst_wrapper.strides()[1] != 1) { dst_format_tag = format_tag::ba; }

    dims_t src_squished_dims;
    dims_t wei_squished_dims;
    detail::get_flattened_dimension(
            src_wrapper.dims(), src_squished_dims, src_wrapper.ndims());
    detail::get_flattened_dimension(
            wei_wrapper.dims(), wei_squished_dims, wei_wrapper.ndims(), true);
    CHECK(memory_desc_init_by_tag(src_reshaped, 2, src_squished_dims,
            src_wrapper.data_type(), src_format_tag));
    CHECK(memory_desc_init_by_tag(weights_reshaped, 2, wei_squished_dims,
            wei_wrapper.data_type(), wei_format_tag));
    if (with_bias()) {
        const auto bias_md = arg_md(DNNL_ARG_BIAS);
        //Reshape bias to 1 x OC;
        dims_t reshaped_bias_dims {1, bias_md->dims[0]};
        CHECK(memory_desc_init_by_tag(bias_reshaped, 2, reshaped_bias_dims,
                bias_md->data_type, format_tag::ab));
    }

    CHECK(memory_desc_init_by_tag(dst_reshaped, 2, dst_wrapper.dims(),
            dst_wrapper.data_type(), dst_format_tag));

    CHECK(gpu::generic::sycl::detail::init_matmul_pd(engine, attr(),
            &src_reshaped, &weights_reshaped, &bias_reshaped, &dst_reshaped,
            matmul_pd));

    // check the memory format. (If 1 is not the innermost stride(ab...) or the second stride (a...b))
    // reorder it to a ab... format. Only check src and weights, as dst will be handled by the matmul.
    memory_desc_t src_reordered;

    // Check if src needs reorde
    bool is_favourable_layout
            = (src_strides[ndims - 1] == 1 || src_strides[1] == 1)
            && src_strides[0] == src_reshaped.dims[1];
    bool is_not_strided = ((src_strides[0] == 1 || src_strides[1] == 1)
            && src_wrapper.ndims() == 2); // check for strided plain layouts
    is_favourable_layout = is_favourable_layout && is_not_strided;
    if (!is_favourable_layout) {
        src_needs_reorder = true;
        memory_desc_init_by_tag(src_reordered, src_wrapper.ndims(),
                src_wrapper.dims(), src_wrapper.data_type(),
                dnnl::impl::get_abx_tag(src_wrapper.ndims()));
        detail::init_reorder_pd(
                engine, arg_md(DNNL_ARG_SRC), &src_reordered, src_reorder_pd);
    }

    memory_desc_t wei_reordered;
    format_tag_t wei_reordered_tag;
    auto src_reordered_wrapper = memory_desc_wrapper(src_reordered);
    if (src_needs_reorder) {
        // if the weight layout is not compatible with with src, reorder that too.
        if (detail::get_dim_order(src_reordered_wrapper.ndims(),
                    src_reordered_wrapper.strides())
                != detail::get_dim_order(
                        wei_wrapper.ndims(), wei_wrapper.strides())) {
            wei_needs_reorder = true;
            wei_reordered_tag = dnnl::impl::get_abx_tag(wei_wrapper.ndims());
        }
    } else {
        if (detail::get_dim_order(src_wrapper.ndims(), src_wrapper.strides())
                        != detail::get_dim_order(
                                wei_wrapper.ndims(), wei_wrapper.strides())
                && wei_wrapper.ndims() > 2) {
            // This implies src is either nhwc or nchw
            wei_needs_reorder = true;
            if (src_strides[ndims - 1] == 1) {
                wei_reordered_tag
                        = dnnl::impl::get_abx_tag(wei_wrapper.ndims());
            } else if (src_strides[1] == 1) {
                wei_reordered_tag
                        = dnnl::impl::get_axb_tag(wei_wrapper.ndims());
            }
        }
    }
    if (wei_needs_reorder) {
        memory_desc_init_by_tag(wei_reordered, wei_wrapper.ndims(),
                wei_wrapper.dims(), wei_wrapper.data_type(), wei_reordered_tag);
        CHECK(detail::init_reorder_pd(engine, arg_md(DNNL_ARG_WEIGHTS),
                &wei_reordered, weights_reorder_pd));
    }

    // book scratchpad for the matmul, src_reorder and wei_reorder
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested,
            matmul_pd->scratchpad_registry());
    if (src_needs_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_src_reorder,
                src_wrapper.nelems(), src_wrapper.data_type_size());
    }
    if (wei_needs_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_weights_reorder,
                wei_wrapper.nelems(), wei_wrapper.data_type_size());
    }
    return status::success;
}

status_t ref_inner_product_bwd_data_t::pd_t::init(impl::engine_t *engine) {

    bool ok = (set_default_params() == status::success)
            && attr()->has_default_values();

    VDISPATCH_INNER_PRODUCT(ok, VERBOSE_UNSUPPORTED_TAG);

    auto src_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_DST));
    auto dst_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_SRC));
    auto wei_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_WEIGHTS));

    VDISPATCH_INNER_PRODUCT(
            utils::one_of(this->desc()->prop_kind, prop_kind::backward,
                    prop_kind::backward_data),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(
            check_bwd_data_dtypes(src_wrapper.data_type(),
                    dst_wrapper.data_type(), wei_wrapper.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_INNER_PRODUCT(
            attr()->has_default_values(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_INNER_PRODUCT(
            src_wrapper.is_plain(), "Blocked memory format is not supported");
    VDISPATCH_INNER_PRODUCT(
            dst_wrapper.is_plain(), "Blocked memory format is not supported");

    if (src_wrapper.has_zero_dim() || wei_wrapper.has_zero_dim()
            || dst_wrapper.has_zero_dim()) {
        has_zero_dim = true;
        return status::success;
    }

    // dL/dX = (dL/dY) x W (hence no transpose required here)
    auto empty_bias_desc = types::
            zero_md(); // empty memory descriptor to signify bias is not applied

    // Temporary memory descriptors to initialize matmul_pd; diff_dst will always be 2D
    memory_desc_t reshaped_diff_src_md;
    memory_desc_t reshaped_weights_md;
    memory_desc_t reshaped_diff_dst_md;
    dims_t diff_src_flattened_dims;
    dims_t wei_flattened_dims;

    // No need to swap dimensions here
    detail::get_flattened_dimension(
            dst_wrapper.dims(), diff_src_flattened_dims, dst_wrapper.ndims());
    detail::get_flattened_dimension(
            wei_wrapper.dims(), wei_flattened_dims, wei_wrapper.ndims());

    format_tag_t diff_dst_format = format_tag::ab;
    format_tag_t diff_src_format = format_tag::ab;
    format_tag_t wei_format = format_tag::ab;

    if (dst_wrapper.ndims() == 2) {
        if (dst_wrapper.strides()[1] != 1) { diff_src_format = format_tag::ba; }
        if (wei_wrapper.strides()[1] != 1) { wei_format = format_tag::ba; }
    }

    if (src_wrapper.strides()[1] != 1) { diff_dst_format = format_tag::ba; }

    CHECK(memory_desc_init_by_tag(reshaped_diff_src_md, 2,
            diff_src_flattened_dims, dst_wrapper.data_type(), diff_src_format));
    CHECK(memory_desc_init_by_tag(reshaped_weights_md, 2, wei_flattened_dims,
            wei_wrapper.data_type(), wei_format));
    CHECK(memory_desc_init_by_tag(reshaped_diff_dst_md, 2, src_wrapper.dims(),
            src_wrapper.data_type(), diff_dst_format));

    CHECK(gpu::generic::sycl::detail::init_matmul_pd(engine, attr(),
            &reshaped_diff_dst_md, &reshaped_weights_md, &empty_bias_desc,
            &reshaped_diff_src_md, matmul_pd));

    // Now check if diff_src and diff_dst need to be reordered
    memory_desc_t dst_reordered_desc;
    if (not detail::strides_in_desc_order(
                dst_wrapper.strides(), dst_wrapper.ndims())
            && dst_wrapper.ndims() > 2) {
        dst_needs_reorder = true;
        CHECK(memory_desc_init_by_tag(dst_reordered_desc, dst_wrapper.ndims(),
                dst_wrapper.dims(), dst_wrapper.data_type(),
                dnnl::impl::get_abx_tag(dst_wrapper.ndims())));
        CHECK(detail::init_reorder_pd(engine, &dst_reordered_desc,
                arg_md(DNNL_ARG_DIFF_SRC), dst_reorder_pd));
    }

    auto dst_reordered_wrapper = memory_desc_wrapper(dst_reordered_desc);
    if (dst_needs_reorder) {
        if (detail::get_dim_order(dst_reordered_wrapper.ndims(),
                    dst_reordered_wrapper.strides())
                != detail::get_dim_order(
                        wei_wrapper.ndims(), wei_wrapper.strides())) {
            wei_needs_reorder = true;
            memory_desc_t wei_reorder_desc;
            CHECK(memory_desc_init_by_tag(wei_reorder_desc, wei_wrapper.ndims(),
                    wei_wrapper.dims(), wei_wrapper.data_type(),
                    dnnl::impl::get_abx_tag(wei_wrapper.ndims())));
            CHECK(detail::init_reorder_pd(engine, arg_md(DNNL_ARG_WEIGHTS),
                    &wei_reorder_desc, wei_reorder_pd));
        }
    } else if (not detail::strides_in_desc_order(
                       wei_wrapper.strides(), wei_wrapper.ndims())
            && wei_wrapper.ndims() > 2) {
        wei_needs_reorder = true;
        memory_desc_t wei_reorder_desc;
        CHECK(memory_desc_init_by_tag(wei_reorder_desc, wei_wrapper.ndims(),
                wei_wrapper.dims(), wei_wrapper.data_type(),
                dnnl::impl::get_abx_tag(wei_wrapper.ndims())));
        CHECK(detail::init_reorder_pd(engine, arg_md(DNNL_ARG_WEIGHTS),
                &wei_reorder_desc, wei_reorder_pd));
    }

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested,
            matmul_pd->scratchpad_registry());
    if (dst_needs_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_src_reorder,
                dst_wrapper.nelems(), dst_wrapper.data_type_size());
    }
    if (wei_needs_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_weights_reorder,
                wei_wrapper.nelems(), wei_wrapper.data_type_size());
    }
    return status::success;
}

status_t ref_inner_product_bwd_weights_t::pd_t::init(impl::engine_t *engine) {

    bool ok = (set_default_params() == status::success);
    VDISPATCH_INNER_PRODUCT(ok, VERBOSE_UNSUPPORTED_TAG);

    auto bias_dt = arg_md(DNNL_ARG_DIFF_BIAS)->data_type;

    auto src_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_DST));
    auto dst_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_DIFF_WEIGHTS));
    auto wei_wrapper = memory_desc_wrapper(arg_md(DNNL_ARG_SRC));

    if (src_wrapper.has_zero_dim() || wei_wrapper.has_zero_dim()
            || dst_wrapper.has_zero_dim()) {
        has_zero_dim = true;
        return status::success;
    }

    VDISPATCH_INNER_PRODUCT(
            utils::one_of(this->desc()->prop_kind, prop_kind::backward,
                    prop_kind::backward_weights),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_INNER_PRODUCT(
            check_bwd_weights_dtypes(src_wrapper.data_type(),
                    dst_wrapper.data_type(), wei_wrapper.data_type(), bias_dt),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_INNER_PRODUCT(
            attr()->has_default_values(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_INNER_PRODUCT(
            src_wrapper.is_plain(), "blocked memory format is not supported");
    VDISPATCH_INNER_PRODUCT(
            wei_wrapper.is_plain(), "blocked memory format is not supported");
    VDISPATCH_INNER_PRODUCT(
            dst_wrapper.is_plain(), "blocked memory format is not supported");

    format_tag_t wei_format_tag = format_tag::ab;
    format_tag_t dst_format_tag = format_tag::ab;
    if (wei_wrapper.ndims() == 2) {
        if (dst_wrapper.strides()[1] != 1) { dst_format_tag = format_tag::ba; }
        if (wei_wrapper.strides()[1] != 1) { wei_format_tag = format_tag::ba; }
    }

    // Since dL/dY is transposed, default is format_tag::ba;
    format_tag_t src_format_tag = format_tag::ba;
    if (src_wrapper.strides()[1] != 1) { src_format_tag = format_tag::ab; }

    memory_desc_t reshaped_src_md;
    memory_desc_t reshaped_diff_wt_md;
    memory_desc_t reshaped_diff_dst_md;
    dims_t wei_reshaped;
    dims_t dst_reshaped;
    detail::get_flattened_dimension(
            wei_wrapper.dims(), wei_reshaped, wei_wrapper.ndims());
    detail::get_flattened_dimension(
            dst_wrapper.dims(), dst_reshaped, dst_wrapper.ndims());
    auto empty_bias_desc = types::
            zero_md(); // empty memory descriptor to signify bias is not applied
    // (dL / dW) = (dL/dY) ^ T x X;
    dims_t src_transposed_dims {src_wrapper.dims()[1], src_wrapper.dims()[0]};
    CHECK(memory_desc_init_by_tag(reshaped_src_md, 2, wei_reshaped,
            wei_wrapper.data_type(), wei_format_tag));
    CHECK(memory_desc_init_by_tag(reshaped_diff_wt_md, 2, dst_reshaped,
            dst_wrapper.data_type(), dst_format_tag));
    CHECK(memory_desc_init_by_tag(reshaped_diff_dst_md, 2, src_transposed_dims,
            src_wrapper.data_type(), src_format_tag));

    // Create matmul_pd for dL/dW
    CHECK(detail::init_matmul_pd(engine, attr(), &reshaped_diff_dst_md,
            &reshaped_src_md, &empty_bias_desc, &reshaped_diff_wt_md,
            matmul_pd));

    memory_desc_t wei_reordered_desc;
    if (wei_wrapper.ndims() > 2
            && !detail::strides_in_desc_order(
                    wei_wrapper.strides(), wei_wrapper.ndims())) {
        wei_requires_reorder = true;
        memory_desc_init_by_tag(wei_reordered_desc, wei_wrapper.ndims(),
                wei_wrapper.dims(), wei_wrapper.data_type(),
                dnnl::impl::get_abx_tag(wei_wrapper.ndims()));
        CHECK(detail::init_reorder_pd(engine, arg_md(DNNL_ARG_SRC),
                &wei_reordered_desc, wei_reorder_pd));
    }

    if (dst_wrapper.ndims() > 2
            && !detail::strides_in_desc_order(
                    dst_wrapper.strides(), dst_wrapper.ndims())) {
        dst_requires_reorder = true;
        memory_desc_t dst_reorder_desc;
        CHECK(memory_desc_init_by_tag(dst_reorder_desc, dst_wrapper.ndims(),
                dst_wrapper.dims(), dst_wrapper.data_type(),
                dnnl::impl::get_abx_tag(dst_wrapper.ndims())));
        CHECK(detail::init_reorder_pd(engine, &dst_reorder_desc,
                arg_md(DNNL_ARG_DIFF_WEIGHTS), dst_reorder_pd));
    }

    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_nested_multiple,
            matmul_pd->scratchpad_registry());

    //Create reduction_pd for dL/dB
    if (with_bias()) {
        CHECK(init_reduction_pd(
                engine, arg_md(DNNL_ARG_DIFF_DST), arg_md(DNNL_ARG_DIFF_BIAS)));
        // book scratchpad for reduction
        scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
                reduction_pd->scratchpad_registry());
    }

    if (wei_requires_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_weights_reorder,
                wei_wrapper.nelems(), wei_wrapper.data_type_size());
    }

    if (dst_requires_reorder) {
        scratchpad.book(memory_tracking::names::key_iprod_src_reorder,
                dst_wrapper.nelems(), dst_wrapper.data_type_size());
    }

    return status::success;
}

status_t ref_inner_product_fwd_t::init(impl::engine_t *engine) {
    if (pd()->has_zero_dim) { return status::success; }
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;

    if (pd()->src_needs_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->src_reorder_pd->create_primitive_nested(p, engine));
        src_reorder_primitive = p.first;
    }

    if (pd()->wei_needs_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->weights_reorder_pd->create_primitive_nested(p, engine));
        weights_reorder_primitive = p.first;
    }

    return status::success;
}

status_t ref_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim) { return status::success; }
    exec_args_t matmul_args(ctx.args());

    std::unique_ptr<memory_t, memory_deleter_t> src_scratch_mem;
    std::unique_ptr<memory_t, memory_deleter_t> wei_scratch_mem;

    if (pd()->src_needs_reorder) {
        auto zero_md = types::zero_md();
        exec_args_t src_reorder_args(ctx.args());
        src_reorder_args[DNNL_ARG_FROM] = src_reorder_args[DNNL_ARG_SRC];
        auto src_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_src_reorder);
        // An md should not be required to simply access the scratchpad storage
        safe_ptr_assign(src_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(src_reorder_scratchpad)));
        src_reorder_args[DNNL_ARG_TO]
                = memory_arg_t {src_scratch_mem.get(), false};
        matmul_args[DNNL_ARG_SRC] = memory_arg_t {src_scratch_mem.get(), true};
        exec_ctx_t src_reorder_ctx(ctx.stream(), std::move(src_reorder_args));
        CHECK(src_reorder_primitive->execute(src_reorder_ctx));
    }

    if (pd()->wei_needs_reorder) {
        auto zero_md = types::zero_md();
        exec_args_t wei_reorder_args(ctx.args());
        wei_reorder_args[DNNL_ARG_FROM] = wei_reorder_args[DNNL_ARG_WEIGHTS];
        auto wei_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_weights_reorder);
        // An md should not be required to simple access the scratchpad storage
        safe_ptr_assign(wei_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(wei_reorder_scratchpad)));
        wei_reorder_args[DNNL_ARG_TO]
                = memory_arg_t {wei_scratch_mem.get(), false};
        matmul_args[DNNL_ARG_WEIGHTS]
                = memory_arg_t {wei_scratch_mem.get(), true};
        exec_ctx_t wei_reorder_ctx(ctx.stream(), std::move(wei_reorder_args));
        CHECK(weights_reorder_primitive->execute(wei_reorder_ctx));
    }

    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested, matmul_primitive);
    exec_ctx_t matmul_ctx(ctx.stream(), std::move(matmul_args));
    matmul_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());
    return matmul_primitive->execute(matmul_ctx);
}

status_t ref_inner_product_bwd_data_t::init(impl::engine_t *engine) {
    if (pd()->has_zero_dim) { return status::success; }
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;
    if (pd()->dst_needs_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->dst_reorder_pd->create_primitive_nested(p, engine));
        dst_reorder_primitive = p.first;
    }

    if (pd()->wei_needs_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->wei_reorder_pd->create_primitive_nested(p, engine));
        wei_reorder_primitive = p.first;
    }
    return status::success;
}

status_t ref_inner_product_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim) { return status::success; }
    std::unique_ptr<memory_t, memory_deleter_t> dst_scratch_mem;
    std::unique_ptr<memory_t, memory_deleter_t> wei_scratch_mem;
    exec_args_t matmul_args(ctx.args());
    matmul_args[DNNL_ARG_SRC] = matmul_args[DNNL_ARG_DIFF_DST];
    matmul_args[DNNL_ARG_DST] = matmul_args[DNNL_ARG_DIFF_SRC];
    exec_args_t dst_reorder_args(ctx.args());

    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested, matmul_primitive);

    // Map src and dst to diff_dst and diff_src respectively
    if (pd()->wei_needs_reorder) {
        auto zero_md = types::zero_md();
        exec_args_t wei_reorder_args(ctx.args());
        wei_reorder_args[DNNL_ARG_FROM] = wei_reorder_args[DNNL_ARG_WEIGHTS];
        auto wei_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_weights_reorder);
        // An md should not be required to simple access the scratchpad storage
        safe_ptr_assign(wei_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(wei_reorder_scratchpad)));
        wei_reorder_args[DNNL_ARG_TO]
                = memory_arg_t {wei_scratch_mem.get(), false};
        matmul_args[DNNL_ARG_WEIGHTS]
                = memory_arg_t {wei_scratch_mem.get(), true};
        exec_ctx_t wei_reorder_ctx(ctx.stream(), std::move(wei_reorder_args));
        CHECK(wei_reorder_primitive->execute(wei_reorder_ctx));
    }

    if (pd()->dst_needs_reorder) {
        auto zero_md = types::zero_md();
        auto dst_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_src_reorder);
        // An md should not be required to simple access the scratchpad storage
        safe_ptr_assign(dst_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(dst_reorder_scratchpad)));
        dst_reorder_args[DNNL_ARG_TO] = matmul_args[DNNL_ARG_DST];
        dst_reorder_args[DNNL_ARG_FROM]
                = memory_arg_t {dst_scratch_mem.get(), true};
        matmul_args[DNNL_ARG_DST] = memory_arg_t {dst_scratch_mem.get(), false};
        ;
    }

    exec_ctx_t matmul_ctx(ctx.stream(), std::move(matmul_args));

    matmul_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());

    CHECK(matmul_primitive->execute(matmul_ctx));
    if (pd()->dst_needs_reorder) {
        exec_ctx_t dst_reorder_ctx(ctx.stream(), std::move(dst_reorder_args));
        CHECK(dst_reorder_primitive->execute(dst_reorder_ctx));
    }
    return status::success;
}

status_t ref_inner_product_bwd_weights_t::init(impl::engine_t *engine) {
    if (pd()->has_zero_dim) { return status::success; }
    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
    CHECK(pd()->matmul_pd->create_primitive_nested(p, engine));
    matmul_primitive = p.first;

    if (pd()->with_bias()) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t>
                p_reduction;
        CHECK(pd()->reduction_pd->create_primitive_nested(p_reduction, engine));
        reduction_primitive = p_reduction.first;
    }

    if (pd()->wei_requires_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->wei_reorder_pd->create_primitive_nested(p, engine));
        wei_reorder_primitive = p.first;
    }

    if (pd()->dst_requires_reorder) {
        std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> p;
        CHECK(pd()->dst_reorder_pd->create_primitive_nested(p, engine));
        dst_reorder_primitve = p.first;
    }

    return status::success;
}

status_t ref_inner_product_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim) { return status::success; }
    std::unique_ptr<memory_t, memory_deleter_t> dst_scratch_mem;
    std::unique_ptr<memory_t, memory_deleter_t> wei_scratch_mem;
    auto zero_md = types::zero_md();

    nested_scratchpad_t nested_scratchpad(
            ctx, memory_tracking::names::key_nested_multiple, matmul_primitive);

    exec_args_t matmul_args(ctx.args());

    auto src_memory_arg = matmul_args[DNNL_ARG_SRC];
    matmul_args[DNNL_ARG_SRC] = matmul_args[DNNL_ARG_DIFF_DST];
    matmul_args[DNNL_ARG_WEIGHTS] = src_memory_arg;
    matmul_args[DNNL_ARG_DST] = matmul_args[DNNL_ARG_DIFF_WEIGHTS];

    if (pd()->wei_requires_reorder) {
        exec_args_t wei_reorder_args(ctx.args());
        wei_reorder_args[DNNL_ARG_FROM] = wei_reorder_args[DNNL_ARG_SRC];
        auto wei_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_weights_reorder);
        // An md should not be required to simple access the scratchpad storage
        safe_ptr_assign(wei_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(wei_reorder_scratchpad)));
        wei_reorder_args[DNNL_ARG_TO]
                = memory_arg_t {wei_scratch_mem.get(), false};
        matmul_args[DNNL_ARG_WEIGHTS]
                = memory_arg_t {wei_scratch_mem.get(), true};

        exec_ctx_t wei_reorder_ctx(ctx.stream(), std::move(wei_reorder_args));
        CHECK(wei_reorder_primitive->execute(wei_reorder_ctx));
    }

    exec_args_t dst_reorder_args(ctx.args());
    if (pd()->dst_requires_reorder) {
        auto zero_md = types::zero_md();
        auto dst_reorder_scratchpad
                = ctx.get_scratchpad_grantor().get_memory_storage(
                        memory_tracking::names::key_iprod_src_reorder);
        // An md should not be required to simple access the scratchpad storage
        safe_ptr_assign(dst_scratch_mem,
                new memory_t(ctx.stream()->engine(), &zero_md,
                        std::move(dst_reorder_scratchpad)));
        dst_reorder_args[DNNL_ARG_TO] = matmul_args[DNNL_ARG_DST];
        dst_reorder_args[DNNL_ARG_FROM]
                = memory_arg_t {dst_scratch_mem.get(), true};
        matmul_args[DNNL_ARG_DST] = memory_arg_t {dst_scratch_mem.get(), false};
    }

    // Map src and dst to diff_dst and diff_src respectively
    exec_ctx_t matmul_ctx(ctx.stream(), std::move(matmul_args));

    matmul_ctx.set_scratchpad_grantor(nested_scratchpad.grantor());
    // calcules dL/dW;
    CHECK(matmul_primitive->execute(matmul_ctx));

    if (pd()->dst_requires_reorder) {
        exec_ctx_t dst_reorder_ctx(ctx.stream(), std::move(dst_reorder_args));
        CHECK(dst_reorder_primitve->execute(dst_reorder_ctx));
    }

    if (pd()->with_bias()) {
        //calculates dL/dB
        nested_scratchpad_t reduction_scratchpad(ctx,
                memory_tracking::names::key_nested_multiple + 1,
                reduction_primitive);
        exec_args_t args_copy_reduction(ctx.args());
        args_copy_reduction[DNNL_ARG_SRC]
                = args_copy_reduction[DNNL_ARG_DIFF_DST];
        args_copy_reduction[DNNL_ARG_DST]
                = args_copy_reduction[DNNL_ARG_DIFF_BIAS];
        exec_ctx_t copied_ctx_reduction(
                ctx.stream(), std::move(args_copy_reduction));

        copied_ctx_reduction.set_scratchpad_grantor(
                reduction_scratchpad.grantor());
        CHECK(reduction_primitive->execute(copied_ctx_reduction));
    }
    return status::success;
}

} // namespace dnnl::impl::gpu::generic::sycl
