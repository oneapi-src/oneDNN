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

#include <limits>
#include <utility>
#include <vector>

#include "interface/backend.hpp"
#include "utils/utils.hpp"

#include "common.hpp"
#include "dnnl_backend.hpp"
#include "tensor.hpp"

const size_t DNNL_CPU_MEMALIGNMENT = 4096;

#if DNNL_GRAPH_WITH_SYCL
#include "dnnl_sycl.hpp"
const size_t DNNL_SYCL_MEMALIGNMENT = 16;
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void *allocator::malloc(size_t size, const dnnl::engine &p_engine,
        const impl::allocator_t *alc) {
#if DNNL_GRAPH_WITH_SYCL
    return alc->allocate(size, dnnl::sycl_interop::get_device(p_engine),
            dnnl::sycl_interop::get_context(p_engine),
            {impl::allocator_lifetime::persistent, DNNL_SYCL_MEMALIGNMENT});
#else
    return p_engine.get_kind() == dnnl::engine::kind::cpu ? alc->allocate(size,
                   {impl::allocator_lifetime::persistent,
                           DNNL_CPU_MEMALIGNMENT})
                                                          : nullptr;
#endif
}

void allocator::free(
        void *p, const dnnl::engine &p_engine, const impl::allocator_t *alc) {
#if DNNL_GRAPH_WITH_SYCL
    return alc->deallocate(p, dnnl::sycl_interop::get_context(p_engine));
#else
    if (p_engine.get_kind() == dnnl::engine::kind::cpu)
        return alc->deallocate(p);
    else
        return;
#endif
}

format_tag get_default_format(size_t ndim) {
    switch (ndim) {
        case 1: return format_tag::a;
        case 2: return format_tag::ab;
        case 3: return format_tag::abc;
        case 4: return format_tag::abcd;
        case 5: return format_tag::abcde;
        case 6: return format_tag::abcdef;
        default: return format_tag::undef;
    }
}

format_tag get_default_format(const dims &adims) {
    const auto size = adims.size();
    return get_default_format(size);
}

dims get_compatible_dilates(const dims &dilates, size_t input_size) {
    if (!dilates.empty() && !impl::utils::any_le(dilates, static_cast<dim>(0)))
        return utils::fmap(dilates, [](dim x) { return x - 1; });
    return dims(input_size - 2, 0);
}

dims group_dims(const dims &adims, dim groups) {
    auto new_dims = adims;
    new_dims.insert(new_dims.begin(), groups);
    new_dims[1] /= groups;
    return new_dims;
}

std::pair<std::vector<float>, std::vector<float>> compute_scales(
        float src_scale, float dst_scale, std::vector<float> weight_scales) {
    auto scale_size = weight_scales.size();
    std::vector<float> bias_scales(scale_size), op_scales(scale_size);

    for (size_t i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scale * weight_scales[i];
        op_scales[i] = dst_scale / bias_scales[i];
    }
    return std::make_pair(std::move(bias_scales), std::move(op_scales));
}

std::pair<bool, int64_t> try_reverse_axis(
        const int64_t axis, const int32_t rank) {
    // oneDNN can not operate on the negative axis
    const auto new_axis = (axis < 0) ? rank + axis : axis;
    if (new_axis < 0 || new_axis >= static_cast<int64_t>(rank))
        return std::make_pair(false, axis);
    return std::make_pair(true, new_axis);
}

dnnl::engine make_dnnl_engine(const impl::engine_t &g_engine) {
#if DNNL_GRAPH_WITH_SYCL
    return dnnl::sycl_interop::make_engine(
            g_engine.sycl_device(), g_engine.sycl_context());
#else
    return dnnl::engine(static_cast<dnnl::engine::kind>(g_engine.kind()),
            static_cast<size_t>(g_engine.device_id()));
#endif
}

dnnl::stream make_dnnl_stream(
        const dnnl::engine &p_engine, const impl::stream_t &g_stream) {
#if DNNL_GRAPH_WITH_SYCL
    return dnnl::sycl_interop::make_stream(
            p_engine, const_cast<cl::sycl::queue &>(g_stream.get_queue()));
#else
    UNUSED(g_stream);
    return dnnl::stream(p_engine);
#endif
}

dnnl::memory::desc make_dnnl_memory_desc(const impl::logical_tensor_t &lt) {
    const impl::logical_tensor_wrapper ltw(lt);
    const auto dtype = static_cast<dnnl::memory::data_type>(ltw.data_type());

    if (ltw.is_opaque()) {
#ifdef DNNL_GRAPH_LAYOUT_DEBUG
        const auto format_tag
                = static_cast<dnnl::memory::format_tag>(ltw.layout_id());
        if (format_tag < dnnl::memory::format_tag::format_tag_last
                && format_tag > dnnl::memory::format_tag::any) {
            return {ltw.vdims(), dtype, format_tag};
        }
#endif // DNNL_GRAPH_LAYOUT_DEBUG

        const auto &td = dnnl_backend::get_singleton().get_mem_desc(
                static_cast<size_t>(ltw.layout_id()));
        return static_cast<dnnl::memory::desc>(
                impl::utils::any_cast<tensor::desc>(td.value()));
    } else if (ltw.is_any()) {
        return {ltw.vdims(), dtype, dnnl::memory::format_tag::any};
    } else if (ltw.is_strided()) {
        return {ltw.vdims(), dtype, ltw.vstrides()};
    } else {
        return {};
    }
}

dnnl::memory make_dnnl_memory(
        const impl::tensor_t &atensor, const dnnl::engine &p_engine) {
    dnnl::memory::desc md = make_dnnl_memory_desc(atensor.get_logical_tensor());
#if DNNL_GRAPH_WITH_SYCL
    return dnnl::sycl_interop::make_memory(md, p_engine,
            dnnl::sycl_interop::memory_kind::usm, atensor.get_data_handle());
#else
    return dnnl::memory(md, p_engine, atensor.get_data_handle());
#endif
}

dnnl::memory make_dnnl_memory(const dnnl::memory::desc &md,
        const dnnl::engine &p_engine, void *handle) {
#if DNNL_GRAPH_WITH_SYCL
    return dnnl::sycl_interop::make_memory(
            md, p_engine, dnnl::sycl_interop::memory_kind::usm, handle);
#else
    return dnnl::memory(md, p_engine, handle);
#endif
}

// fill 1 in the front of adesc, to make its ndims to be same as tgt_ndims
memory::desc expand(const memory::desc &adesc, int tgt_ndims) {
    int64_t org_ndims = adesc.data.ndims;
    dnnl::memory::dims expanded_dims = adesc.dims();
    expanded_dims.insert(expanded_dims.begin(), tgt_ndims - org_ndims, 1);
    return adesc.reshape(expanded_dims);
}

// transpose the right-most two dimensions
memory::desc permute_last_two_dims(const memory::desc &adesc) {
    assert(adesc.data.ndims > 1);
    int count = 0;
    std::vector<int> axes(adesc.data.ndims);
    std::generate(axes.begin(), axes.end(), [&count]() { return count++; });
    const auto last_dim = static_cast<dims::size_type>(adesc.data.ndims - 1);
    std::swap(axes[last_dim], axes[last_dim - 1]);
    return adesc.permute_axes(axes);
}

// permute the NXC format adesc to NCX format
/// \note
/// The logical axes will be permuted in the following manner:
/// for (i = 0; i < ndims(); i++)
///     new_desc.dims()[permutation[i]] = dims()[i];
/// if we want to permute nhwc to nchw, we need:
///     permutation[0] = 0
///     permutation[1] = 2
///     permutation[2] = 3
///     permutation[3] = 1
memory::desc permute_NXC2NCX(const memory::desc &adesc) {
    assert(adesc.data.ndims > 2);
    int count = 0;
    std::vector<int> axes(adesc.data.ndims);
    std::generate(axes.begin(), axes.end(), [&count]() { return count++; });
    axes.push_back(axes[1]);
    axes.erase(axes.begin() + 1);
    memory::desc ret = adesc.permute_axes(axes);
    return ret;
}

memory::desc permute_NCX2NXC(const memory::desc &adesc) {
    assert(adesc.data.ndims > 2);
    int count = 0;
    std::vector<int> axes(adesc.data.ndims);
    std::generate(axes.begin(), axes.end(), [&count]() { return count++; });
    axes.insert(axes.begin() + 1, axes.back());
    axes.pop_back();
    memory::desc ret = adesc.permute_axes(axes);
    return ret;
}

// permute the XIO format adesc to OIX format
/// \note
/// The logical axes will be permuted in the following manner:
/// for (i = 0; i < ndims(); i++)
///     new_desc.dims()[permutation[i]] = dims()[i];
/// if we want to permute hwio to oihw, we need:
///     permutation[0] = 2
///     permutation[1] = 3
///     permutation[2] = 1
///     permutation[3] = 0
memory::desc permute_XIO2OIX(const memory::desc &adesc) {
    assert(adesc.data.ndims > 2);
    int count = 0;
    std::vector<int> axes(adesc.data.ndims);
    std::generate(axes.begin(), axes.end(), [&count]() { return count++; });
    axes.push_back(axes[1]);
    axes.push_back(axes[0]);
    axes.erase(axes.begin());
    axes.erase(axes.begin());
    return adesc.permute_axes(axes);
}

// permute the OIX format adesc to XIO format
/// \note
/// The logical axes will be permuted in the following manner:
/// for (i = 0; i < ndims(); i++)
///     new_desc.dims()[permutation[i]] = dims()[i];
/// if we want to permute oihw to hwio, we need:
///     permutation[0] = 3
///     permutation[1] = 2
///     permutation[2] = 0
///     permutation[3] = 1
memory::desc permute_OIX2XIO(const memory::desc &adesc) {
    int count = 0;
    std::vector<int> axes(adesc.data.ndims);
    std::generate(axes.begin(), axes.end(), [&count]() { return count++; });
    axes.insert(axes.begin(), axes[axes.size() - 2]);
    axes.insert(axes.begin(), axes[axes.size() - 1]);
    axes.pop_back();
    axes.pop_back();
    return adesc.permute_axes(axes);
}

memory::desc to_grouped(const memory::desc &adesc, dim groups) {
    auto grouped_shape = group_dims(adesc.dims(), groups);
    return adesc.reshape(grouped_shape);
}

memory::desc to_format_any(const memory::desc &adesc) {
    return memory::desc(
            adesc.dims(), adesc.data_type(), memory::format_tag::any);
}

bool is_4c_blocked(const memory::desc &adesc) {
    if (adesc.data.format_kind != dnnl_blocked) return false;

    const auto &blk = adesc.data.format_desc.blocking;
    return blk.inner_nblks == 1 && blk.inner_idxs[0] == 1
            && blk.inner_blks[0] == 4;
}

memory::desc to_default_format(const memory::desc &adesc) {
    return memory::desc(adesc.dims(), adesc.data_type(),
            get_default_format(adesc.data.ndims));
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
