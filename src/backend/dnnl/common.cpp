/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "interface/allocator.hpp"
#include "interface/backend.hpp"
#include "interface/shape_infer.hpp"

#include "utils/utils.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"

#ifndef DNNL_GRAPH_CPU_SYCL
const size_t DNNL_CPU_MEMALIGNMENT = 64;
#endif

#ifdef DNNL_GRAPH_WITH_SYCL
#include "dnnl_sycl.hpp"
const size_t DNNL_SYCL_MEMALIGNMENT = 64;
#endif

#if DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
#include "dnnl_threadpool.hpp"
#endif

namespace dnnl {
namespace graph {
namespace impl {
namespace dnnl_impl {

void *dnnl_allocator_t::malloc(size_t size, const dnnl::engine &p_engine,
        const impl::allocator_t *alc, allocator_t::mem_type_t type) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        return alc->allocate(size, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine),
                {type, DNNL_SYCL_MEMALIGNMENT});
#else
        return alc->allocate(size, {type, DNNL_CPU_MEMALIGNMENT});
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        return alc->allocate(size, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine),
                {type, DNNL_SYCL_MEMALIGNMENT});
#else
        return nullptr;
#endif
    } else {
        return nullptr;
    }
}

void dnnl_allocator_t::free(
        void *p, const dnnl::engine &p_engine, const impl::allocator_t *alc) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        assert(!"use event based free");
#else
        return alc->deallocate(p);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        assert(!"use event based free");
#endif
    }
}

#ifdef DNNL_GRAPH_WITH_SYCL
void dnnl_allocator_t::free(void *p, const dnnl::engine &p_engine,
        const impl::allocator_t *alc, const ::sycl::event &deps) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        return alc->deallocate(p, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine), deps);
#else
        return alc->deallocate(p);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        return alc->deallocate(p, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine), deps);
#endif
    }
}
#endif

format_tag get_ncx_format(size_t ndim) {
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

format_tag get_ncx_format(const dims &adims) {
    const auto size = adims.size();
    return get_ncx_format(size);
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

dnnl::engine make_dnnl_engine(const impl::engine_t &g_engine) {
    if (g_engine.kind() == impl::engine_kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        return dnnl::sycl_interop::make_engine(
                g_engine.sycl_device(), g_engine.sycl_context());
#else
        return dnnl::engine(static_cast<dnnl::engine::kind>(g_engine.kind()),
                g_engine.index());
#endif
    } else if (g_engine.kind() == impl::engine_kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        return dnnl::sycl_interop::make_engine(
                g_engine.sycl_device(), g_engine.sycl_context());
#else
        return dnnl::engine(static_cast<dnnl::engine::kind>(g_engine.kind()),
                g_engine.index());
#endif
    } else {
        assert(!"only cpu and gpu engine are valid");
        return {};
    }
}

dnnl::stream make_dnnl_stream(
        const dnnl::engine &p_engine, const impl::stream_t &g_stream) {
    UNUSED(g_stream);
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        return dnnl::sycl_interop::make_stream(
                p_engine, const_cast<::sycl::queue &>(g_stream.get_queue()));
#elif DNNL_GRAPH_CPU_RUNTIME == DNNL_GRAPH_RUNTIME_THREADPOOL
        dnnl::graph::threadpool_interop::threadpool_iface *tp = nullptr;
        g_stream.get_threadpool(&tp);
        return dnnl::threadpool_interop::make_stream(p_engine,
                reinterpret_cast<dnnl::threadpool_interop::threadpool_iface *>(
                        tp));
#else
        return dnnl::stream(p_engine);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        return dnnl::sycl_interop::make_stream(
                p_engine, const_cast<::sycl::queue &>(g_stream.get_queue()));
#else
        return dnnl::stream(p_engine);
#endif
    } else {
        assert(!"only cpu and gpu stream are valid");
        return {};
    }
}

dnnl::memory::desc make_dnnl_memory_desc(const impl::logical_tensor_t &lt) {
    const impl::logical_tensor_wrapper_t ltw(lt);
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
        return impl::utils::any_cast<memory::desc>(td.value());
    } else if (ltw.is_any()) {
        if (ltw.ndims() > 0) {
            return {ltw.vdims(), dtype, dnnl::memory::format_tag::any};
        } else if (ltw.ndims() == 0) {
            // we convert the scalar to a 1d memory
            return {impl::dims {1}, dtype, dnnl::memory::format_tag::any};
        } else {
            // not an error, since the scratchpad output logical tensor will be
            // empty and with any layout type before layout propagation.
            return {impl::dims {}, dtype, dnnl::memory::format_tag::any};
        }
    } else if (ltw.is_strided()) {
        if (ltw.ndims() > 0) {
            return {ltw.vdims(), dtype, ltw.vstrides()};
        } else if (ltw.ndims() == 0) {
            // we convert the scalar to a 1d memory
            return {impl::dims {1}, dtype, impl::dims {1}};
        } else {
            assertm(false,
                    "An empty strided logical tensor can't be convert to "
                    "memory desc");
            return {impl::dims {}, dtype, impl::dims {}};
        }
    } else {
        // not an error, since the scratchpad output logical tensor will be
        // empty and with undef layout type after layout propagation if the op
        // doesn't need scratchpad memory.
        return {};
    }
}

dnnl::memory make_dnnl_memory(const dnnl::memory::desc &md,
        const dnnl::engine &p_engine, void *handle) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#ifdef DNNL_GRAPH_CPU_SYCL
        return dnnl::sycl_interop::make_memory(
                md, p_engine, dnnl::sycl_interop::memory_kind::usm, handle);
#else
        return dnnl::memory(md, p_engine, handle);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#ifdef DNNL_GRAPH_GPU_SYCL
        return dnnl::sycl_interop::make_memory(
                md, p_engine, dnnl::sycl_interop::memory_kind::usm, handle);
#else
        return dnnl::memory(md, p_engine, handle);
#endif
    } else {
        assert(!"only cpu and gpu memory are valid");
        return {};
    }
}

dnnl::memory make_dnnl_memory(
        const impl::tensor_t &atensor, const dnnl::engine &p_engine) {
    dnnl::memory::desc md = make_dnnl_memory_desc(atensor.get_logical_tensor());
    return make_dnnl_memory(md, p_engine, atensor.get_data_handle());
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

memory::desc transpose(const memory::desc &adesc, dim dim0, dim dim1) {
    std::vector<int> axes(static_cast<std::size_t>(adesc.dims().size()));
    std::iota(axes.begin(), axes.end(), 0);
    axes[static_cast<std::size_t>(dim0)] = dim1;
    axes[static_cast<std::size_t>(dim1)] = dim0;
    return adesc.permute_axes(axes);
}

memory::desc to_grouped(const memory::desc &adesc, dim groups) {
    auto grouped_shape = group_dims(adesc.dims(), groups);
    return adesc.reshape(grouped_shape);
}

memory::desc from_grouped(const memory::desc &adesc) {
    auto new_dims = adesc.dims();
    const dim groups = new_dims.front();
    new_dims.erase(new_dims.begin());
    new_dims[0] *= groups;

    return adesc.reshape(new_dims, true);
}

memory::desc to_format_any(const memory::desc &adesc) {
    return memory::desc(
            adesc.dims(), adesc.data_type(), memory::format_tag::any);
}

dims get_ncx_strides(const dims &shape) {
    auto _shape = shape;
    // replace 0 in shape to 1 when computing the strides
    for (size_t i = 0; i < _shape.size(); i++) {
        if (_shape[i] == 0) _shape[i] = 1;
    }
    dims strides(_shape.size());
    for (auto it = _shape.begin(); it < _shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), _shape.end(), 1, std::multiplies<dim_t>());
        const auto dist = std::distance(_shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
    }
    return strides;
}

dims get_nxc_strides(const dims &shape) {
    auto _shape = shape;
    // replace 0 in shape to 1 when computing the strides
    for (size_t i = 0; i < _shape.size(); i++) {
        if (_shape[i] == 0) _shape[i] = 1;
    }
    dims strides(_shape.size());
    dim tmp, tmp1, tmp2;
    switch (_shape.size()) {
        case 3:
            strides[0] = _shape[1] * _shape[2];
            strides[1] = 1;
            strides[2] = _shape[1];
            break;
        case 4:
            tmp = _shape[1] * _shape[3];
            strides[0] = tmp * _shape[2];
            strides[1] = 1;
            strides[2] = tmp;
            strides[3] = _shape[1];
            break;
        case 5:
            tmp1 = _shape[1] * _shape[4];
            tmp2 = tmp1 * _shape[3];
            strides[0] = tmp2 * _shape[2];
            strides[1] = 1;
            strides[2] = tmp2;
            strides[3] = tmp1;
            strides[4] = _shape[1];
            break;
        case 6:
            tmp1 = _shape[1] * _shape[5];
            tmp2 = tmp1 * _shape[3] * _shape[4];
            strides[0] = tmp2 * _shape[2];
            strides[1] = 1;
            strides[2] = tmp2;
            strides[3] = tmp1 * _shape[4];
            strides[4] = tmp1;
            strides[5] = _shape[1];
            break;
        default: strides = get_ncx_strides(_shape);
    }
    return strides;
}

memory::desc to_nxc_format(const memory::desc &adesc) {
    if (is_format(adesc, "nxc")) return adesc;

    const auto ndims = adesc.data.ndims;
    const dims shape {adesc.data.dims, adesc.data.dims + ndims};

    dims strides = get_nxc_strides(shape);
    return {adesc.dims(), adesc.data_type(), strides};
}

bool is_format(const memory::desc &adesc, memory::format_tag tag) {
    return adesc == memory::desc(adesc.dims(), adesc.data_type(), tag);
}

// check if format is ncx or nxc
bool is_format(const memory::desc &adesc, const std::string &tag) {
    if (!impl::utils::one_of(tag, "ncx", "nxc")) {
        assertm(false, "wrong tag to check memory format");
        return false;
    }

    if (adesc.data.format_kind != dnnl_blocked
            || adesc.data.format_desc.blocking.inner_nblks != 0)
        return false;

    auto ndims = adesc.dims().size();
    const auto &strides = adesc.data.format_desc.blocking.strides;
    const auto &shape = adesc.dims();
    std::vector<dim> stride_v {strides, strides + ndims};
    if ("ncx" == tag) { return stride_v == get_ncx_strides(shape); }

    return stride_v == get_nxc_strides(shape);
}

bool is_4c_blocked(const memory::desc &adesc) {
    if (adesc.data.format_kind != dnnl_blocked) return false;

    const auto &blk = adesc.data.format_desc.blocking;
    return blk.inner_nblks == 1 && blk.inner_idxs[0] == 1
            && blk.inner_blks[0] == 4;
}

memory::desc to_ncx_format(const memory::desc &adesc) {
    return memory::desc(
            adesc.dims(), adesc.data_type(), get_ncx_format(adesc.data.ndims));
}

void set_all_layout_to_any(std::vector<std::shared_ptr<impl::op_t>> &subgraph) {
    for (auto &cur_op : subgraph) {
        for (const auto &val : cur_op->get_input_values()) {
            val->set_layout_type(impl::layout_type::any);
        }

        for (const auto &val : cur_op->get_output_values()) {
            val->set_layout_type(impl::layout_type::any);
        }
    }
}

impl::status_t fill_layout_info(
        impl::logical_tensor_t *lt, const memory::desc &md) {
    const impl::logical_tensor_wrapper_t ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        const int lt_ndims = ltw.ndims();
        const int md_ndims = md.data.ndims;

        if (md_ndims == 0) {
            if (lt_ndims < 0) {
                lt->layout_type = impl::layout_type::undef;
                return impl::status::success;
            } else {
                assertm(false,
                        "The logical tensor should be also empty when the "
                        "memory desc is empty");
                return impl::status::invalid_arguments;
            }
        }

        if (lt_ndims < 0 && md_ndims > 0) { // some scratchpads mem
            lt->ndims = md_ndims;
            const dnnl_dims_t &dims = md.data.dims;
            std::copy(dims, dims + md_ndims, lt->dims);
            lt->data_type = static_cast<impl::data_type_t>(md.data_type());
        }

        if (lt_ndims == 0 && impl::utils::prod(md.dims()) == 1) { // scalar
            lt->layout_type = impl::layout_type::strided;
        }

        if (lt->id != std::numeric_limits<size_t>::max()
                && (is_format(md, "ncx") || is_format(md, "nxc"))) {
            lt->layout_type = impl::layout_type::strided;
            impl::utils::array_copy(lt->layout.strides,
                    md.data.format_desc.blocking.strides, md.data.ndims);
        } else {
            impl::utils::optional<size_t> layout_id
                    = dnnl_backend::get_singleton().set_mem_desc(md);
            lt->layout.layout_id = layout_id.value();
            lt->layout_type = impl::layout_type::opaque;
        }
    }
    return impl::status::success;
}

impl::status_t fill_layout_info(
        const std::shared_ptr<impl::value_t> &val, const memory::desc &md) {
    impl::logical_tensor_t lt = val->get_logical_tensor();
    const impl::logical_tensor_wrapper_t ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        const int lt_ndims = ltw.ndims();
        const int md_ndims = md.data.ndims;

        if (md_ndims == 0) {
            if (lt_ndims < 0) {
                val->set_layout_type(impl::layout_type::undef);
                return impl::status::success;
            } else {
                assertm(false,
                        "The logical tensor should be also empty when the "
                        "memory desc is empty");
                return impl::status::invalid_arguments;
            }
        }

        if (lt_ndims < 0 && md_ndims > 0) { // some scratchpads mem
            val->set_dims(md.dims());
            val->set_data_type(static_cast<impl::data_type_t>(md.data_type()));
        }

        if (lt_ndims == 0 && impl::utils::prod(md.dims()) == 1) { // scalar
            val->set_strides({});
        }

        if (ltw.id() != std::numeric_limits<size_t>::max()
                && (is_format(md, "ncx") || is_format(md, "nxc"))) {
            const auto &strides = md.data.format_desc.blocking.strides;
            val->set_strides({strides, strides + lt_ndims});
        } else {
            val->set_layout_id(
                    dnnl_backend::get_singleton().set_mem_desc(md).value());
        }
    }
    return impl::status::success;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
