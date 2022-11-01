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

// used to permute the right-most two dimensions
std::vector<int64_t> get_last_two_dims_permutation(int ndims) {
    assert(ndims > 1);
    std::vector<int64_t> axes(ndims);
    std::iota(axes.begin(), axes.end(), 0);
    const auto last_dim = static_cast<dims::size_type>(ndims - 1);
    std::swap(axes[last_dim], axes[last_dim - 1]);
    return axes;
}

// used to permute the from_format to to_format
/// \note
/// The logical axes will be permuted in the following manner:
/// for (i = 0; i < ndims(); i++)
///     new_desc.dims()[permutation[i]] = dims()[i];
/// if we want to permute nxc to ncx:
///     from_format      to_format         permutation
/// n       0                0           permutation[0] = 0
/// x    1 ~ s_dims     2 ~ s_dims + 1   permutation[1~s_dims] = 2~s_dims+1
/// c    s_dims + 1          1           permutation[s_dims+1] = 1
std::vector<int64_t> get_permutation(int ndims, const std::string &from_format,
        const std::string &to_format) {
    assert(ndims > 2);
    assert(from_format == "NCX" || from_format == "NXC" || from_format == "IOX"
            || from_format == "OIX" || from_format == "XIO"
            || from_format == "XOI");
    assert(to_format == "NCX" || to_format == "NXC" || to_format == "IOX"
            || to_format == "OIX" || to_format == "XIO" || to_format == "XOI");

    size_t spatial_dims = static_cast<size_t>(ndims - 2);
    std::vector<int64_t> axes(ndims);
    size_t axes_idx = 0;
    for (const auto &ch : from_format) {
        size_t to_idx = to_format.find(ch);
        if (ch == 'X') {
            // spatial dims
            for (size_t spatial_idx = 0; spatial_idx < spatial_dims;
                    ++spatial_idx) {
                axes[axes_idx++] = to_idx + spatial_idx;
            }
        } else {
            if (to_idx > to_format.find('X')) {
                axes[axes_idx++] = to_idx + spatial_dims - 1;
            } else {
                axes[axes_idx++] = to_idx;
            }
        }
    }

    return axes;
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

// get the dense strides of a given shape
// eg. (3, 4, 5) -> (20, 5, 1)
dims get_dense_strides(const dims &shape) {
    dims strides(shape.size());
    for (auto it = shape.begin(); it < shape.end(); ++it) {
        const auto val = std::accumulate(
                std::next(it), shape.end(), 1, [](dim_t x, dim_t y) {
                    // replace 0 in shape to 1 when computing the strides
                    return std::max<dim_t>(x, 1) * std::max<dim_t>(y, 1);
                });
        const auto dist = std::distance(shape.begin(), it);
        strides[static_cast<size_t>(dist)] = val;
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

bool is_plain(const memory::desc &adesc) {
    if (adesc.data.format_kind != dnnl_blocked) return false;

    const auto &blk = adesc.data.format_desc.blocking;
    return blk.inner_nblks == 0;
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

        // use shape and stride to describe plain layout
        if (lt->id != std::numeric_limits<size_t>::max() && is_plain(md)) {
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

        // use shape and stride to descripe plain layout
        if (ltw.id() != std::numeric_limits<size_t>::max() && is_plain(md)) {
            const auto &strides = md.data.format_desc.blocking.strides;
            val->set_strides({strides, strides + lt_ndims});
        } else {
            val->set_layout_id(
                    dnnl_backend::get_singleton().set_mem_desc(md).value());
        }
    }
    return impl::status::success;
}

std::shared_ptr<impl::value_t> insert_empty_scratchpad(
        std::shared_ptr<impl::op_t> &op) {
    logical_tensor_t lt = impl::empty_logical_tensor_with_default_id();
    auto scratchpad_val = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(scratchpad_val);
    scratchpad_val->set_data_type(impl::data_type::u8);
    return scratchpad_val;
}

std::shared_ptr<impl::value_t> insert_empty_workspace(
        std::shared_ptr<impl::op_t> &op) {
    logical_tensor_t lt = impl::empty_logical_tensor_with_default_id();
    auto workspace_val = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(workspace_val);
    return workspace_val;
}

} // namespace dnnl_impl
} // namespace impl
} // namespace graph
} // namespace dnnl
