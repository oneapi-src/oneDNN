/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include "graph/interface/allocator.hpp"
#include "graph/interface/backend.hpp"
#include "graph/interface/shape_infer.hpp"

#include "graph/utils/utils.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/dnnl_backend.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
const size_t DNNL_CPU_MEMALIGNMENT = 64;
#endif

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
const size_t DNNL_SYCL_MEMALIGNMENT = 64;
#endif

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

void *dnnl_allocator_t::malloc(size_t size, const dnnl::engine &p_engine,
        const graph::allocator_t *alc, allocator_t::mem_type_t type) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        return alc->allocate(size, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine),
                {type, DNNL_SYCL_MEMALIGNMENT});
#else
        return alc->allocate(size, {type, DNNL_CPU_MEMALIGNMENT});
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
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
        void *p, const dnnl::engine &p_engine, const allocator_t *alc) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        assert(!"use event based free");
#else
        return alc->deallocate(p);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        assert(!"use event based free");
#endif
    }
}

#ifdef DNNL_WITH_SYCL
void dnnl_allocator_t::free(void *p, const dnnl::engine &p_engine,
        const allocator_t *alc, const ::sycl::event &deps) {
    if (p_engine.get_kind() == dnnl::engine::kind::cpu) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, dnnl::sycl_interop::get_device(p_engine),
                dnnl::sycl_interop::get_context(p_engine), deps);
#else
        alc->deallocate(p);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
        alc->deallocate(p, dnnl::sycl_interop::get_device(p_engine),
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
    if (!dilates.empty() && !graph::utils::any_le(dilates, static_cast<dim>(0)))
        return utils::fmap(dilates, [](dim x) { return x - 1; });
    return dims(input_size - 2, 0);
}

dims group_dims(const dims &adims, dim groups) {
    auto new_dims = adims;
    new_dims.insert(new_dims.begin(), groups);
    new_dims[1] /= groups;
    return new_dims;
}

dnnl::engine make_dnnl_engine(const engine_t &g_engine) {
    dnnl::engine engine;
    engine.reset(const_cast<engine_t *>(&g_engine), true); // not own
    return engine;
}

dnnl::stream make_dnnl_stream(
        const dnnl::engine &p_engine, const stream_t &g_stream) {
    UNUSED(p_engine);
    dnnl::stream strm;
    strm.reset(const_cast<stream_t *>(&g_stream), true); // not own
    return strm;
}

dnnl::memory::desc make_dnnl_memory_desc(const logical_tensor_t &lt) {
    const logical_tensor_wrapper_t ltw(lt);
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
        return graph::utils::any_cast<memory::desc>(td.value());
    } else if (ltw.is_any()) {
        if (ltw.ndims() > 0) {
            return {ltw.vdims(), dtype, dnnl::memory::format_tag::any};
        } else if (ltw.ndims() == 0) {
            // we convert the scalar to a 1d memory
            return {dims {1}, dtype, dnnl::memory::format_tag::any};
        } else {
            // not an error, since the scratchpad output logical tensor will be
            // empty and with any layout type before layout propagation.
            return {dims {}, dtype, dnnl::memory::format_tag::any};
        }
    } else if (ltw.is_strided()) {
        if (ltw.ndims() > 0) {
            return {ltw.vdims(), dtype, ltw.vstrides()};
        } else if (ltw.ndims() == 0) {
            // we convert the scalar to a 1d memory
            return {dims {1}, dtype, dims {1}};
        } else {
            assertm(false,
                    "An empty strided logical tensor can't be convert to "
                    "memory desc");
            return {dims {}, dtype, dims {}};
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
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
        return dnnl::sycl_interop::make_memory(
                md, p_engine, dnnl::sycl_interop::memory_kind::usm, handle);
#else
        return dnnl::memory(md, p_engine, handle);
#endif
    } else if (p_engine.get_kind() == dnnl::engine::kind::gpu) {
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
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
        const tensor_t &atensor, const dnnl::engine &p_engine) {
    dnnl::memory::desc md = make_dnnl_memory_desc(atensor.get_logical_tensor());
    return make_dnnl_memory(md, p_engine, atensor.get_data_handle());
}

// fill 1 in the front of adesc, to make its ndims to be same as tgt_ndims
memory::desc expand(const memory::desc &adesc, int tgt_ndims) {
    int64_t org_ndims = adesc.get_ndims();
    dnnl::memory::dims expanded_dims = adesc.get_dims();
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
    std::vector<int> axes(static_cast<std::size_t>(adesc.get_ndims()));
    std::iota(axes.begin(), axes.end(), 0);
    axes[static_cast<std::size_t>(dim0)] = dim1;
    axes[static_cast<std::size_t>(dim1)] = dim0;
    return adesc.permute_axes(axes);
}

memory::desc to_grouped(const memory::desc &adesc, dim groups) {
    auto grouped_shape = group_dims(adesc.get_dims(), groups);
    return adesc.reshape(grouped_shape);
}

memory::desc from_grouped(const memory::desc &adesc) {
    auto new_dims = adesc.get_dims();
    const dim groups = new_dims.front();
    new_dims.erase(new_dims.begin());
    new_dims[0] *= groups;

    return adesc.reshape(new_dims, true);
}

memory::desc to_format_any(const memory::desc &adesc) {
    return memory::desc(
            adesc.get_dims(), adesc.get_data_type(), memory::format_tag::any);
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

    const auto ori_dims = adesc.get_dims();

    dims strides = get_nxc_strides(ori_dims);
    return {ori_dims, adesc.get_data_type(), strides};
}

bool is_format(const memory::desc &adesc, memory::format_tag tag) {
    return adesc == memory::desc(adesc.get_dims(), adesc.get_data_type(), tag);
}

// check if format is ncx or nxc
bool is_format(const memory::desc &adesc, const std::string &tag) {
    if (!graph::utils::one_of(tag, "ncx", "nxc")) {
        assertm(false, "wrong tag to check memory format");
        return false;
    }

    if (adesc.get_format_kind() != format_kind::blocked
            || adesc.get_inner_nblks() != 0)
        return false;

    const auto &strides = adesc.get_strides();
    const auto &shape = adesc.get_dims();
    if ("ncx" == tag) { return strides == get_ncx_strides(shape); }

    return strides == get_nxc_strides(shape);
}

bool is_4c_blocked(const memory::desc &adesc) {
    if (adesc.get_format_kind() != format_kind::blocked) return false;

    return adesc.get_inner_nblks() == 1 && adesc.get_inner_idxs()[0] == 1
            && adesc.get_inner_blks()[0] == 4;
}

bool is_plain(const memory::desc &adesc) {
    if (adesc.get_format_kind() != format_kind::blocked) return false;
    return adesc.get_inner_nblks() == 0;
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

memory::desc to_ncx_format(const memory::desc &adesc) {
    return memory::desc(adesc.get_dims(), adesc.get_data_type(),
            get_ncx_format(adesc.get_ndims()));
}

inline bool maybe_reorder_value(const value_t *val) {
    for (const auto &consumer : val->get_consumers()) {
        if (consumer.get_op().get_kind() == graph::op_kind::Reorder) {
            return true;
        }
    }
    bool is_out_value = val->has_producer()
            && val->get_producer().get_kind() == graph::op_kind::Reorder;
    return is_out_value;
}

void set_all_layout_to_any(std::vector<std::shared_ptr<op_t>> &subgraph) {
    for (auto &cur_op : subgraph) {
        for (const auto &val : cur_op->get_input_values()) {
            if (maybe_reorder_value(val.get())) continue;
            val->set_layout_type(layout_type::any);
        }

        for (const auto &val : cur_op->get_output_values()) {
            if (maybe_reorder_value(val.get())) continue;
            val->set_layout_type(layout_type::any);
        }
    }
}

status_t fill_layout_info(logical_tensor_t *lt, const memory::desc &md) {
    const logical_tensor_wrapper_t ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        const int lt_ndims = ltw.ndims();
        const int md_ndims = md.get_ndims();

        if (md_ndims == 0) {
            if (lt_ndims < 0) {
                lt->layout_type = layout_type::undef;
                return status::success;
            } else {
                assertm(false,
                        "The logical tensor should be also empty when the "
                        "memory desc is empty");
                return status::invalid_arguments;
            }
        }

        if (lt_ndims < 0 && md_ndims > 0) { // some scratchpads mem
            lt->ndims = md_ndims;
            const auto &dims = md.get_dims();
            std::copy(dims.data(), dims.data() + md_ndims, lt->dims);
            lt->data_type = static_cast<data_type_t>(md.get_data_type());
        }

        if (lt_ndims == 0 && graph::utils::prod(md.get_dims()) == 1) { // scalar
            lt->layout_type = layout_type::strided;
        }

        if (lt->id != std::numeric_limits<size_t>::max() && is_plain(md)) {
            lt->layout_type = layout_type::strided;
            graph::utils::array_copy(lt->layout.strides,
                    md.get_strides().data(), md.get_ndims());
        } else {
            graph::utils::optional_t<size_t> layout_id
                    = dnnl_backend::get_singleton().set_mem_desc(md);
            lt->layout.layout_id = layout_id.value();
            lt->layout_type = layout_type::opaque;
        }
    }
    return status::success;
}

status_t fill_layout_info(
        const std::shared_ptr<value_t> &val, const memory::desc &md) {
    logical_tensor_t lt = val->get_logical_tensor();
    const logical_tensor_wrapper_t ltw(lt);
    if (ltw.is_any()) { // we only reset any format
        const int lt_ndims = ltw.ndims();
        const int md_ndims = md.get_ndims();

        if (md_ndims == 0) {
            if (lt_ndims < 0) {
                val->set_layout_type(layout_type::undef);
                return status::success;
            } else {
                assertm(false,
                        "The logical tensor should be also empty when the "
                        "memory desc is empty");
                return status::invalid_arguments;
            }
        }

        if (lt_ndims < 0 && md_ndims > 0) { // some scratchpads mem
            val->set_dims(md.get_dims());
            val->set_data_type(
                    static_cast<impl::data_type_t>(md.get_data_type()));
        }

        if (lt_ndims == 0 && graph::utils::prod(md.get_dims()) == 1) { // scalar
            val->set_strides({});
        }

        if (ltw.id() != std::numeric_limits<size_t>::max() && is_plain(md)) {
            val->set_strides(md.get_strides());
        } else {
            val->set_layout_id(
                    dnnl_backend::get_singleton().set_mem_desc(md).value());
        }
    }
    return status::success;
}

std::shared_ptr<value_t> insert_empty_scratchpad(std::shared_ptr<op_t> &op) {
    logical_tensor_t lt = empty_logical_tensor_with_default_id();
    auto scratchpad_val = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(scratchpad_val);
    scratchpad_val->set_data_type(graph::data_type::u8);
    return scratchpad_val;
}

std::shared_ptr<value_t> insert_empty_workspace(std::shared_ptr<op_t> &op) {
    logical_tensor_t lt = empty_logical_tensor_with_default_id();
    auto workspace_val = std::make_shared<value_t>(*op, op->num_outputs(), lt);
    op->add_output(workspace_val);
    return workspace_val;
}

// This function refers to the md2fmt_tag_str in src/common/verbose.cpp, which
// is used to recover a format tag string from a memory descriptor
std::string get_format_tag_str(const dnnl::memory::desc &md) {
    assertm(md.get_format_kind() == format_kind::blocked,
            "can get format tag only for blocked format kind");

    int ndims = md.get_ndims();
    const auto &inner_blks = md.get_inner_blks();
    const auto &inner_idxs = md.get_inner_idxs();
    const int inner_nblks = md.get_inner_nblks();

    dnnl_dims_t blocks = {0};
    std::fill(blocks, blocks + ndims, 1);
    for (int iblk = 0; iblk < inner_nblks; ++iblk)
        blocks[inner_idxs[iblk]] *= inner_blks[iblk];

    char dim_chars[DNNL_MAX_NDIMS + 1] = {'\0'};

    dims_t ou_blocks = {0};
    const auto &padded_dims = md.get_padded_dims();
    std::copy(padded_dims.begin(), padded_dims.end(), ou_blocks);

    bool plain = true;
    for (int d = 0; d < ndims; ++d) {
        dim_chars[d] = static_cast<char>((blocks[d] == 1 ? 'a' : 'A') + d);
        if (blocks[d] != 1) plain = false;
        ou_blocks[d] /= blocks[d];
    }

    dnnl_dims_t strides = {0};
    const auto &strs = md.get_strides();
    std::copy(strs.begin(), strs.end(), strides);

    utils::simultaneous_sort(strides, ou_blocks, dim_chars, ndims,
            [](dim_t a, dim_t b) { return b - a; });

    std::string blk_tag = std::string(dim_chars);

    if (!plain) {
        for (int iblk = 0; iblk < inner_nblks; ++iblk) {
            blk_tag += std::to_string(inner_blks[iblk])
                    + static_cast<char>('a' + inner_idxs[iblk]);
        }
    }

    return blk_tag;
}

dnnl::memory::format_tag get_format_tag(const dnnl::memory::desc &md) {
    std::string blk_tag = get_format_tag_str(md);

    dnnl::memory::format_tag format_tag = dnnl::memory::format_tag::undef;
    for (size_t tag = 0; tag < dnnl_format_tag_last; ++tag) {
        if (dnnl_fmt_tag2str((dnnl_format_tag_t)tag) == blk_tag) {
            format_tag = static_cast<dnnl::memory::format_tag>(tag);
            break;
        }
    }
    return format_tag;
}

size_t generate_constant_cache_key(
        size_t part_id, const std::vector<dnnl::memory::desc> &const_mds) {
    size_t key = 0;
    key = hash_combine(key, part_id);
    for (auto &md : const_mds) {
        auto md_hash = impl::primitive_hashing::get_md_hash(*md.get());
        key = hash_combine(key, md_hash);
    }
    return key;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
