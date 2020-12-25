/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef LLGA_BACKEND_DNNL_TENSOR_HPP
#define LLGA_BACKEND_DNNL_TENSOR_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "attributes.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/tensor.hpp"
#include "interface/utils.hpp"
#include "utils.hpp"

namespace llga {
namespace impl {
namespace dnnl_impl {

namespace impl = llga::impl;

class tensor : public memory {
public:
    using dim_t = dnnl_dim_t;
    using dims_t = dnnl_dims_t;
    using format_kind_t = dnnl_format_kind_t;
    using blocking_desc_t = dnnl_blocking_desc_t;

    struct desc : public memory::desc {
        friend class tensor;

        // avoid conflicts with function desc::dims() and desc::data_type()
        using dim = typename memory::dim;
        using dims = typename memory::dims;
        using data_type = typename memory::data_type;
        using format_tag = typename memory::format_tag;

        desc() : memory::desc() {};

        // copy ctor
        desc(const desc &adesc) : memory::desc(adesc.data) {
            set_g(adesc.g());
        };

        // supplement group info for memory::desc
        desc(const memory::desc &adesc, dim groups = 1)
            : memory::desc(adesc.data) {
            set_g(groups);
        };

        desc &operator=(const desc &adesc) {
            memory::desc::operator=(adesc);
            set_g(adesc.g());
            return *this;
        }

        desc(const dnnl_memory_desc_t &adata) : memory::desc(adata) {};

        desc(const dims &adims, data_type adata_type, format_tag aformat_tag)
            : memory::desc(adims, adata_type, aformat_tag) {
            set_g(1);
        }

        desc(const dims &adims, data_type adata_type)
            : desc(adims, adata_type, get_default_format(adims)) {}

        desc(const dims &adims, data_type adata_type, const dims &astrides)
            : memory::desc(adims, adata_type, astrides) {
            set_g(1);
        }

        desc(const impl::logical_tensor_t &lt)
            : memory::desc(get_tensor_desc(lt)) {}

        /// public ndims
        inline int get_ndims() const {
            return is_grouped() ? data.ndims - 1 : data.ndims;
        }

        /// Return size of specified dimension
        inline dim_t get_dim(int index) const {
            if (!is_grouped()) {
                if (index < 0 || index >= data.ndims)
                    return static_cast<dim_t>(0);
                return data.dims[index];
            } else {
                if (index < 0 || index >= data.ndims - 1)
                    return static_cast<dim_t>(0);
                return index == 0 ? data.dims[0] * data.dims[1]
                                  : data.dims[index + 1];
            }
        }

        /// Returns dimension vector
        inline dims get_dims() const {
            if (!is_grouped()) {
                return dims(data.dims, data.dims + data.ndims);
            } else {
                auto ret = dims(data.dims + 1, data.dims + data.ndims);
                ret[0] *= data.dims[0]; // g == data.dims[0]
                return ret;
            }
        }

        /// Returns descriptor data type
        inline data_type get_data_type() const {
            return static_cast<data_type>(data.data_type);
        }

        inline dims get_strides() const {
            BACKEND_DNNL_ENFORCE(
                    is_plain(), "Call to_public() before get_strides()");
            const auto &strides = blocking_strides();
            if (!is_grouped()) {
                return dims(strides, strides + data.ndims);
            } else {
                auto ret = dims(strides + 1, strides + data.ndims);
                ret[0] = std::min(strides[0], strides[1]);
                return ret;
            }
        }

        inline format_tag get_format_tag() const {
            if (format_kind() == format_kind_t::dnnl_format_kind_undef)
                return format_tag::undef;
            if (format_kind() == format_kind_t::dnnl_format_kind_any)
                return format_tag::any;

            BACKEND_DNNL_ENFORCE(is_plain(),
                    "Only can get format_tag for non-opaque layout");

            format_tag ret = format_tag::undef;
            const auto &strides = blocking_strides();
            const auto &dims = data.dims;
            const int a = 0, b = 1, c = 2, d = 3, e = 4, f = 5;

            auto check = [&](const std::vector<int> &mem_order) -> bool {
                bool status = true;
                for (int i = 0; i < mem_order.size(); i++) {
                    int64_t d_i_stride = 1;
                    for (int j = i + 1; j < mem_order.size(); j++)
                        d_i_stride *= dims[mem_order[j]];
                    status = status && d_i_stride == strides[mem_order[i]];
                }
                return status;
            };

            if (data.ndims == 1 && check({a}))
                ret = format_tag::a;
            else if (data.ndims == 2 && check({a, b}))
                ret = format_tag::ab;
            else if (data.ndims == 2 && check({b, a}))
                ret = format_tag::ba;
            else if (data.ndims == 3 && check({a, b, c}))
                ret = format_tag::abc;
            else if (data.ndims == 3 && check({a, c, b}))
                ret = format_tag::acb;
            else if (data.ndims == 3 && check({b, a, c}))
                ret = format_tag::bac;
            else if (data.ndims == 3 && check({b, c, a}))
                ret = format_tag::bca;
            else if (data.ndims == 3 && check({c, b, a}))
                ret = format_tag::cba;
            else if (data.ndims == 4 && check({a, b, c, d}))
                ret = format_tag::abcd;
            else if (data.ndims == 4 && check({a, b, d, c}))
                ret = format_tag::abdc;
            else if (data.ndims == 4 && check({a, c, d, b}))
                ret = format_tag::acdb;
            else if (data.ndims == 4 && check({b, a, c, d}))
                ret = format_tag::bacd;
            else if (data.ndims == 4 && check({b, c, d, a}))
                ret = format_tag::bcda;
            else if (data.ndims == 4 && check({c, d, b, a}))
                ret = format_tag::cdba;
            else if (data.ndims == 4 && check({d, c, a, b}))
                ret = format_tag::dcab;
            else if (data.ndims == 5 && check({a, b, c, d, e}))
                ret = format_tag::abcde;
            else if (data.ndims == 5 && check({a, b, d, e, c}))
                ret = format_tag::abdec;
            else if (data.ndims == 5 && check({a, c, b, d, e}))
                ret = format_tag::acbde;
            else if (data.ndims == 5 && check({a, c, d, e, b}))
                ret = format_tag::acdeb;
            else if (data.ndims == 5 && check({b, a, c, d, e}))
                ret = format_tag::bacde;
            else if (data.ndims == 5 && check({b, c, d, e, a}))
                ret = format_tag::bcdea;
            else if (data.ndims == 5 && check({c, d, e, b, a}))
                ret = format_tag::cdeba;
            else if (data.ndims == 5 && check({d, e, c, a, b}))
                ret = format_tag::decab;
            else if (data.ndims == 6 && check({a, b, c, d, e, f}))
                ret = format_tag::abcdef;
            else if (data.ndims == 6 && check({a, c, b, d, e, f}))
                ret = format_tag::acbdef;
            else if (data.ndims == 6 && check({d, e, f, c, a, b}))
                ret = format_tag::defcab;

            return ret;
        }

        /** returns true if memory descriptor is zero */
        bool is_zero() const { return data.ndims == 0; }

        /** returns the number of elements including padding if
         * \param with_padding is true, and the number of data elements
         * otherwise
         * */
        inline dim_t nelems(bool with_padding = false) const {
            if (is_zero()) return 0;
            auto dims = with_padding ? data.padded_dims : data.dims;
            return std::accumulate(
                    dims, dims + data.ndims, 1, std::multiplies<dim_t>());
        }

        inline bool is_plain() const {
            return is_blocking_desc() && blocking_desc().inner_nblks == 0;
        }

        inline bool is_default() const {
            if (!is_plain()) return false;

            const auto &strides = blocking_strides();
            for (int i = 0; i < data.ndims - 1; i++) {
                if (strides[i] < strides[i + 1]) { return false; }
            }
            return true;
        }

        inline bool is_nhwc() const {
            if (!is_plain() || data.ndims != 4) return false;
            const auto &dims = data.dims;
            const auto &strides = blocking_strides();
            const auto n = 0, c = 1, h = 2, w = 3;
            return strides[n] == dims[h] * dims[w] * dims[c]
                    && strides[h] == dims[w] * dims[c] && strides[w] == dims[c]
                    && strides[c] == 1;
        }

        inline bool is_nchw() const {
            if (!is_plain() || data.ndims != 4) return false;
            const auto &dims = data.dims;
            const auto &strides = blocking_strides();
            const auto n = 0, c = 1, h = 2, w = 3;
            return strides[n] == dims[c] * dims[h] * dims[w]
                    && strides[c] == dims[h] * dims[w] && strides[h] == dims[w]
                    && strides[w] == 1;
        }

        inline bool is_iohw() const {
            if (!is_plain() || data.ndims != 4) return false;
            const auto &dims = data.dims;
            const auto &strides = blocking_strides();
            const auto o = 0, i = 1, h = 2, w = 3;
            return strides[i] == dims[o] * dims[h] * dims[w]
                    && strides[o] == dims[h] * dims[w] && strides[h] == dims[w]
                    && strides[w] == 1;
        }

        // workaround for issue intel/mkl-dnn#588
        bool is_4c_blocked() const {
            const auto &blk = blocking_desc();
            return blk.inner_nblks == 1 && blk.inner_idxs[0] == 1
                    && blk.inner_blks[0] == 4;
        }

        // legacy API for caffe2
        bool is_limited_blockable() const {
            const auto &blk = blocking_desc();
            // compute compatible block_dims with v0.x
            dims block_dims(data.ndims, 1);
            for (auto i = 0; i < blk.inner_nblks; i++) {
                block_dims[blk.inner_idxs[i]] *= blk.inner_blks[i];
            }
            for (auto i = 0; i < data.ndims; i++) {
                if (data.dims[i] < block_dims[i]) continue;
                if (data.dims[i] % block_dims[i] == 0) continue;
                return false;
            }
            return true;
        }

        desc to_format(format_tag aformat_tag) const {
            auto ret = desc(get_internal_dims(), get_data_type(), aformat_tag);
            ret.set_g(g());
            return ret;
        }

        desc to_format_any() const {
            auto ret = desc(
                    get_internal_dims(), get_data_type(), format_tag::any);
            ret.set_g(g());
            return ret;
        }

        desc to_default_format() const {
            auto ret = desc(get_internal_dims(), get_data_type());
            ret.set_g(g());
            return ret;
        }

        desc clone() const { return desc(*this); }

        desc to_type(data_type atype) const {
            auto ret = clone();
            ret.data.data_type = static_cast<dnnl_data_type_t>(atype);
            ret.set_g(g());
            return ret;
        }

        desc to_grouped(dim groups, bool is_deconv = false) const {
            auto grouped_dims = utils::group_dims(get_internal_dims(), groups);
            auto grouped_desc = desc(grouped_dims, get_data_type());
            grouped_desc.set_g(groups);
            return grouped_desc;
        }

        bool has_same_shape_as(const desc &that) const {
            if (data.ndims != that.data.ndims) return false;
            return std::equal(
                    data.dims, data.dims + data.ndims, that.data.dims);
        }

        // to be replaced with memory_desc_permute_axes in DNNL v1.3
        desc permute(const std::vector<int> &permute_axes = {}) const {
            if (data.ndims <= 1) { return clone(); }

            auto perms = permute_axes;
            if (perms.empty()) {
                perms.resize(data.ndims);
                std::iota(perms.rbegin(), perms.rend(), 0);
            } else {
                BACKEND_DNNL_ENFORCE(perms.size() == data.ndims,
                        "Axes should be size like source tensor.");
                auto perms_sorted = perms;
                std::sort(perms_sorted.begin(), perms_sorted.end());
                for (auto i = 0; i < perms_sorted.size(); ++i) {
                    BACKEND_DNNL_ENFORCE(perms_sorted[i] == i,
                            "Axes should be a permutation of 0 to ndim.");
                }
                if (perms_sorted == perms) { return clone(); }
            }

            desc new_desc {};
            auto ndims = data.ndims;
            new_desc.data.ndims = data.ndims;
            new_desc.data.data_type = data.data_type;
            new_desc.data.format_kind = data.format_kind;
            new_desc.data.offset0 = data.offset0;
            new_desc.set_g(g());

            // permute dims, padded_dims, padded_offsets, strides
            auto &new_dims = new_desc.data.dims;
            auto &old_dims = data.dims;
            auto &new_stride = new_desc.data.format_desc.blocking.strides;
            auto &old_stride = data.format_desc.blocking.strides;
            auto &new_paddim = new_desc.data.padded_dims;
            auto &old_paddim = data.padded_dims;
            auto &new_padoff = new_desc.data.padded_offsets;
            auto &old_padoff = data.padded_offsets;
            for (int i = 0; i < ndims; i++) {
                new_dims[i] = old_dims[perms[i]];
                new_stride[i] = old_stride[perms[i]];
                new_paddim[i] = old_paddim[perms[i]];
                new_padoff[i] = old_padoff[perms[i]];
            }

            // permute blocking
            auto inner_nblks = data.format_desc.blocking.inner_nblks;
            new_desc.data.format_desc.blocking.inner_nblks = inner_nblks;
            auto &old_inner_idxs = data.format_desc.blocking.inner_idxs;
            auto &new_inner_idxs
                    = new_desc.data.format_desc.blocking.inner_idxs;
            auto &old_inner_blks = data.format_desc.blocking.inner_blks;
            auto &new_inner_blks
                    = new_desc.data.format_desc.blocking.inner_blks;
            for (int i = 0; i < inner_nblks; i++) {
                new_inner_idxs[i] = perms[old_inner_idxs[i]];
                new_inner_blks[i] = old_inner_blks[i];
            }

            return new_desc;
        }

        desc transpose(dim dim0, dim dim1) const {
            std::vector<int> axes(data.ndims);
            std::iota(axes.begin(), axes.end(), 0);
            std::swap(axes[dim0], axes[dim1]);
            return permute(axes);
        }

        /** inits descriptor with logical dimensions adims and keep the current
        * blocking structure
        */
        desc to_dims(const dims &adims) const {
            BACKEND_DNNL_ENFORCE(
                    adims.size() == data.ndims, "Rank mismatched.");

            dnnl_memory_desc_t md;
            md.ndims = data.ndims;
            md.data_type = data.data_type;

            auto &blk = blocking_desc();

            dims_t blocks;
            for (auto i = 0; i < data.ndims; i++)
                blocks[i] = 1;

            dim_t block_size = 1;
            for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
                blocks[blk.inner_idxs[iblk]] *= blk.inner_blks[iblk];
                block_size *= blk.inner_blks[iblk];
            }

            for (int d = 0; d < data.ndims; ++d) {
                md.dims[d] = adims[d];
                md.padded_dims[d] = utils::rnd_up(adims[d], blocks[d]);
                md.padded_offsets[d] = 0;
            }
            md.offset0 = 0;

            md.format_kind = dnnl_blocked;
            auto &mblk = md.format_desc.blocking;
            mblk = blk;

            for (auto i = 0; i < data.ndims; i++)
                mblk.strides[i] = blk.strides[i];

            int perm[DNNL_MAX_NDIMS];
            for (int d = 0; d < data.ndims; ++d)
                perm[d] = d;

            utils::simultaneous_sort(mblk.strides, perm, data.ndims,
                    [](dim_t a, dim_t b) { return b - a; });

            dim_t stride = block_size;
            for (int _d = data.ndims - 1; _d >= 0; --_d) {
                const int d = perm[_d];
                md.format_desc.blocking.strides[d] = stride;
                stride *= md.padded_dims[d] / blocks[d];
            }

            md.extra = dnnl_memory_extra_desc_t {};

            return desc(md);
        }

    private:
        /// Returns dimension vector
        inline dims get_internal_dims() const {
            return dims(data.dims, data.dims + data.ndims);
        }

        const dims_t &padded_dims() const { return data.padded_dims; }

        const dims_t &padded_offsets() const { return data.padded_offsets; }

        dim_t offset0() const { return data.offset0; }

        inline format_kind_t format_kind() const { return data.format_kind; }

        bool is_blocking_desc() const { return format_kind() == dnnl_blocked; }

        bool is_wino_desc() const {
            return format_kind() == dnnl_format_kind_wino;
        }

        bool is_rnn_packed_desc() const {
            return format_kind() == dnnl_format_kind_rnn_packed;
        }

        const blocking_desc_t &blocking_desc() const {
            BACKEND_DNNL_ENFORCE(is_blocking_desc(),
                    "Cannot get blocking desc on a non-blocking desc");
            return data.format_desc.blocking;
        }

        dims_t &blocking_strides() const {
            BACKEND_DNNL_ENFORCE(is_blocking_desc(),
                    "Cannot get blocking desc on a non-blocking desc");
            return const_cast<dnnl_memory_desc_t &>(data)
                    .format_desc.blocking.strides;
        }

        void set_g(dim groups) {
            auto reserved_size
                    = sizeof(((dnnl_memory_extra_desc_t *)0)->reserved);
            auto offset = reserved_size / sizeof(dim) - 1;
            reinterpret_cast<dim *>(data.extra.reserved)[offset] = groups;
        }

        dim g() const {
            auto reserved_size
                    = sizeof(((dnnl_memory_extra_desc_t *)0)->reserved);
            auto offset = reserved_size / sizeof(dim) - 1;
            return reinterpret_cast<const dim *>(data.extra.reserved)[offset];
        }

        inline bool is_grouped() const { return g() > 1; }

        desc get_tensor_desc(const impl::logical_tensor_t &lt);
    };

    desc get_desc() const {
        const dnnl_memory_desc_t *cdesc;
        error::wrap_c_api(dnnl_memory_get_memory_desc(get(), &cdesc),
                "could not get memory descriptor from a memory");
        return desc(*cdesc);
    }

    // Constructs an tensor with no buffer and zero memory description
    tensor() : dnnl::memory() {}

    // desc, buffer
    tensor(const desc &adesc, const dnnl::engine &aengine,
            const impl::allocator_t *alc, void *ahandle)
        : memory(adesc, aengine, ahandle), alc_(alc) {
        buffer_.reset();
    }

    // desc, no buffer
    tensor(const desc &adesc, const dnnl::engine &aengine,
            const impl::allocator_t *alc)
        : memory(adesc, aengine,
                allocator::malloc(adesc.get_size(), aengine, alc))
        , alc_(alc) {
        buffer_.reset(this->get_data_handle(),
                [aengine, alc](void *p) { allocator::free(p, aengine, alc); });
    }

    // logical tensor, buffer
    tensor(const impl::logical_tensor_t &lt, const dnnl::engine &aengine,
            const impl::allocator_t *alc, void *ahandle)
        : memory(desc(lt), aengine, ahandle), alc_(alc) {
        buffer_.reset();
    }

    // logical tensor, no buffer
    tensor(const impl::logical_tensor_t &lt, const dnnl::engine &aengine,
            const impl::allocator_t *alc)
        : memory(desc(lt), aengine,
                allocator::malloc(desc(lt).get_size(), aengine, alc))
        , alc_(alc) {
        buffer_.reset(this->get_data_handle(),
                [aengine, alc](void *p) { allocator::free(p, aengine, alc); });
    }

    // dnnl_graph::tensor
    tensor(const impl::tensor &impl_tensor, const dnnl::engine &aengine,
            const impl::allocator_t *alc)
        : memory(desc(impl_tensor.get_logical_tensor()), aengine,
                impl_tensor.get_data_handle())
        , alc_(alc) {
        buffer_.reset();
    }

    bool reinit_if_possible(const desc &expected_desc) {
        if (is_empty()) return false;

        dnnl::memory::desc curr_desc = get_desc();
        if (expected_desc != curr_desc) {
            if (curr_desc.dims() == expected_desc.get_dims()) {
                to_format(expected_desc);
            } else {
                tensor tmp {expected_desc, get_engine(), get_allocator()};
                *this = std::move(tmp);
            }
        }
        return true;
    }

    /// Copy constructor
    tensor(const tensor &t)
        : memory(t)
        , workspace_(t.workspace_)
        , buffer_(t.buffer_)
        , alc_(t.alc_) {}

    /// Move constructor
    tensor(tensor &&t)
        : memory(std::move(t))
        , workspace_(std::move(t.workspace_))
        , buffer_(std::move(t.buffer_))
        , alc_(std::move(t.alc_)) {}

    /// Assignment operator
    tensor &operator=(const tensor &t) {
        memory::operator=(t);
        buffer_ = t.buffer_;
        workspace_ = t.workspace_;
        alc_ = t.alc_;
        return *this;
    }

    /// Move assignment operator
    tensor &operator=(tensor &&t) {
        memory::operator=(std::move(t));
        buffer_ = std::move(t.buffer_);
        workspace_ = std::move(t.workspace_);
        alc_ = std::move(t.alc_);
        return *this;
    }

    const impl::allocator_t *get_allocator() const { return alc_; }

    /// Returns number of dimensions
    inline int ndims() const { return get_desc().get_ndims(); }

    /// Return size of specified dimension
    inline dim_t get_dim(int index) const { return get_desc().get_dim(index); }

    /// Returns dimension vector
    inline dims get_dims() const { return get_desc().get_dims(); }

    inline dims get_strides() const { return get_desc().get_strides(); }

    /// Return element number of the param.
    /// The number is the meaning values for a tensor, instead of whole buffer.
    /// It is the number without counting in paddings.
    inline dim_t get_nelems() const { return get_desc().nelems(); }

    /// Returns descriptor data type
    inline data_type get_data_type() const {
        return get_desc().get_data_type();
    }

    inline size_t get_size() const { return get_desc().get_size(); }

    /// Return whether the tensor is empty
    inline bool is_empty() const {
        return get(true) == nullptr
                ? true
                : (get_desc().is_zero() && get_data_handle() == nullptr);
    }

    // "public format" has the same semantic as DNNL's "plain format"
    inline bool is_public_format() const { return get_desc().is_plain(); }

    inline static format_tag get_default_format(const dims &adims) {
        switch (adims.size()) {
            case 1: return format_tag::a;
            case 2: return format_tag::ab;
            case 3: return format_tag::abc;
            case 4: return format_tag::abcd;
            case 5: return format_tag::abcde;
            case 6: return format_tag::abcdef;
            default: return format_tag::undef;
        }
    }

    tensor reorder_if_differ_in(
            const desc &expected_desc, const attr_t &aattr = attr_t()) const {
        if (expected_desc == get_desc()) {
            return *this;
        } else {
            tensor dst {expected_desc, get_engine(), get_allocator()};
            reorder_to(dst, aattr);
            return dst;
        }
    }

    // no data copy
    tensor make_grouped_weights(dim groups, bool is_deconv = false) const {
        if (groups <= 1) return *this;

        auto old_desc = get_desc();
        auto old_groups = old_desc.g();
        if (old_groups > 1) {
            // weights have already been pre-converted if old_groups > 1
            BACKEND_DNNL_ENFORCE(old_groups == groups,
                    "groups does not match the pre-converted weights");
            return *this;
        }

        auto grouped_desc = is_deconv
                ? old_desc.transpose(0, 1).to_grouped(groups).transpose(1, 2)
                : old_desc.to_grouped(groups);
        auto this_copy = *this;
        return this_copy.set_desc(grouped_desc);
    }

    /// Return an new tensor with new shape
    tensor &reshape(const dims &adims) {
        BACKEND_DNNL_ENFORCE(
                has_same_volume(adims), "reshape to incompatible shape");

        // count the number of non-one dimensions
        // e.g. the actual rank of shape [1, 1, 35, 1] is one
        auto actual_rank = [](const dims &shape) {
            auto cnt = 0;
            for (auto d : shape)
                if (d > 1) cnt++;
            return cnt;
        };

        auto old_dims = get_dims();
        if (adims != old_dims) {
            // Since we are going to set the desc to new dims with default
            // format, we have to make sure it's already in default format.
            // In particular, tensor format does not matter if actual rank <= 1
            if (!get_desc().is_default() && actual_rank(old_dims) > 1) {
                to_default_format();
            }
            // set desc with default format
            set_desc({adims, get_data_type()});
        }
        return *this;
    }

    inline void to_default_format() {
        to_format(get_desc().to_default_format());
    }

    inline void to_format(format_tag aformat_tag) {
        to_format(get_desc().to_format(aformat_tag));
    }

    // TODO(xpz): not a good name
    inline void to_type(data_type adata_type) {
        set_desc(get_desc().to_type(adata_type));
    }

    inline void reorder_from(const tensor &src) {
        dnnl::stream s {this->get_engine()};
        dnnl::reorder(src, *this).execute(s, const_cast<tensor &>(src), *this);
        s.wait();
    }

    inline void reorder_to(tensor &dst, const attr_t &aattr = attr_t()) const {
        auto pd = dnnl::reorder::primitive_desc(*this, dst, aattr);
        dnnl::stream s {this->get_engine()};
        dnnl::reorder(pd).execute(s, const_cast<tensor &>(*this), dst);
        s.wait();
    }

    /// Convert the tensor to public format, and f32 data type by default
    tensor to_public(
            void *buffer = nullptr, data_type dst_type = data_type::f32) const {
        auto dst_desc = get_desc();

        // If we get a non-plain blocking format, say `Acdb16A`, we may not be
        // able to recover it to its "unblocked" format `acdb`. Instead, we will
        // convert it to its default format `abcd` based on its dimensions.
        if (!is_public_format()) { dst_desc = dst_desc.to_default_format(); }

        if (dst_type != data_type::undef) {
            dst_desc = dst_desc.to_type(dst_type);
        }

        auto dst = buffer
                ? tensor(dst_desc, get_engine(), get_allocator(), buffer)
                : tensor(dst_desc, get_engine(), get_allocator());

        this->reorder_to(dst);

        return dst;
    }

    void init_workspace(const desc &desc) {
        auto workspace = new tensor(desc, get_engine(), get_allocator());
        workspace_.reset(workspace);
    }

    /// Return extra packed tensor
    tensor &get_workspace() const { return *workspace_; }

    /// Decide wether there is an extra tensor packed in
    bool has_workspace() const { return workspace_ != nullptr; }

    tensor &permute_(const std::vector<int> &permute_axes = {}) {
        return set_desc(get_desc().permute(permute_axes));
    }

    tensor permute(const std::vector<int> &permute_axes = {}) const {
        auto src_mask = *this;
        src_mask.permute_(permute_axes);
        auto dst = tensor(src_mask.get_desc().to_default_format(), get_engine(),
                get_allocator());
        src_mask.reorder_to(dst);
        return dst;
    }

    tensor &transpose_(dim dim0, dim dim1) {
        return set_desc(get_desc().transpose(dim0, dim1));
    }

    tensor transpose(dim dim0, dim dim1) const {
        auto src_mask = *this;
        src_mask.transpose_(dim0, dim1);
        auto dst = tensor(src_mask.get_desc().to_default_format(), get_engine(),
                get_allocator());
        src_mask.reorder_to(dst);
        return dst;
    }

private:
    inline void to_format(const desc &adesc) {
        if (get_desc() != adesc) {
            auto dst = tensor(adesc, get_engine(), get_allocator());
            this->reorder_to(dst);
            *this = std::move(dst);
        }
    }

    bool has_same_volume(const dims &new_dims) const {
        auto old_dims = get_dims();
        auto volume_old = std::accumulate(
                old_dims.begin(), old_dims.end(), 1, std::multiplies<dim_t>());
        auto volume_new = std::accumulate(
                new_dims.begin(), new_dims.end(), 1, std::multiplies<dim_t>());
        return volume_old == volume_new;
    }

    /// Set a descriptor into tensor to replace the older one, keep buffer
    /// It is caller's responsibility to make sure the original buffer is large
    /// enough for specified descriptor
    tensor &set_desc(const desc &new_desc) {
        // Keep the original management
        auto buf = std::move(buffer_);
        auto ws = std::move(workspace_);

        tensor tmp {new_desc, get_engine(), get_allocator(), get_data_handle()};
        *this = std::move(tmp);

        buffer_ = std::move(buf);
        workspace_ = std::move(ws);
        return *this;
    }

    std::shared_ptr<tensor> workspace_;
    std::shared_ptr<void> buffer_;
    const impl::allocator_t *alc_;
};

} // namespace dnnl_impl
} // namespace impl
} // namespace llga

#endif
