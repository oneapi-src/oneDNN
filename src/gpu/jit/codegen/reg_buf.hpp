/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_JIT_CODEGEN_REG_BUF_HPP
#define GPU_JIT_CODEGEN_REG_BUF_HPP

#include <vector>
#include <unordered_set>

#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/ir/grf_permutation.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Represents a register buffer allocated in blocks.
class reg_buf_t {
public:
    reg_buf_t() = default;

    reg_buf_t(ngen::HW hw, const ngen::GRFRange &range)
        : hw_(hw)
        , block_regs_(range.getLen())
        , block_bases_({range.getBase()}) {}

    reg_buf_t(ngen::HW hw, int block_regs, const std::vector<int> &block_bases)
        : hw_(hw), block_regs_(block_regs), block_bases_(block_bases) {}

    bool is_empty() const { return block_bases_.empty(); }

    ngen::HW hw() const { return hw_; }

    bool with_permute() const { return !grf_perm_.is_empty(); }

    int base(int reg_idx, bool apply_permute = true) const {
        if (apply_permute && !grf_perm_.is_empty())
            reg_idx = grf_perm_.map(reg_idx);
        ir_assert(reg_idx >= 0 && reg_idx < regs())
                << "Invalid index: " << reg_idx;
        int block_idx = reg_idx / block_regs_;
        return block_bases_[block_idx] + (reg_idx % block_regs_);
    }

    int blocks() const { return int(block_bases_.size()); }

    int block_regs() const { return block_regs_; }

    int regs() const { return blocks() * block_regs(); }

    void set_grf_permutation(const grf_permutation_t &grf_perm) {
#if !defined(NDEBUG) || defined(DNNL_DEV_MODE)
        // Check that it's a valid permutation.
        std::unordered_set<int> seen;
        for (int i = 0; i < regs(); i++) {
            int i_mapped = grf_perm.map(i);
            ir_assert(i_mapped >= 0 && i_mapped < regs());
            seen.insert(i_mapped);
        }
        ir_assert(int(seen.size()) == regs()) << "Invalid permutation.";
#endif
        grf_perm_ = grf_perm;
    }

    bool operator==(const reg_buf_t &other) const {
        if (hw() != other.hw()) return false;
        if (block_regs() != other.block_regs()) return false;
        if (blocks() != other.blocks()) return false;
        for (int i = 0; i < blocks(); i++) {
            if (block_bases_[i] != other.block_bases_[i]) return false;
        }
        if (grf_perm_ != other.grf_perm_) return false;
        return true;
    }

    void claim(reg_allocator_t &ra) const {
        for (int i = 0; i < blocks(); i++) {
            ngen::GRFRange range(block_bases_[i], block_regs_);
            ra.claim(range);
        }
    }

    void release(reg_allocator_t &ra) const {
        for (int i = 0; i < blocks(); i++) {
            ngen::GRFRange range(block_bases_[i], block_regs_);
            ra.safeRelease(range);
        }
    }

private:
    ngen::HW hw_ = ngen::HW::Unknown;
    int block_regs_ = 0;
    std::vector<int> block_bases_;
    grf_permutation_t grf_perm_;
};

// ngen::RegData wrapper attached to a register buffer.
class reg_buf_data_t {
public:
    reg_buf_data_t() = default;

    reg_buf_data_t(const reg_buf_t &reg_buf)
        : reg_buf_(std::make_shared<reg_buf_t>(reg_buf))
        , rd_(ngen::GRF(reg_buf_->base(0))) {}

    reg_buf_data_t(const reg_buf_t &reg_buf, const ngen::RegData &rd)
        : reg_buf_(std::make_shared<reg_buf_t>(reg_buf)), rd_(rd) {}

    reg_buf_data_t(ngen::HW hw, const ngen::Subregister &sub)
        : reg_buf_(std::make_shared<reg_buf_t>(
                hw, ngen::GRFRange(sub.getBase(), 1)))
        , rd_(sub) {}

    bool is_empty() const { return !reg_buf_; }

    bool with_permute() const { return reg_buf_->with_permute(); }

    ngen::HW hw() const { return reg_buf_->hw(); }

    ngen::DataType type() const { return rd_.getType(); }

    int base() const { return rd_.getBase(); }

    int byte_offset() const { return rd_.getByteOffset(); }

    int offset() const { return rd_.getOffset(); }

    int hs() const { return rd_.getHS(); }

    const ngen::RegData &reg_data() const { return rd_; }

    operator ngen::RegData() const { return rd_; }

    void set_grf_permutation(const grf_permutation_t &grf_perm) {
        reg_buf_->set_grf_permutation(grf_perm);
    }

    bool check_bounds(
            int off_bytes, int len_bytes, bool is_dense = false) const {
        ir_assert(off_bytes >= 0);
        ir_assert(len_bytes >= 0);
        if (len_bytes == 0) return true;

        int grf_size = ngen::GRF::bytes(hw());
        int beg_off = (byte_offset() + off_bytes) / grf_size;
        int end_off = (byte_offset() + off_bytes + len_bytes - 1) / grf_size;

        // Check for out of bound accesses.
        if (get_grf_buf_index() + end_off >= reg_buf_->regs()) return false;

        // Check if access is dense.
        if (is_dense) {
            int base0 = get_grf_base(beg_off);
            for (int i = beg_off + 1; i < end_off + 1; i++) {
                if (get_grf_base(i) != base0 + i) return false;
            }
        }
        return true;
    }

    bool is_dense(int bytes) const {
        ir_assert(check_bounds(0, bytes)) << "Invalid access.";
        return check_bounds(0, bytes, /*is_dense=*/true);
    }

    bool operator==(const reg_buf_data_t &other) const {
        return (*reg_buf_ == *other.reg_buf_) && (rd_ == other.rd_);
    }

    bool operator!=(const reg_buf_data_t &other) const {
        return !operator==(other);
    }

    // Retype register region while preserving data.
    reg_buf_data_t reinterpret(ngen::DataType new_type) const {
        int new_size = ngen::getBytes(new_type);
        int old_size = ngen::getBytes(type());
        if (new_size == old_size) {
            auto ret = *this;
            ret.rd_.setType(new_type);
            return ret;
        } else if (new_size < old_size) {
            ir_assert(rd_.getHS() <= 1) << "Can't reinterpret strided data to "
                                           "differently sized data type.";
            return format(0, new_type, rd_.getWidth() * old_size / new_size, 1);
        } else {
            ir_error_not_expected() << "Can't reinterpret to larger data type.";
        }
        return reg_buf_data_t();
    }

    ngen::Subregister subregister(int off_bytes,
            ngen::DataType type = ngen::DataType::invalid) const {
        ir_assert(check_bounds(off_bytes, 1)) << "Invalid access.";
        if (type == ngen::DataType::invalid) type = rd_.getType();
        auto rd = format(off_bytes, type, 1, 0).reg_data();
        return ngen::Subregister(rd, rd.getOffset(), rd.getType());
    }

    ngen::Subregister subregister(int off, int width, int stride_bytes,
            ngen::DataType type = ngen::DataType::invalid) const {
        if (type == ngen::DataType::invalid) type = rd_.getType();
        int off_bytes = off * stride_bytes;

        ir_assert(check_bounds(off_bytes, stride_bytes * (width - 1)))
                << "Invalid access.";

        auto rd = format(off_bytes, type, 1, 0).reg_data();
        return ngen::Subregister(rd, rd.getOffset(), rd.getType());
    }

    // Format register region to parameters regardless of data.
    reg_buf_data_t format(int off_bytes,
            ngen::DataType type = ngen::DataType::invalid, int width = 1,
            int hstride = 1) const {
        if (type == ngen::DataType::invalid) type = rd_.getType();
        auto grf_size = ngen::GRF::bytes(hw());
        auto new_off = rd_.getByteOffset() + off_bytes;
        auto new_grf_off = new_off % grf_size;
        auto type_size = ngen::getBytes(type);
        auto grf = get_grf(new_off / grf_size).retype(type);

        ir_assert(new_grf_off % type_size == 0);

        if (width == 1) {
            hstride = 0;
        } else if (hstride == 0) {
            ir_assert(width == 1);
        } else {
            int max_width = 32 / type_size;
            width = std::min(width, max_width / hstride);
            width = std::min(width, 16);
        }
        int vstride = width * hstride;

        int region_bytes = ((width - 1) * hstride + 1) * type_size;
        ir_assert(check_bounds(off_bytes, region_bytes)) << "Invalid access.";

        auto ret = *this;
        ret.rd_ = grf[new_grf_off / type_size](vstride, width, hstride);
        return ret;
    }

    reg_buf_data_t unpermute() const {
        int idx = get_grf_buf_index();
        int base = reg_buf_->base(idx, /*apply_permute=*/false);

        auto ret = *this;
        ret.rd_.setBase(base);
        return ret;
    }

private:
    ngen::GRF get_grf(int off_regs) const {
        return ngen::GRF(get_grf_base(off_regs));
    }

    int get_grf_base(int off_regs) const {
        int idx = get_grf_buf_index();
        return reg_buf_->base(idx + off_regs);
    }

    int get_grf_buf_index() const {
        if (reg_buf_->blocks() == 1 && !reg_buf_->with_permute()) {
            return rd_.getBase() - reg_buf_->base(0);
        }
        for (int i = 0; i < reg_buf_->regs(); i++) {
            if (reg_buf_->base(i) == rd_.getBase()) return i;
        }
        ir_error_not_expected();
        return -1;
    }

    std::shared_ptr<reg_buf_t> reg_buf_;
    ngen::RegData rd_;
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
