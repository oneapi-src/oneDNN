/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "gpu/intel/compute/dispatch_reusable.hpp"
#include "common/c_types_map.hpp"
#include "gemmstone/../../dsl/ir/core.hpp"
#include "gemmstone/../../dsl/ir/pass/simplify.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/data_type_converter.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

// Enables the use of intel subgroups in the kernel.
// A buffer is supplied to specify which block (stride=1 block for the buffer)
// is guaranteed to be dispatched across in the subgroup. Memory access
// patterns may be non-contiguous in other buffers (i.e. block read/write is only guaranteed
// to be valid for this buffer)
status_t reusable_dispatch_config_t::use_subgroup(
        name_id_t buf_name, size_t size) {
    if (!engine->mayiuse_sub_group(static_cast<int>(size))) {
        return status::unimplemented;
    }

    // Cannot use a subgroup on two buffers
    gpu_assert(!subgroup.used());

    // Look for a registered buffer with the given name
    for (size_t i = 0; i < buffers.size(); i++) {
        if (buffers[i].get_name_id() == buf_name) {
            subgroup = subgroup_data_t(i, size);
            break;
        }
    }

    // If we couldn't find the buffer, something has gone wrong
    if (!subgroup.used()) { return status::runtime_error; }

    return status::success;
}

status_t reusable_dispatch_config_t::define_dim_index(
        name_id_t dim_name, dim_idx_t dim_id, dim_t size) {
    memory_desc_t md = types::zero_md();
    md.ndims = 1;
    md.dims[0] = size;
    md.padded_dims[0] = size;
    md.data_type = data_type::f32; // doesn't matter
    md.format_kind = format_kind::blocked;
    md.format_desc.blocking.strides[0] = 1;
    md.format_desc.blocking.inner_nblks = 0;

    named_buffer_t buf(dim_name, md, {dim_id});
    CHECK(register_buffer(buf));
    return status::success;
}

// Validate whether the given buffer is consistent with existing buffer layouts,
// and then add to the internal buffer registry.
status_t reusable_dispatch_config_t::register_buffer(
        const named_buffer_t &buffer) {
    if (buffers.size() >= MAX_REGISTERED_BUFFERS) return status::unimplemented;

    // Don't allow zero-padding
    bool has_zero_padding = false;
    for (size_t dim_idx = 0; dim_idx < static_cast<size_t>(buffer.ndims);
            dim_idx++) {
        if (buffer.dims[dim_idx] < buffer.padded_dims[dim_idx]) {
            has_zero_padding = true;
        }
    }
    if (has_zero_padding) return status::unimplemented;

    // Validate dim sizes
    std::unordered_map<dim_idx_t, bool, dim_id_hash_t> dim_seen;
    for (const auto &dim : dispatched_dims) {
        size_t canonical_idx = buffer.get_dim_idx(dim);
        if (canonical_idx == dim_not_found) {
            // broadcasted dimension - nothing to check
            continue;
        }

        dim_seen[dim] = (dim_sizes.find(dim) != dim_sizes.end());

        if (dim_seen[dim] && (dim_sizes[dim] != buffer.dims[canonical_idx])) {
            // Attempting to dispatch to multiple buffers with differently
            // sized dispatched dimensions. These buffers are incompatible.
            return status::runtime_error;
        }
    }

    // All validation complete - start updating this object
    for (const auto &dim : dispatched_dims) {
        size_t canonical_idx = buffer.get_dim_idx(dim);
        if (canonical_idx == dim_not_found) continue;

        // Save the dimension size if it hasn't been saved yet
        if (!dim_seen[dim]) { dim_sizes[dim] = buffer.dims[canonical_idx]; }
    }
    buffers.emplace_back(buffer);
    return status::success;
}

// Will mutate a vector of layouts as needed to make each dimension:
// 1. have the same number of blocks,
// 2. each with the same size,
// 3. in the same order
// by subdividing existing blocks
class layout_equalizer_t {
public:
    static constexpr int broadcasted_block = -1;
    layout_equalizer_t() = default;

    status_t register_layout(const block_layout_t &layout) {
        if (master_layout.empty()) {
            for (const block_t &block : layout) {
                master_layout.emplace_back(num_layouts, block);
            }
            num_layouts++;
            return status::success;
        }

        // subdivide the new and master layouts as needed to match
        block_layout_t new_layout;
        CHECK(subdivide(layout, new_layout));

        // For each block, find the correct master term to add to
        std::vector<bool> is_mapped_to(master_layout.size(), false);
        for (const block_t &block : new_layout) {
            bool is_mapped = false;
            for (size_t i = 0; i < master_layout.size(); i++) {
                if (is_mapped_to[i]) continue;

                auto &master_block = master_layout[i];
                if (master_block.matches(block)) {
                    is_mapped = true;
                    is_mapped_to[i] = true;
                    master_block.map(num_layouts, block);
                    break;
                }
            }
            if (!is_mapped) {
                master_layout.emplace_back(num_layouts, block);
                is_mapped_to.push_back(true);
            }
        }
        num_layouts++;

        return status::success;
    }

    const std::unordered_map<size_t, block_t> &buffer_blocks(size_t idx) {
        return master_layout[idx].get_buffer_blocks();
    }

    // mutates master_layout and returns a matching layout
    status_t subdivide(const block_layout_t &layout, block_layout_t &res) {
        // Can subdivide as long as all dims have the same size as master_layout
        // (or layout size is 1, as in broadcasted dims)
        std::array<size_t, DNNL_MAX_NDIMS> layout_dim_sizes;
        layout_dim_sizes.fill(1);
        for (const block_t &block : layout) {
            layout_dim_sizes[static_cast<size_t>(block.dim_idx)]
                    *= static_cast<size_t>(block.block);
        }

        std::array<size_t, DNNL_MAX_NDIMS> master_dim_sizes;
        master_dim_sizes.fill(1);
        for (const mapped_block_t &block : master_layout) {
            master_dim_sizes[block.get_dim_idx()] *= block.get_size();
        }

        for (size_t i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (layout_dim_sizes[i] == 1 || master_dim_sizes[i] == 1) continue;
            if (layout_dim_sizes[i] != master_dim_sizes[i]) {
                return status::runtime_error;
            }
        }

        // Shapes are coherent, start subdividing
        res = layout;
        std::vector<bool> is_mapped_to(master_layout.size(), false);
        for (size_t i = 0; i < res.size(); i++) {
            block_t &block = res[i];
            dim_t block_size = block.block;
            for (size_t j = 0; j < master_layout.size(); j++) {
                if (is_mapped_to[j]) continue;

                mapped_block_t &master_block = master_layout[j];
                if (master_block.get_dim_idx() != block.dim_idx) continue;

                dim_t master_size = master_block.get_size();
                if (master_size == block_size) {
                    // Nothing to do, already matches
                } else if (block_size % master_size == 0) {
                    // subdivide block
                    block.block = master_size;
                    block_t next_block(block.dim_idx, block_size / master_size,
                            block.stride * master_size);
                    res.insert(i + 1, next_block);
                } else if (master_size % block_size == 0) {
                    // subdivide master block
                    mapped_block_t next_block = master_block.split(block_size);
                    master_layout.insert(
                            master_layout.begin() + j + 1, next_block);
                } else {
                    // Should never be able to reach this case...
                    return status::runtime_error;
                }
                is_mapped_to[j] = true;
                break;
            }
        }

        return status::success;
    }

    std::vector<block_bin_t> compute_block_bins(
            const lws_strategy_t &lws_strat, const subgroup_data_t &subgroup) {
        std::vector<block_bin_t> bins;
        for (size_t i = 0; i < master_layout.size(); i++) {
            const mapped_block_t &mapped_blocks = master_layout[i];

            // mapped_block_t that are in the lws have to be
            // at the start of a new bin
            if (subgroup.used()) {
                // The subgroup block has to be in the lws
                size_t sg_buf_idx = subgroup.buffer_idx();
                if (!mapped_blocks.is_broadcasted(sg_buf_idx)) {
                    const block_t &buf_block
                            = mapped_blocks.get_buffer_blocks().at(sg_buf_idx);
                    if (buf_block.stride * buf_block.block <= subgroup.size()) {
                        // This mapped_block_t corresponds to the subgroup block
                        bins.emplace_back(mapped_blocks, num_layouts, true);
                        continue;
                    }
                }
            }

            // The lws_strategy_t can specify other blocks to be in the lws as well
            if (lws_strat.is_included(mapped_blocks)) {
                bins.emplace_back(mapped_blocks, num_layouts, true);
                continue;
            }

            bool found_bin = false;
            for (block_bin_t &bin : bins) {
                if (bin.get_blocks().back().can_merge(mapped_blocks)) {
                    found_bin = true;
                    bin.append(mapped_blocks);
                    break;
                }
            }
            if (!found_bin) bins.emplace_back(mapped_blocks, num_layouts);
        }

        return bins;
    }

private:
    void split_block(size_t block_idx, size_t size) {
        mapped_block_t next_block = master_layout[block_idx].split(size);
        master_layout.insert(master_layout.begin() + block_idx + 1, next_block);
    }

    std::vector<mapped_block_t> master_layout;
    size_t num_layouts = 0;
};

// Used in compute_terms to store the block_t data and info about
// where it's mapped to in the GWS
struct gws_mapped_block_t : public gpu::intel::block_t {
    gws_mapped_block_t() = default;
    gws_mapped_block_t(
            const block_t &block, size_t gws_idx, stride_t gws_stride)
        : block_t(block), gws_idx(gws_idx), gws_stride(gws_stride) {}

    std::string str() const {
        ostringstream_t ss;
        ss << static_cast<const block_t *>(this)->str().c_str();
        ss << " , gws_stride=" << gws_stride.str();
        ss << " / gws_idx=" << gws_idx;
        return ss.str();
    }

    size_t gws_idx;
    stride_t gws_stride;
};

namespace dsl = jit::dsl;
namespace ir = gemmstone::dsl::ir;

const dsl::expr_t &global_ids(size_t idx) {
    static const thread_local std::array<dsl::expr_t, 3> ids([] {
        std::array<dsl::expr_t, 3> ret;
        for (int i = 0; i < 3; i++) {
            ret[i] = ir::var_t::make(
                    dsl::u64, "get_global_id(" + std::to_string(i) + ")");
        }
        return ret;
    }());
    return ids[idx];
}

struct term_registry_t {

    dsl::expr_t add(int64_t value, const std::string &name) {
        auto name_it = name_map.emplace(name, dsl::expr_t());
        auto &expr = name_it.first->second;
        if (name_it.second) expr = ir::var_t::make(dsl::s64, name);

        auto value_it = value_map.emplace(expr, value);
        gpu_assert(value == value_it.first->second);
        return expr;
    }
    std::unordered_map<std::string, dsl::expr_t> name_map;
    ir::object_map_t<dsl::expr_t, int64_t> value_map;
};

// Encodes an dsl::expr into a compressed format optimized for calculating
// buffer offsets. The general format of an expression is a prefix (kind_t)
// followed by a fixed size sequence of bytes corresponding any following
// expressions.
struct expr_encoder_t : public ir::ir_visitor_t {
    uint8_t operator()(
            const dsl::expr_t &offset, const term_registry_t &registry) {
        registry_ = &registry;
        ir::ir_visitor_t::visit(offset);
        return expr_locations[offset];
    }
    const std::vector<uint8_t> &expr_data() const { return expr_data_; }
    const std::vector<int64_t> &term_list() const { return term_list_; }

    static std::string decode(const uint8_t *expr_data, uint8_t offset,
            std::vector<int64_t> *term_list = nullptr) {
        auto dec = [&](uint8_t off) {
            return decode(expr_data, expr_data[offset + off], term_list);
        };
        auto sym = [](uint8_t kind) {
            switch (kind_t(kind)) {
                case kind_t::add: return "+";
                case kind_t::sub: return "-";
                case kind_t::mul: return "*";
                case kind_t::div: return "/";
                case kind_t::mod: return "%";
                case kind_t::_or: return "||";
                case kind_t::_and: return "&&";
                case kind_t::lt: return "<";
                case kind_t::le: return "<=";
                case kind_t::gt: return ">";
                case kind_t::ge: return ">=";
                case kind_t::ne: return "!=";
                case kind_t::eq: return "==";
                default: gpu_error_not_expected(); return "";
            }
        };
        ostringstream_t oss;
        switch (kind_t(expr_data[offset])) {
            case kind_t::add:
            case kind_t::sub:
            case kind_t::mul:
            case kind_t::div:
            case kind_t::mod:
            case kind_t::_or:
            case kind_t::_and:
            case kind_t::lt:
            case kind_t::le:
            case kind_t::gt:
            case kind_t::ge:
            case kind_t::ne:
            case kind_t::eq:
                oss << "(" << dec(1) << sym(expr_data[offset]) << dec(2) << ")";
                break;
            case kind_t::idiv: {
                auto rt_off = expr_data[offset + 2];
                oss << "idiv(" << dec(1) << ","
                    << "rt.params[" + std::to_string(rt_off) << "],"
                    << "rt.params[" + std::to_string(rt_off + 1) << "])";
                break;
            }
            case kind_t::runtime_term:
                oss << "rt.params["
                    << std::to_string(int(expr_data[offset + 1])) << "]";
                if (term_list)
                    oss << "(="
                        << std::to_string((*term_list)[expr_data[offset + 1]])
                        << ")";
                break;
            case kind_t::constant_true: oss << "true"; break;
            case kind_t::constant_false: oss << "false"; break;
            case kind_t::constant_s8:
                oss << std::to_string(int8_t(expr_data[offset + 1]));
                break;
            case kind_t::constant_u8:
                oss << std::to_string(expr_data[offset + 1]) << "u";
                break;
            case kind_t::constant_s16:
                oss << std::to_string(int16_t(
                        expr_data[offset + 1] + (expr_data[offset + 2] << 8)));
                break;
            case kind_t::constant_u16:
                oss << std::to_string(uint16_t(
                        expr_data[offset + 1] + (expr_data[offset + 2] << 8)))
                    << "u";
                break;
            case kind_t::constant_s32:
                oss << std::to_string(int32_t(expr_data[offset + 1]
                        + (expr_data[offset + 2] << 8)
                        + (expr_data[offset + 3] << 16)
                        + (expr_data[offset + 4] << 24)));
                break;
            case kind_t::constant_u32:
                oss << std::to_string(uint32_t(expr_data[offset + 1]
                        + (expr_data[offset + 2] << 8)
                        + (expr_data[offset + 3] << 16)
                        + (expr_data[offset + 4] << 24)))
                    << "u";
                break;

            case kind_t::global_id0: oss << global_ids(0).str(); break;
            case kind_t::global_id1: oss << global_ids(1).str(); break;
            case kind_t::global_id2: oss << global_ids(2).str(); break;
            default: gpu_error_not_expected();
        }
        return oss.str();
    }
    std::string decode(uint8_t offset) {
        return decode(expr_data_.data(), offset, &term_list_);
    }

private:
    void _visit(const ir::unary_op_t &obj) override {
        gpu_except_not_implemented();
    }
    void _visit(const ir::binary_op_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        auto offset = it.first->second;
        expr_data_.resize(expr_data_.size() + 3);

        ir::ir_visitor_t::visit(obj.a);
        ir::ir_visitor_t::visit(obj.b);

        expr_data_[offset] = uint8_t(to_kind(obj.op_kind));
        expr_data_[offset + 1] = expr_locations.at(obj.a);
        expr_data_[offset + 2] = expr_locations.at(obj.b);
    }

    void _visit(const ir::ternary_op_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        gpu_assert(obj.op_kind == ir::op_kind_t::_idiv);
        auto offset = it.first->second;
        expr_data_.resize(expr_data_.size() + 3);

        // Directly encode constants for idiv to save encoding space, since
        // the magic numbers are uniquely associated with this object.
        ir::ir_visitor_t::visit(obj.a);
        auto b = into<int8_t>(term_list_.size());
        term_list_.emplace_back(registry_->value_map.at(obj.b));
        term_list_.emplace_back(registry_->value_map.at(obj.c));

        expr_data_[offset] = uint8_t(to_kind(obj.op_kind));
        expr_data_[offset + 1] = expr_locations.at(obj.a);
        expr_data_[offset + 2] = b;
    }

    void _visit(const ir::var_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        int idx = gid_idx(obj);
        if (idx >= 0) {
            expr_data_.emplace_back(uint8_t(kind_t::global_id) + idx);
        } else {
            expr_data_.emplace_back(uint8_t(kind_t::runtime_term));
            expr_data_.emplace_back(into<uint8_t>(term_list_.size()));
            term_list_.emplace_back(registry_->value_map.at(obj));
        }
    }

    void _visit(const ir::bool_imm_t &obj) override {
        if (obj.value)
            expr_data_.emplace_back(uint8_t(kind_t::constant_true));
        else
            expr_data_.emplace_back(uint8_t(kind_t::constant_false));
    }

    void _visit(const ir::int_imm_t &obj) override {
        auto it = expr_locations.emplace(obj, into<uint8_t>(expr_data_.size()));
        if (!it.second) return;

        if (obj.value == uint8_t(obj.value) || obj.value == int8_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint8_t(obj.value)
                            ? uint8_t(kind_t::constant_u8)
                            : uint8_t(kind_t::constant_s8));
            expr_data_.emplace_back(into<uint8_t>(obj.value));
            return;
        }
        if (obj.value == uint16_t(obj.value)
                || obj.value == int16_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint16_t(obj.value)
                            ? uint8_t(kind_t::constant_u16)
                            : uint8_t(kind_t::constant_s16));
            expr_data_.emplace_back(uint8_t(obj.value));
            expr_data_.emplace_back(uint8_t(obj.value >> 8));
            return;
        }
        if (obj.value == uint32_t(obj.value)
                || obj.value == int32_t(obj.value)) {
            expr_data_.emplace_back(obj.value == uint32_t(obj.value)
                            ? uint8_t(kind_t::constant_u32)
                            : uint8_t(kind_t::constant_s32));
            expr_data_.emplace_back(uint8_t(obj.value));
            expr_data_.emplace_back(uint8_t(obj.value >> 8));
            expr_data_.emplace_back(uint8_t(obj.value >> 16));
            expr_data_.emplace_back(uint8_t(obj.value >> 24));
            return;
        }
        gpu_except_not_implemented();
    }

    int gid_idx(const ir::var_t &var) {
        if (var == global_ids(0).as<ir::var_t>()) return 0;
        if (var == global_ids(1).as<ir::var_t>()) return 1;
        if (var == global_ids(2).as<ir::var_t>()) return 2;
        return -1;
    }

    enum class kind_t : uint8_t {
        undef = 0,
        add, // encoded operands (expr, expr)
        sub, // encoded operands (expr, expr)
        mul, // encoded operands (expr, expr)
        div, // encoded operands (expr, expr)
        idiv, // encoded operands (expr, term)
        mod, // encoded operands (expr, expr)
        _or, // encoded operands (expr, expr)
        _and, // encoded operands (expr, expr)
        lt, // encoded operands (expr, expr)
        le, // encoded operands (expr, expr)
        gt, // encoded operands (expr, expr)
        ge, // encoded operands (expr, expr)
        ne, // encoded operands (expr, expr)
        eq, // encoded operands (expr, expr)
        runtime_term, // encoded operands (term)
        constant_true, // encodes true, no operands
        constant_false, // encodes false, no operands
        constant_s8, // encoded operands (int8_t)
        constant_u8, // encoded operands (uint8_t)
        constant_s16, // encoded operands (int16_t)
        constant_u16, // encoded operands (uint16_t)
        constant_s32, // encoded operands (int32_t)
        constant_u32, // encoded operands (int32_t)
        global_id, // encodes expression for global_id, no operands
        global_id0 = global_id,
        global_id1 = global_id + 1,
        global_id2 = global_id + 2,
    };

    kind_t to_kind(ir::op_kind_t op_kind) {
        switch (op_kind) {
            case ir::op_kind_t::_add: return kind_t::add;
            case ir::op_kind_t::_sub: return kind_t::sub;
            case ir::op_kind_t::_mul: return kind_t::mul;
            case ir::op_kind_t::_div: return kind_t::div;
            case ir::op_kind_t::_idiv: return kind_t::idiv;
            case ir::op_kind_t::_mod: return kind_t::mod;
            case ir::op_kind_t::_or: return kind_t::_or;
            case ir::op_kind_t::_and: return kind_t::_and;
            case ir::op_kind_t::_lt: return kind_t::lt;
            case ir::op_kind_t::_le: return kind_t::le;
            case ir::op_kind_t::_gt: return kind_t::gt;
            case ir::op_kind_t::_ge: return kind_t::ge;
            case ir::op_kind_t::_ne: return kind_t::ne;
            case ir::op_kind_t::_eq: return kind_t::eq;
            default: gpu_except_not_implemented();
        }
        return kind_t::undef;
    }

    std::vector<uint8_t> expr_data_;
    std::vector<int64_t> term_list_;
    ir::object_eq_map_t<dsl::expr_t, uint8_t> expr_locations;
    const term_registry_t *registry_;
};

dsl::expr_t calculate_buffer_offset(const gws_bin_mapping_t &gws_map,
        dim_idx_t buf_idx, term_registry_t &registry) {
    dsl::expr_t ret = dsl::expr_t(0);

    for (size_t gws_idx = 0; gws_idx < range_t::max_ndims; gws_idx++) {
        const std::vector<block_bin_t> &bins = gws_map.get_bins(gws_idx);

        std::vector<gws_mapped_block_t> gws_blocks;
        stride_t gws_stride = 1;
        for (size_t i = 0; i < bins.size(); i++) {
            const block_bin_t &bin = bins[i];
            if (!bin.is_broadcasted(buf_idx)) {
                block_t block = bin.combined_block(buf_idx);
                gws_blocks.emplace_back(block, gws_idx, gws_stride);
            };
            gws_stride *= static_cast<dim_t>(bin.size());
        }

        for (size_t i = 0; i < gws_blocks.size(); i++) {
            auto block = gws_blocks[i];
            std::string dim_suffix = std::to_string(gws_idx) + "["
                    + std::to_string(block.dim_idx) + "]";
            std::string size_name = "size_dim" + dim_suffix;

            // Merge dense blocks
            while (i + 1 < gws_blocks.size()) {
                gws_mapped_block_t &next_block = gws_blocks[i + 1];
                bool is_buffer_dense
                        = (block.stride * block.block == next_block.stride);
                bool is_gws_dense = (block.gws_stride * block.block
                        == next_block.gws_stride);

                if (!(is_buffer_dense && is_gws_dense)) break;

                size_name += "*size_dim" + std::to_string(gws_idx) + "["
                        + std::to_string(next_block.dim_idx) + "]";
                block.block *= next_block.block;
                i++;
            }

            dsl::expr_t outer_stride = registry.add(int64_t(block.stride),
                    "buffer[" + std::to_string(buf_idx) + "].stride"
                            + dim_suffix);

            dsl::expr_t inner_stride = registry.add(
                    int64_t(block.gws_stride), "stride_dim" + dim_suffix);
            dsl::expr_t size = registry.add(block.block, size_name);

            // TODO: When using int32_t offsets, use idiv and imod
            if (size_t(block.block) * block.gws_stride
                    == gws_map.gws()[gws_idx])
                ret += global_ids(gws_idx) / inner_stride * outer_stride;
            else
                ret += global_ids(gws_idx) / inner_stride % size * outer_stride;
        }
    }

    ret = ir::simplify(ret);
    return ret;
}

// XXX: Mapping blocks into the gws cannot happen until all necessary dim indices
// have been requested and all buffers have been registered. Only then can the terms
// be computed, thus it's all done in the generate function
status_t reusable_dispatch_config_t::generate(
        reusable_dispatch_t &dispatch, const lws_strategy_t &lws_strategy) {
    // The reusable dispatcher must have at least one buffer to dispatch against
    gpu_assert(!buffers.empty());

    // Sort to enable deterministic output and simplify serialization
    std::sort(buffers.begin(), buffers.end(),
            [&](const named_buffer_t &a, const named_buffer_t &b) {
        return a.get_name_id() < b.get_name_id();
    });

    // Every dispatched dim must have a defined size
    for (dim_idx_t id : dispatched_dims) {
        if (dim_sizes.find(id) == dim_sizes.end()) {
            return status::unimplemented;
        }
    }

    std::array<bool, DNNL_MAX_NDIMS> is_dispatched;
    is_dispatched.fill(false);
    for (dim_idx_t dim : dispatched_dims) {
        is_dispatched[dim] = true;
    }

    // Store layouts for each buffer, since they'll be manipulated
    // for the rest of the generate function
    layout_equalizer_t equalizer;
    std::vector<block_layout_t> buf_layouts(buffers.size());
    for (size_t i = 0; i < buffers.size(); i++) {
        block_layout_t layout = buffers[i].layout();
        block_layout_t new_layout;
        // Only keep dispatched blocks
        for (const auto &block : layout) {
            if (is_dispatched[static_cast<size_t>(block.dim_idx)]) {
                new_layout.append(block);
            }
        }
        buf_layouts[i] = new_layout;
        CHECK(equalizer.register_layout(new_layout));
    }

    std::vector<block_bin_t> bins
            = equalizer.compute_block_bins(lws_strategy, subgroup);

    // Map bins into gws dims - start with lws bins, then map the rest
    gws_bin_mapping_t gws_map(subgroup);
    for (const block_bin_t &bin : bins) {
        if (bin.is_in_lws()) gws_map.add(bin);
    }
    for (const block_bin_t &bin : bins) {
        if (!bin.is_in_lws()) gws_map.add(bin);
    }

    std::vector<uint8_t> buffer_exprs;
    expr_encoder_t encoder;
    term_registry_t registry;
    for (dim_idx_t buf_idx = 0; buf_idx < buffers.size(); buf_idx++) {
        // Deduplicate buffers with the same layout
        for (dim_idx_t i = 0; i < buf_idx; i++) {
            if (buffers[i].layout() == buffers[buf_idx].layout()) {
                buffer_exprs.emplace_back(buffer_exprs[i]);
                break;
            }
        }
        if (buffer_exprs.size() > buf_idx) continue;

        buffer_exprs.emplace_back(encoder(
                calculate_buffer_offset(gws_map, buf_idx, registry), registry));
    }

    if (encoder.expr_data().size() >= MAX_EXPR_TERMS)
        return status::unimplemented;
    if (encoder.term_list().size() >= MAX_RUNTIME_TERMS)
        return status::unimplemented;
    if (buffer_exprs.size() >= MAX_REGISTERED_BUFFERS)
        return status::unimplemented;

    dispatch = reusable_dispatch_t(gws_map.nd_range(lws_strategy), subgroup,
            buffers, buffer_exprs, encoder.expr_data(), encoder.term_list());

    return status::success;
}

void dispatch_compile_params_t::def_kernel_macros(
        kernel_ctx_t &kernel_ctx, const char *suffix) const {
    kernel_ctx.define_int("GWS_WITH_RUNTIME_PARAMS", 1);
    kernel_ctx.use_int32_offset(use_int32_offset);
    if (use_int32_offset) kernel_ctx.define_int("GWS_USE_PARAMS32", 1);

    // Define data types for conversion (Ignore the default suffix)
    std::string conv_suff = (suffix == std::string("DEFAULT"))
            ? ""
            : utils::format("_%s", suffix);

    // For each buffer, define the sum that leads to the offset calculation
    data_type_converter_t converter;
    int buf_idx = 0;
    for (size_t bit_idx = 0; bit_idx < sizeof(name_id_t) * 8; bit_idx++) {
        name_id_t name_id = name_id_t(uint64_t(1) << bit_idx);
        if (!static_cast<uint64_t>(buffer_set & name_id)) continue;

        if (buffer_types[buf_idx] != data_type::undef) {
            converter.register_type(
                    to_string(name_id) + conv_suff, buffer_types[buf_idx]);
        }

        std::string equation
                = expr_encoder_t::decode(exprs, buffer_off_expr[buf_idx]);
        kernel_ctx.add_option(utils::format("-DGWS_%s_%s_OFF(rt)=(%s)",
                to_string(name_id), suffix, equation));
        buf_idx++;
    }
    converter.def_kernel_macros(kernel_ctx);

    kernel_ctx.define_int(
            utils::format("GWS_WITH_SG_%s", suffix), subgroup.used() ? 1 : 0);
    if (subgroup.used()) {
        kernel_ctx.define_int(utils::format("GWS_SGS_%s", suffix),
                static_cast<int64_t>(subgroup.size()));
    }
}

std::string dispatch_compile_params_t::str() const {
    ostringstream_t ss;
    auto buf_idx = 0;
    ss << "{";
    for (size_t bit_idx = 0; bit_idx < 8 * sizeof(name_id_t); bit_idx++) {
        name_id_t name_id = name_id_t(uint64_t(1) << bit_idx);
        if (!static_cast<uint64_t>(buffer_set & name_id)) continue;
        if (buf_idx > 0) ss << ", ";

        ss << to_string(name_id)
           << " - [ type: " << dnnl_dt2str(buffer_types[buf_idx])
           << " offset : "
           << expr_encoder_t::decode(exprs, buffer_off_expr[buf_idx]) << "]";
        buf_idx++;
    }
    ss << "}";
    return ss.str();
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
