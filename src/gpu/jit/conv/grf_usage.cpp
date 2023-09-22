/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gpu/jit/conv/grf_usage.hpp"

#include <sstream>

#include "gpu/jit/codegen/register_allocator.hpp"
#include "gpu/jit/conv/plan.hpp"
#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/ir/reorder.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

std::string to_string(grf_usage_label_t label) {
    switch (label) {
#define CASE(l) \
    case grf_usage_label_t::l: return #l;
        CASE(unknown)
        CASE(gmem_load)
        CASE(out_buf)
        CASE(reorder)
        CASE(reserved)
        CASE(reused_headers)
        CASE(slm_load)
        CASE(slm_store)
        CASE(tmp_vars)
        CASE(zero_points)
#undef CASE
        default: ir_error_not_expected();
    }
    return "";
}

std::ostream &operator<<(std::ostream &out, grf_usage_label_t label) {
    out << to_string(label);
    return out;
}

std::string grf_buf_usage_t::str() const {
    std::ostringstream oss;
    oss << "Buffers:";
    for (auto label : all_grf_usage_labels()) {
        int regs = total_regs(label);
        if (regs == 0) continue;
        oss << std::endl << " " << label << " (" << regs << "): ";
        bool is_first = true;
        for (auto &buf : sorted_bufs()) {
            if (get_label(buf) != label) continue;
            if (!is_first) oss << ", ";
            is_first = false;
            oss << buf << "[" << get_size(buf) << "]";
        }
    }
    return oss.str();
}

std::string grf_usage_t::str() const {
    std::vector<std::string> headers = {"Label", "Regs"};
    ir_utils::table_t table("GRF usage", headers);
    int total = 0;
    for (auto label : all_grf_usage_labels()) {
        int regs = regs_.at(label);
        if (regs == 0) continue;
        table << to_string(label) << regs << std::endl;
        total += regs;
    }
    table << "Total" << total << std::endl;
    std::ostringstream oss;
    oss << table << std::endl;
    oss << buf_usage_;
    return oss.str();
}

class ir_usage_analyzer_t : public ir_visitor_t {
public:
    ir_usage_analyzer_t(int grf_size)
        : grf_size_(grf_size), buf_usage_(grf_size) {}

    void analyze(const stmt_t &stmt, bool allow_errors = false) {
        visit(stmt);
        if (!is_invalid_) {
            if (peak_headers_ <= 1) {
                std::vector<expr_t> header_bufs;
                for (auto &buf : buf_usage_.bufs()) {
                    if (is_header(buf)) header_bufs.push_back(buf);
                }
                int peak_header_usage_ = 0;
                for (auto &buf : header_bufs) {
                    peak_header_usage_ = std::max(
                            peak_header_usage_, buf_usage_.get_size(buf));
                    buf_usage_.remove(buf);
                }
            }
        }
        if (!verify(allow_errors)) is_invalid_ = true;
    }

    void _visit(const alloc_t &obj) override {
        if (is_invalid_) return;
        int size = (obj.kind == alloc_kind_t::grf ? obj.size : 0);
        size = utils::rnd_up(size, grf_size_);
        mem_usage_guard_t alloc_guard(&alloc_usage_, &peak_alloc_usage_, size);
        mem_usage_guard_t guard(&grf_usage_, &peak_grf_usage_, size);
        if (size > 0) {
            buf_usage_.add(obj.buf, obj.size, grf_usage_label_t::unknown);
            mark_known_bufs(obj.buf);
        }
        mem_usage_guard_t header_guard;
        if (is_header(obj.buf))
            header_guard = mem_usage_guard_t(&headers_, &peak_headers_, 1);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const func_call_t &obj) override {
        if (is_invalid_) return;
        auto &func = obj.func;
        if (auto *reorder = func.as_ptr<reorder_t>()) {
            auto &src = get_base(reorder_t::arg_src_buf(obj));
            auto &dst = get_base(reorder_t::arg_dst_buf(obj));
            mark_bufs(*reorder, src, dst);
        } else if (auto *send = func.as_ptr<send_t>()) {
            if (!send_t::arg_header_buf(obj).is<var_t>()) {
                is_invalid_ = true;
                return;
            }
            auto &buf = get_base(send_t::arg_reg_buf(obj));
            auto &header = get_base(send_t::arg_header_buf(obj));
            mark_bufs(*send, buf, header);
        } else if (is_func_call<dpas_t>(obj)) {
            auto &dst = get_base(dpas_t::arg_dst(obj));
            auto &src1 = get_base(dpas_t::arg_src1(obj));
            auto &src2 = get_base(dpas_t::arg_src2(obj));
            mark_fma_bufs(dst, src1, src2);
        } else if (is_func_call<mad_t>(obj)) {
            auto &dst = get_base(mad_t::arg_dst(obj));
            auto &src1 = get_base(mad_t::arg_src1(obj));
            auto &src2 = get_base(mad_t::arg_src2(obj));
            mark_fma_bufs(dst, src1, src2);
        }
    }

    void _visit(const let_t &obj) override {
        if (is_invalid_) return;
        int size = (obj.value.is_empty() ? 0 : obj.var.type().size());
        size = utils::rnd_up(size, reg_allocator_t::granularity);
        mem_usage_guard_t guard(&grf_usage_, &peak_grf_usage_, size);
        ir_visitor_t::_visit(obj);
    }

    void _visit(const stmt_group_t &obj) override {
        if (is_invalid_) return;
        if (obj.label == stmt_label_t::c_store()) {
            // Do not analyze C store consumption for simplicity. Assume there
            // is enough space after releasing A/B and other buffers after the
            // main loop.
            return;
        }
        ir_visitor_t::_visit(obj);
    }

    void _visit(const store_t &obj) override {
        if (is_invalid_) return;
        auto loads = find_objects<load_t>(obj);
        for (auto &l : loads) {
            if (obj.buf.is_same(l.as<load_t>().buf)) {
                set_label(obj.buf, grf_usage_label_t::tmp_vars);
                break;
            }
        }
        ir_visitor_t::_visit(obj);
    }

    grf_usage_t get_grf_usage(int external_regs) const {
        if (is_invalid_) return grf_usage_t();
        grf_usage_t info(grf_size_);
        info.add(buf_usage_);
        info.add(grf_usage_label_t::reserved, external_regs);
        info.add(grf_usage_label_t::tmp_vars,
                utils::div_up(peak_grf_usage_ - peak_alloc_usage_, grf_size_));
        if (peak_headers_ <= 1) {
            info.add(grf_usage_label_t::reused_headers, peak_header_usage_);
        }
        return info;
    }

private:
    bool verify(bool allow_errors) const {
        if (is_invalid_) {
            if (!allow_errors)
                ir_error_not_expected() << "Can't collect GRF usage.";
            return false;
        }
        for (auto &buf : buf_usage_.bufs()) {
            if (buf_usage_.get_label(buf) != grf_usage_label_t::unknown)
                continue;
            if (!allow_errors)
                ir_error_not_expected() << "Buffer doesn't have label: " << buf;
            return false;
        }
        return true;
    }

    bool is_buffer(const expr_t &buf) const { return buf_usage_.has(buf); }

    bool is_header(const expr_t &buf) const {
        if (!is_buffer(buf)) return false;
        auto &name = buf.as<var_t>().name;
        return name.find("h_") == 0;
    }

    bool should_skip_if_set(const expr_t &buf, grf_usage_label_t label) const {
        if (is_known_buf(buf)) return true;
        switch (label) {
            case grf_usage_label_t::tmp_vars:
            case grf_usage_label_t::slm_store: return true;
            default: return false;
        }
    }

    void set_label(const expr_t &buf, grf_usage_label_t label) {
        if (is_invalid_) return;
        bool skip_if_set = should_skip_if_set(buf, label);
        auto buf_label = buf_usage_.get_label(buf);
        if (utils::one_of(buf_label, grf_usage_label_t::unknown, label)) {
            buf_usage_.set_label(buf, label);
        } else {
            if (skip_if_set) return;
            ir_error_not_expected()
                    << "Label already set. Buffer: " << buf
                    << ", old label: " << buf_label << ", new label: " << label;
        }
    }

    void mark_known_bufs(const expr_t &buf) {
        if (is_invalid_) return;
        ir_assert(is_buffer(buf));
        auto &name = buf.as<var_t>().name;
        if (name.find("x_reduce") == 0) {
            set_label(buf, grf_usage_label_t::out_buf);
        } else if (name.find("zp_") == 0 || name.find("src_zp") == 0) {
            set_label(buf, grf_usage_label_t::zero_points);
        }
    }

    bool is_known_buf(const expr_t &buf) const {
        ir_assert(is_buffer(buf));
        auto &name = buf.as<var_t>().name;
        if (name.find("zp_") == 0) return true;
        if (name.find("src_zp") == 0) return true;
        if (name.find("x_reduce")) return true;
        return false;
    }

    void mark_bufs(
            const reorder_t &reorder, const expr_t &src, const expr_t &dst) {
        if (is_invalid_) return;
        ir_assert(is_buffer(src));
        ir_assert(is_buffer(dst));
        set_label(dst, grf_usage_label_t::reorder);
    }

    void mark_bufs(
            const send_t &send, const expr_t &buf, const expr_t &header) {
        if (is_invalid_) return;
        if (!buf.is_empty()) ir_assert(is_buffer(buf));
        ir_assert(is_buffer(header));
        ir_assert(is_header(header));
        grf_usage_label_t label = grf_usage_label_t::unknown;
        if (buf.is_empty()) {
            label = grf_usage_label_t::gmem_load;
        } else if (buf_usage_.get_label(buf)
                == grf_usage_label_t::zero_points) {
            label = grf_usage_label_t::zero_points;
        } else if (send.is_slm()) {
            label = (send.is_load() ? grf_usage_label_t::slm_load
                                    : grf_usage_label_t::slm_store);
        } else {
            if (!send.is_load() && !send.is_load_2d()) {
                is_invalid_ = true;
                return;
            }
            label = grf_usage_label_t::gmem_load;
        }
        if (!buf.is_empty()) set_label(buf, label);
        set_label(header, label);
    }

    void mark_fma_bufs(
            const expr_t &dst, const expr_t &src1, const expr_t &src2) {
        if (is_invalid_) return;
        ir_assert(is_buffer(dst));
        ir_assert(is_buffer(src1));
        ir_assert(is_buffer(src2));
        set_label(dst, grf_usage_label_t::out_buf);
    }

    int grf_size_;
    bool is_invalid_ = false;
    grf_buf_usage_t buf_usage_;

    int grf_usage_ = 0;
    int alloc_usage_ = 0;

    int peak_grf_usage_ = 0;
    int peak_alloc_usage_ = 0;

    int headers_ = 0;
    int peak_headers_ = 0;
    int peak_header_usage_ = 0; // Unset when headers are not reused.
};

grf_usage_t get_grf_usage(const stmt_t &body, int grf_size) {
    ir_usage_analyzer_t analyzer(grf_size);
    analyzer.visit(body);
    return analyzer.get_grf_usage(0);
}

void compare(const grf_usage_t &est_usage, const grf_usage_t &ir_usage,
        const ir_usage_analyzer_t &analyzer) {
    std::vector<std::string> headers
            = {"Label", "Estimated regs", "IR regs", "Status"};
    ir_utils::table_t table("GRF usage", headers);
    int est_total = 0;
    int ir_total = 0;
    for (auto label : all_grf_usage_labels()) {
        int est_regs = est_usage.get(label);
        int ir_regs = ir_usage.get(label);
        table << to_string(label) << est_regs << ir_regs;
        table << (ir_regs > est_regs ? "FAIL" : "");
        table << std::endl;
        est_total += est_regs;
        ir_total += ir_regs;
    }
    table << "Total" << est_total << ir_total;
    table << (ir_total > est_total ? "FAIL" : "");
    table << std::endl;
    ir_trace() << table << std::endl;
    ir_trace() << ir_usage.buf_usage() << std::endl;
}

void verify_grf_usage(
        const conv_config_t &cfg, const stmt_t &body, int external_usage) {
    ir_usage_analyzer_t analyzer(cfg.grf_size());
    analyzer.analyze(body);

    auto ir_info = analyzer.get_grf_usage(external_usage);
    auto est_info = cfg.plan().grf_usage();
    compare(est_info, ir_info, analyzer);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
