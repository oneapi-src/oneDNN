/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include "cpu/x64/zen64/common/zen_post_ops_fallback.hpp"

#include <cstring>

#include "common/dnnl_thread.hpp"
#include "common/eltwise_pd.hpp"
#include "common/memory.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/reorder.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {

#define VDISPATCH_ZEN_POST_OPS_FALLBACK(cond, ...) \
    VCONDCHECK(primitive, create, dispatch, zen_post_ops_fallback, (cond), \
            status::unimplemented, __VA_ARGS__)

namespace {

using memory_ptr_t = std::unique_ptr<memory_t, memory_deleter_t>;

// First implementation in dispatch order; the op descriptor fixes all layouts.
status_t create_pd(const engine_t *engine, const op_desc_t *op_desc,
        const primitive_attr_t &attr, std::shared_ptr<primitive_desc_t> &pd) {
    primitive_desc_iterator_t it(engine, op_desc, &attr, nullptr);
    while (++it != it.end()) {
        pd = *it;
        if (pd) return status::success;
    }
    return status::unimplemented;
}

memory_ptr_t wrap_memory(
        engine_t *engine, const memory_desc_t *md, const void *ptr) {
    return memory_ptr_t(
            new memory_t(engine, md, use_runtime_ptr, const_cast<void *>(ptr)));
}

} // namespace

status_t post_ops_fallback_t::init(const engine_t *engine,
        const post_ops_t &post_ops, const memory_desc_t &dst_md,
        int start_index, const memory_desc_t *acc_md) {
    post_ops_.entry_ = post_ops.entry_;
    start_index_ = start_index;
    uses_acc_ = acc_md != nullptr;
    prelu_copy_size_ = 0;
    pds_.clear();
    primitives_.clear();

    // Entries update the kernel output in place: dst, or the accumulator.
    const memory_desc_t &out_md = uses_acc_ ? *acc_md : dst_md;
    const primitive_attr_t default_attr;
    for (int idx = start_index; idx < post_ops.len(); idx++) {
        const auto &e = post_ops.entry_[idx];
        const bool is_sum = e.is_sum(
                /*require_scale_one=*/false, /*require_zp_zero=*/false);
        std::shared_ptr<primitive_desc_t> pd;

        if (is_sum || e.is_inplace_binary()) {
            VDISPATCH_ZEN_POST_OPS_FALLBACK(uses_acc_,
                    "post-op reading the original dst needs an accumulator");
            VDISPATCH_ZEN_POST_OPS_FALLBACK(!is_sum || e.sum.zero_point == 0,
                    "sum post-op zero point must be 0");

            // acc = acc + scale * dst (sum, which reads dst in its sum.dt) or
            // acc = acc * dst (in-place binary), in place on acc (src0).
            memory_desc_t orig_dst_md = dst_md;
            if (is_sum && e.sum.dt != data_type::undef)
                orig_dst_md.data_type = e.sum.dt;
            binary_desc_t bd;
            bd.alg_kind = is_sum ? alg_kind::binary_add : alg_kind::binary_mul;
            bd.src_desc[0] = out_md;
            bd.src_desc[1] = orig_dst_md;
            bd.dst_desc = out_md;
            primitive_attr_t attr;
            if (is_sum && e.sum.scale != 1.f)
                CHECK(attr.scales_.set(DNNL_ARG_SRC_1, /*mask=*/0));
            CHECK(create_pd(engine, &bd, attr, pd));
        } else if (e.is_binary()) {
            const bool with_src2 = e.is_binary_with_ternary_op();
            VDISPATCH_ZEN_POST_OPS_FALLBACK(
                    !memory_desc_wrapper(e.binary.src1_desc).format_any()
                            && IMPLICATION(with_src2,
                                    !memory_desc_wrapper(e.binary.src2_desc)
                                             .format_any()),
                    VERBOSE_UNSUPPORTED_POSTOP);

            binary_desc_t bd;
            bd.alg_kind = e.binary.alg;
            bd.src_desc[0] = out_md;
            bd.src_desc[1] = e.binary.src1_desc;
            if (with_src2) bd.src_desc[2] = e.binary.src2_desc;
            bd.dst_desc = out_md;
            CHECK(create_pd(engine, &bd, default_attr, pd));
        } else if (e.is_eltwise()) {
            VDISPATCH_ZEN_POST_OPS_FALLBACK(
                    e.eltwise.scale == 1.f, "eltwise post-op scale must be 1");

            eltwise_desc_t ed;
            CHECK(eltwise_desc_init(&ed, prop_kind::forward_inference,
                    e.eltwise.alg, &out_md, &out_md, nullptr, nullptr,
                    e.eltwise.alpha, e.eltwise.beta));
            CHECK(create_pd(engine, &ed, default_attr, pd));
        } else if (e.is_prelu()) {
            // The user passes f32 weights with the dst dims reduced by the
            // mask, in the channels-last (axb) layout.
            const int ndims = dst_md.ndims;
            VDISPATCH_ZEN_POST_OPS_FALLBACK(utils::one_of(ndims, 1, 2, 3, 4, 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            dims_t wei_dims;
            utils::copy_dims_with_mask(wei_dims, dst_md.dims, ndims,
                    e.prelu.mask, /*fill_with_one=*/true);
            memory_desc_t wei_md;
            CHECK(memory_desc_init_by_tag(wei_md, ndims, wei_dims,
                    data_type::f32,
                    utils::pick(ndims - 1, format_tag::a, format_tag::ab,
                            format_tag::acb, format_tag::acdb,
                            format_tag::acdeb)));

            prelu_desc_t prd;
            prd.prop_kind = prop_kind::forward_inference;
            prd.src_desc = out_md;
            prd.weights_desc = wei_md;
            prd.dst_desc = out_md;
            CHECK(create_pd(engine, &prd, default_attr, pd));
            // The copy is read through out_md, so it runs from the base
            // pointer through the last element out_md addresses, offset0
            // included. size() would also count the stride gap after that
            // element, which can lie outside the caller's allocation.
            const memory_desc_wrapper out_d(out_md);
            dims_t last_pos;
            for (int d = 0; d < out_d.ndims(); d++)
                last_pos[d] = out_d.padded_dims()[d] - 1;
            const dim_t last_off
                    = out_d.off_v(last_pos, /*is_pos_padded=*/true);
            prelu_copy_size_ = static_cast<size_t>(last_off + 1)
                    * out_d.data_type_size();
        } else {
            VDISPATCH_ZEN_POST_OPS_FALLBACK(false, VERBOSE_UNSUPPORTED_POSTOP);
        }
        pds_.push_back(std::move(pd));
    }

    if (uses_acc_) {
        std::shared_ptr<primitive_desc_t> pd;
        CHECK(reorder_primitive_desc_create(pd, engine, acc_md, &dst_md));
        pds_.push_back(std::move(pd));
    }
    return status::success;
}

status_t post_ops_fallback_t::init_primitives(engine_t *engine) {
    primitives_.clear();
    primitives_.reserve(pds_.size());
    for (const auto &pd : pds_) {
        std::shared_ptr<primitive_t> p;
        CHECK(pd->create_primitive(p, engine));
        primitives_.push_back(std::move(p));
    }
    return status::success;
}

void post_ops_fallback_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    for (size_t i = 0; i < pds_.size(); i++)
        scratchpad.book(memory_tracking::names::key_nested_multiple + (int)i,
                pds_[i]->scratchpad_registry());
    if (prelu_copy_size_ > 0)
        scratchpad.book(memory_tracking::names::key_eltwise_src,
                prelu_copy_size_, /*data_size=*/1, /*alignment=*/64);
}

status_t post_ops_fallback_t::execute(
        const exec_ctx_t &ctx, void *kernel_dst) const {
    void *dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    if (kernel_dst == nullptr || (kernel_dst == dst) == uses_acc_)
        return status::runtime_error;

    const int n_post_ops = (int)primitives_.size() - (uses_acc_ ? 1 : 0);
    for (int i = 0; i < n_post_ops; i++) {
        const int idx = start_index_ + i;
        const auto &e = post_ops_.entry_[idx];
        const int arg = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx);
        if (e.is_sum(/*require_scale_one=*/false, /*require_zp_zero=*/false)) {
            CHECK(execute_binary(ctx, i, kernel_dst, dst, nullptr, kernel_dst,
                    &e.sum.scale));
        } else if (e.is_inplace_binary()) {
            CHECK(execute_binary(
                    ctx, i, kernel_dst, dst, nullptr, kernel_dst, nullptr));
        } else if (e.is_binary()) {
            const void *src1 = CTX_IN_MEM(const void *, arg | DNNL_ARG_SRC_1);
            const void *src2 = CTX_IN_MEM(const void *, arg | DNNL_ARG_SRC_2);
            CHECK(execute_binary(
                    ctx, i, kernel_dst, src1, src2, kernel_dst, nullptr));
        } else if (e.is_prelu()) {
            const void *weights
                    = CTX_IN_MEM(const void *, arg | DNNL_ARG_WEIGHTS);
            CHECK(execute_prelu(ctx, i, weights, kernel_dst));
        } else {
            CHECK(execute_eltwise(ctx, i, kernel_dst));
        }
    }
    if (uses_acc_) CHECK(execute_reorder(ctx, n_post_ops, kernel_dst, dst));
    return status::success;
}

status_t post_ops_fallback_t::execute_binary(const exec_ctx_t &ctx, int i,
        const void *src0, const void *src1, const void *src2, void *dst,
        const float *src1_scale) const {
    const primitive_desc_t *pd = primitives_[i]->pd().get();
    const bool with_src2
            = !memory_desc_wrapper(pd->arg_md(DNNL_ARG_SRC_2)).is_zero();
    const bool with_src1_scale
            = !pd->attr()->scales_.has_default_values(DNNL_ARG_SRC_1);
    if (src0 == nullptr || src1 == nullptr || dst == nullptr
            || (with_src2 && src2 == nullptr)
            || (with_src1_scale && src1_scale == nullptr))
        return status::invalid_arguments;

    engine_t *engine = ctx.stream()->engine();
    memory_ptr_t src0_mem
            = wrap_memory(engine, pd->arg_md(DNNL_ARG_SRC_0), src0);
    memory_ptr_t src1_mem
            = wrap_memory(engine, pd->arg_md(DNNL_ARG_SRC_1), src1);
    memory_ptr_t src2_mem = with_src2
            ? wrap_memory(engine, pd->arg_md(DNNL_ARG_SRC_2), src2)
            : nullptr;
    memory_ptr_t dst_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_DST), dst);

    exec_args_t args;
    args[DNNL_ARG_SRC_0] = {src0_mem.get(), true};
    args[DNNL_ARG_SRC_1] = {src1_mem.get(), true};
    if (with_src2) args[DNNL_ARG_SRC_2] = {src2_mem.get(), true};
    args[DNNL_ARG_DST] = {dst_mem.get(), false};

    memory_ptr_t scale_mem;
    if (with_src1_scale) {
        memory_desc_t scale_md;
        const dims_t scale_dims = {1};
        CHECK(memory_desc_init_by_tag(
                scale_md, 1, scale_dims, data_type::f32, format_tag::a));
        scale_mem = wrap_memory(engine, &scale_md, src1_scale);
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC_1] = {scale_mem.get(), true};
    }
    return execute_nested(ctx, i, std::move(args));
}

status_t post_ops_fallback_t::execute_eltwise(
        const exec_ctx_t &ctx, int i, void *data) const {
    const primitive_desc_t *pd = primitives_[i]->pd().get();
    engine_t *engine = ctx.stream()->engine();
    memory_ptr_t src_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_SRC), data);
    memory_ptr_t dst_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_DST), data);

    exec_args_t args;
    args[DNNL_ARG_SRC] = {src_mem.get(), true};
    args[DNNL_ARG_DST] = {dst_mem.get(), false};
    return execute_nested(ctx, i, std::move(args));
}

status_t post_ops_fallback_t::execute_prelu(
        const exec_ctx_t &ctx, int i, const void *weights, void *data) const {
    if (weights == nullptr) return status::invalid_arguments;

    // PReLU does not support in-place execution, so it reads a copy.
    auto *src = ctx.get_scratchpad_grantor().get<uint8_t>(
            memory_tracking::names::key_eltwise_src);
    if (src == nullptr) return status::out_of_memory;
    const auto *in = static_cast<const uint8_t *>(data);
    parallel(0, [&](int ithr, int nthr) {
        size_t start = 0, end = 0;
        balance211(prelu_copy_size_, nthr, ithr, start, end);
        if (end > start) std::memcpy(src + start, in + start, end - start);
    });

    const primitive_desc_t *pd = primitives_[i]->pd().get();
    engine_t *engine = ctx.stream()->engine();
    memory_ptr_t src_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_SRC), src);
    memory_ptr_t wei_mem
            = wrap_memory(engine, pd->arg_md(DNNL_ARG_WEIGHTS), weights);
    memory_ptr_t dst_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_DST), data);

    exec_args_t args;
    args[DNNL_ARG_SRC] = {src_mem.get(), true};
    args[DNNL_ARG_WEIGHTS] = {wei_mem.get(), true};
    args[DNNL_ARG_DST] = {dst_mem.get(), false};
    return execute_nested(ctx, i, std::move(args));
}

status_t post_ops_fallback_t::execute_reorder(
        const exec_ctx_t &ctx, int i, const void *src, void *dst) const {
    const primitive_desc_t *pd = primitives_[i]->pd().get();
    engine_t *engine = ctx.stream()->engine();
    memory_ptr_t src_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_FROM), src);
    memory_ptr_t dst_mem = wrap_memory(engine, pd->arg_md(DNNL_ARG_TO), dst);

    exec_args_t args;
    args[DNNL_ARG_FROM] = {src_mem.get(), true};
    args[DNNL_ARG_TO] = {dst_mem.get(), false};
    return execute_nested(ctx, i, std::move(args));
}

status_t post_ops_fallback_t::execute_nested(
        const exec_ctx_t &ctx, int i, exec_args_t &&args) const {
    const auto &p = primitives_[i];
    exec_ctx_t nested_ctx(ctx, std::move(args));
    nested_ctx.set_scratchpad_grantor(
            memory_tracking::create_nested_grantor(ctx.get_scratchpad_grantor(),
                    memory_tracking::names::key_nested_multiple + i,
                    p->pd()->scratchpad_registry()));
    return p->execute(nested_ctx);
}

} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
