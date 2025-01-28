/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
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

#include "oneapi/dnnl/dnnl_ukernel.h"

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/verbose.hpp"

#include "cpu/ref_io_helper.hpp"

#include "cpu/x64/amx_tile_configure.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_utils.hpp"

#include "cpu/x64/brgemm/capi/brgemm_api.hpp"

#ifdef DNNL_EXPERIMENTAL_UKERNEL

using namespace dnnl::impl;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::status;
using namespace dnnl::impl::cpu::x64;

using brgemm_t = dnnl_brgemm;
using transform_t = dnnl_transform;

#define VCHECK_BRGEMM(cond, msg, ...) \
    VCONDCHECK(ukernel, create, check, brgemm, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_BRGEMM_STATUS(status, cond, msg, ...) \
    VCONDCHECK(ukernel, create, check, brgemm, (cond), (status), msg, \
            ##__VA_ARGS__)

status_t attr_params_t::set_post_ops_args(const void **post_ops_args) {
    post_ops_args_ = post_ops_args;
    return status::success;
}

status_t attr_params_t::set_scales(const void *scales, int arg) {
    switch (arg) {
        case DNNL_ARG_SRC: a_scales_ = scales; break;
        case DNNL_ARG_WEIGHTS: b_scales_ = scales; break;
        case DNNL_ARG_DST: d_scales_ = scales; break;
        default: assert(!"unsupported arg");
    }
    return status::success;
}

const void *attr_params_t::get_scales(int arg) const {
    switch (arg) {
        case DNNL_ARG_SRC: return a_scales_;
        case DNNL_ARG_WEIGHTS: return b_scales_;
        case DNNL_ARG_DST: return d_scales_;
        default: assert(!"unsupported arg");
    }
    return nullptr;
}

dnnl_brgemm::~dnnl_brgemm() {
    brgemm_kernel_destroy(brgemm_kernel_);
}

// Typical usage is either `1.f` to append to previous result, or `0.f` to write
// C from scratch.
status_t brgemm_t::set_add_C(int add_C) {
    if (add_C == 0)
        beta_ = 0.f;
    else if (add_C == 1)
        beta_ = 1.f;
    return status::success;
}

status_t brgemm_t::set_post_ops(
        dim_t ldd, data_type_t d_dt, const post_ops_t *post_ops) {
    ldd_ = ldd;
    d_dt_ = d_dt;
    CHECK(attr_.set_post_ops(*post_ops));
    return status::success;
}

status_t brgemm_t::set_scales(int mask, int arg) {
    if (mask < 0) return status::invalid_arguments;
    CHECK(attr_.scales_.set(arg, mask));
    return status::success;
}

status_t brgemm_t::finalize() {
    brgemm_batch_kind_t batch_kind = brgemm_batch_kind_t::brgemm_offs;

    auto status = brgemm_desc_init(&brgemm_desc_, cpu_isa_t::isa_undef,
            batch_kind, a_dt_, b_dt_, /* transA = */ false,
            /* trans_B = */ false, brgemm_row_major, /* alpha = */ 1.f, beta_,
            lda_, ldb_, ldc_, M_, N_, K_,
            /* strides = */ nullptr);
    if (status != status::success) {
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_init failed");
    }

    memory_desc_t D_md;
    dims_t dims {M_, N_};
    dims_t strides {ldc_, 1};
    status = memory_desc_init_by_strides(
            D_md, /* ndims = */ 2, dims, d_dt_, strides);
    if (status != status::success) {
        VCHECK_BRGEMM_STATUS(status, false, "D_md creation failed");
    }

    // This one is not used anywhere in implementation, but, maybe, could be
    // used in the future in fpmath mode if users would like to override the
    // default accumulation data type.
    UNUSED(c_dt_);

    status = brgemm_desc_set_postops(
            &brgemm_desc_, &attr_, &D_md, ldd_, data_type::undef);
    if (status != status::success) {
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_set_postops failed");
    }

    brgemm_attr_t brgemm_attr;
    brgemm_attr.max_bs = batch_size_;
    if (mayiuse(avx512_core_amx)) {
        brgemm_attr.use_uker = true;
        brgemm_attr.use_interleave_stores = true;
        brgemm_attr.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;
    }

    status = brgemm_desc_set_attr(&brgemm_desc_, brgemm_attr);
    if (status != status::success) {
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_set_attr failed");
    }

    status = brgemm_desc_finalize(&brgemm_desc_);
    if (status != status::success) {
        VCHECK_BRGEMM_STATUS(status, false, "brgemm_desc_finalize failed");
    }

    // Note: API can't take a compensation buffer externally. Users must add
    // compensation on their own as a binary post-op.
    brgemm_desc_.req_s8s8_compensation = false;

    return status::success;
}

status_t brgemm_t::get_B_pack_type(
        pack_type_t *pack_type, data_type_t dt_a, data_type_t dt_b) {
    // Use a descriptor to obtain the ISA to have compatible values when the
    // user creates an object.
    brgemm_desc_t brg {};
    brg.dt_a = dt_a;
    brg.dt_b = dt_b;
    init_kernel_datatype(&brg, dt_a, dt_b);
    brgemm_utils::set_isa_impl(&brg);
    if (brg.isa_impl == cpu_isa_t::isa_undef) {
        VCHECK_BRGEMM_STATUS(
                status::invalid_arguments, false, "get_B_pack_type failed");
    }
    const bool has_vnni_layout = brgemm_desc_t::is_b_data_layout_vnni(
            dt_a, dt_b, /* brgattr.b_is_vnni = */ false, brg.isa_impl);
    *pack_type = has_vnni_layout ? pack_type::pack32 : pack_type::no_trans;
    return status::success;
}

size_t brgemm_t::get_scratchpad_size() const {
    return brgemm_desc_.get_wsp_buffer_size();
}

bool brgemm_t::is_execute_postops_valid() const {
    return brgemm_desc_.are_post_ops_applicable();
}

status_t brgemm_t::set_hw_context() const {
    char palette[AMX_PALETTE_SIZE] = {};
    auto status = brgemm_init_tiles(brgemm_desc_, palette);
    // If status isn't successful, it means tiles configuration is not required.
    if (status == status::success) {
        status = amx_tile_lazy_configure(palette);
        VCHECK_BRGEMM_STATUS(
                status, status == status::success, "amx_tile_configure failed");
    }
    return status::success;
}

status_t brgemm_t::generate() {
    // Re-generation won't take any effect.
    if (brgemm_kernel_ != nullptr) return status::success;

    auto status = brgemm_kernel_create(&brgemm_kernel_, brgemm_desc_);
    VCHECK_BRGEMM_STATUS(
            status, status == status::success, "brgemm_kernel_create failed");

    // Generate a verbose info string at the point where configuration is done.
    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        create_verbose_info();
    }
    return status::success;
}

status_t brgemm_t::execute(const void *A_ptr, const void *B_ptr,
        const dim_t *A_B_offsets, void *C_ptr, void *scratchpad_ptr) const {
    const auto batch_size = brgemm_desc_.brgattr.max_bs;
    std::vector<brgemm_batch_element_t> v_batch_element(batch_size);
    for (int i = 0; i < batch_size; i++) {
        v_batch_element[i].offset.A = A_B_offsets[2 * i];
        v_batch_element[i].offset.B = A_B_offsets[2 * i + 1];
    }

    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        double start_ms = get_msec();
        brgemm_kernel_execute(brgemm_kernel_, batch_size, A_ptr, B_ptr,
                v_batch_element.data(), C_ptr, scratchpad_ptr,
                /* dynamic_values = */ nullptr);
        double duration_ms = get_msec() - start_ms;

        std::stringstream ss;
        ss << "cpu,brgemm,,undef," << verbose_info_;
        VPROF(start_ms, ukernel, exec, VERBOSE_profile, ss.str().c_str(),
                duration_ms);
    } else {
        brgemm_kernel_execute(brgemm_kernel_, batch_size, A_ptr, B_ptr,
                v_batch_element.data(), C_ptr, scratchpad_ptr,
                /* dynamic_values = */ nullptr);
    }
    return status::success;
}

status_t brgemm_t::execute(const void *A_ptr, const void *B_ptr,
        const dim_t *A_B_offsets, const void *C_ptr, void *D_ptr,
        void *scratchpad_ptr, const attr_params_t *attr_params) const {
    if (attr_params == nullptr) return status::invalid_arguments;

    if (!brgemm_desc_.are_post_ops_applicable()) {
        if (C_ptr == D_ptr) {
            return execute(A_ptr, B_ptr, A_B_offsets, const_cast<void *>(C_ptr),
                    scratchpad_ptr);
        } else {
            VCHECK_BRGEMM_STATUS(status::runtime_error, false,
                    "the kernel won't return correct results with this "
                    "execute_with_postops call.");
        }
    }

    const auto batch_size = brgemm_desc_.brgattr.max_bs;
    std::vector<brgemm_batch_element_t> v_batch_element(batch_size);
    for (int i = 0; i < batch_size; i++) {
        v_batch_element[i].offset.A = A_B_offsets[2 * i];
        v_batch_element[i].offset.B = A_B_offsets[2 * i + 1];
    }

    brgemm_post_ops_data_t post_ops_data;
    // Note: this member is used to compute an offset from the base DST address.
    // Thus, it's not a C buffer that should be passed, but D buffer.
    post_ops_data.data_C_ptr_ = reinterpret_cast<const char *>(D_ptr);
    // This member expects a pointer to a vector of pointers to binary_po args.
    // It's exactly what `attr_params` stores when gets a pointer from the user.
    post_ops_data.binary_post_ops_rhs = attr_params->get_post_ops_args();

    // Scales (quantization case, happens after accumulation). Require manual
    // combining when both are present, and extending to full simd broadcast,
    // when single values are provided.
    // Note: this piece is pretty close to what `precompute_scales` does.
    // TODO: switch to `precompute_scales` directly.
    alignas(64) float scales_buf[16] = {0};
    // TODO: delegate extra memory to scratchpad?
    std::vector<float> wei_scales_v(N_);

    const bool has_src_scales = !attr_.scales_.has_default_values(DNNL_ARG_SRC);
    const bool has_wei_scales
            = !attr_.scales_.has_default_values(DNNL_ARG_WEIGHTS);

    // Save src scale value to re-use it.
    float src_scale_val = 1.f;
    if (has_src_scales) {
        const void *src_scales_ptr = attr_params->get_scales(DNNL_ARG_SRC);
        if (src_scales_ptr == nullptr) return status::invalid_arguments;

        src_scale_val
                = cpu::io::load_float_value(data_type::f32, src_scales_ptr, 0);
    }
    if (has_wei_scales) {
        // Handle weights entirely here to avoid duplicating the logic.

        const void *wei_scales_ptr = attr_params->get_scales(DNNL_ARG_WEIGHTS);
        if (wei_scales_ptr == nullptr) return status::invalid_arguments;

        int wei_mask = attr_.scales_.get_mask(DNNL_ARG_WEIGHTS);
        if (wei_mask > 0) {
            for (dim_t i = 0; i < N_; i++) {
                const float wei_scale_val = cpu::io::load_float_value(
                        data_type::f32, wei_scales_ptr, i);
                wei_scales_v[i] = wei_scale_val * src_scale_val;
            }
            post_ops_data.scales = wei_scales_v.data();
        } else {
            const float s = cpu::io::load_float_value(
                    data_type::f32, wei_scales_ptr, 0);
            utils::array_set(scales_buf, s * src_scale_val, 16);
            post_ops_data.scales = scales_buf;
        }
    } else if (has_src_scales) {
        utils::array_set(scales_buf, src_scale_val, 16);
        post_ops_data.scales = scales_buf;
    }

    // Destination scales. Require manual extending to full simd broadcast.
    alignas(64) float dst_scales_buf[16] = {0};
    if (!attr_.scales_.has_default_values(DNNL_ARG_DST)) {
        const void *dst_scales_ptr = attr_params->get_scales(DNNL_ARG_DST);
        if (dst_scales_ptr == nullptr) return status::invalid_arguments;

        const float s
                = cpu::io::load_float_value(data_type::f32, dst_scales_ptr, 0);
        utils::array_set(dst_scales_buf, 1.f / s, 16);
        post_ops_data.dst_scales = dst_scales_buf;
    }

    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        double start_ms = get_msec();
        brgemm_kernel_execute_postops(brgemm_kernel_, batch_size, A_ptr, B_ptr,
                v_batch_element.data(), const_cast<void *>(C_ptr), D_ptr,
                post_ops_data, scratchpad_ptr,
                /* dynamic_values = */ nullptr);
        double duration_ms = get_msec() - start_ms;

        std::stringstream ss;
        ss << "cpu,brgemm,,undef," << verbose_info_;
        VPROF(start_ms, ukernel, exec, VERBOSE_profile, ss.str().c_str(),
                duration_ms);
    } else {
        brgemm_kernel_execute_postops(brgemm_kernel_, batch_size, A_ptr, B_ptr,
                v_batch_element.data(), const_cast<void *>(C_ptr), D_ptr,
                post_ops_data, scratchpad_ptr,
                /* dynamic_values = */ nullptr);
    }
    return status::success;
}

status_t brgemm_t::create_verbose_info() {
#if defined(DISABLE_VERBOSE)
    return status::success;
#endif

    const auto &d = brgemm_desc_;
    std::stringstream ss;

    memory_desc_t src_md;
    const dims_t src_dims = {M_, K_};
    const dims_t src_strides = {lda_, 1};
    CHECK(memory_desc_init_by_strides(src_md, 2, src_dims, a_dt_, src_strides));

    memory_desc_t wei_md;
    const dims_t wei_dims = {K_, N_};
    const dims_t wei_strides = {ldb_, 1};
    CHECK(memory_desc_init_by_strides(wei_md, 2, wei_dims, b_dt_, wei_strides));

    memory_desc_t dst_md;
    const dims_t dst_dims = {M_, N_};
    const dims_t dst_strides = {ldd_, 1};
    CHECK(memory_desc_init_by_strides(dst_md, 2, dst_dims, d_dt_, dst_strides));

    ss << md2fmt_str("src", &src_md, format_kind::undef) << " ";
    ss << md2fmt_str("wei", &wei_md, format_kind::undef) << " ";
    ss << md2fmt_str("dst", &dst_md, format_kind::undef);
    ss << "," << attr2str(&attr_) << ",";
    ss << "bs:" << d.brgattr.max_bs << " beta:" << beta_;
    ss << "," << md2dim_str(&src_md) << ":" << md2dim_str(&wei_md);

    verbose_info_ = ss.str();
    return status::success;
}

dnnl_transform::dnnl_transform(dim_t K, dim_t N, pack_type_t in_pack_type,
        dim_t in_ld, dim_t out_ld, data_type_t in_dt, data_type_t out_dt)
    : K_(K)
    , N_(N)
    , in_ld_(in_ld)
    , out_ld_(out_ld)
    , in_dt_(in_dt)
    , out_dt_(out_dt) {
    // Check for a valid in_ld depending on a pack type.
    assert(in_pack_type == pack_type::no_trans
                    ? IMPLICATION(K_ > 1, in_ld_ >= N_)
                    : in_ld_ >= K_);
    // Only special N_blk sizes are supported by matmul copy routines. Rest
    // will crash.
    assert(utils::one_of(out_ld_, 16, 32, 48, 64));

    const auto in_tag = in_pack_type == pack_type::trans ? format_tag::ba
                                                         : format_tag::ab;
    auto status = matmul::init_conf(bmc_, /* batch = */ 1, /* M = */ 0, K_, N_,
            in_ld_, out_ld_, in_dt_, out_dt_, in_tag);
    assert(status == status::success);
    if (status != status::success) return;

    if (in_pack_type == pack_type::trans) {
        strides_[0] = 1;
        strides_[1] = in_ld_;
    } else if (in_pack_type == pack_type::no_trans) {
        strides_[0] = in_ld_;
        strides_[1] = 1;
    } else {
        assert(!"Unsupported pack type");
    }
}

status_t transform_t::generate() {
    // Re-generation won't take any effect.
    if (pack_B_kernel_ != nullptr) return status::success;

    CHECK(matmul::create_brgemm_matmul_copy_b(pack_B_kernel_, &bmc_));

    // Generate a verbose info string at the point where configuration is done.
    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        CHECK(create_verbose_info());
    }
    return status::success;
}

status_t transform_t::execute(const void *src, void *dst) const {
    double start_ms = 0;
    if (get_verbose(verbose_t::exec_profile, component_t::ukernel))
        start_ms = get_msec();

    const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(src);
    uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst);

    const auto &kernel_conf = bmc_;
    const dim_t n_blks = utils::div_up(kernel_conf.N, kernel_conf.N_blk);
    const dim_t k_blks = utils::div_up(kernel_conf.K, kernel_conf.K_blk);
    const auto blk_size = kernel_conf.K_blk * kernel_conf.N_blk;

    const auto i_dt_sz = kernel_conf.b_dt_sz;
    const auto o_dt_sz = kernel_conf.a_dt_sz;

    for (dim_t n_blk_idx = 0; n_blk_idx < n_blks; n_blk_idx++) {
        const auto n = n_blk_idx * kernel_conf.N_blk;
        const bool is_N_tail = (kernel_conf.N - n) < kernel_conf.N_blk;
        auto ker_exec_ctx = matmul::jit_brgemm_matmul_copy_b_t::ctx_t();
        ker_exec_ctx.current_N_blk
                = is_N_tail ? kernel_conf.N_tail : kernel_conf.N_blk;

        int k_blk_idx = 0;
        for (; k_blk_idx < kernel_conf.K / kernel_conf.K_blk; k_blk_idx++) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            const auto src_offset
                    = i_dt_sz * (k * strides_[0] + n * strides_[1]);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_blk;
            (*pack_B_kernel_)(&ker_exec_ctx);
        }
        if (kernel_conf.K_tail > 0) {
            const auto k = k_blk_idx * kernel_conf.K_blk;
            const auto src_offset
                    = i_dt_sz * (k * strides_[0] + n * strides_[1]);
            const auto dst_offset
                    = o_dt_sz * (k_blk_idx * blk_size + n_blk_idx * k_blks);
            ker_exec_ctx.src = &src_ptr[src_offset];
            ker_exec_ctx.tr_src = &dst_ptr[dst_offset];
            ker_exec_ctx.current_K_start = k;
            ker_exec_ctx.current_K_iters = kernel_conf.K_tail;
            (*pack_B_kernel_)(&ker_exec_ctx);
        }
    }

    if (get_verbose(verbose_t::exec_profile, component_t::ukernel)) {
        double duration_ms = get_msec() - start_ms;

        std::stringstream ss;
        ss << "cpu,transform,pack_B,undef," << verbose_info_;
        VPROF(start_ms, ukernel, exec, VERBOSE_profile, ss.str().c_str(),
                duration_ms);
    }
    return status::success;
}

status_t transform_t::create_verbose_info() {
#if defined(DISABLE_VERBOSE)
    return status::success;
#endif

    std::stringstream ss;

    memory_desc_t src_md;
    const dims_t dims = {K_, N_};
    CHECK(memory_desc_init_by_strides(src_md, 2, dims, in_dt_, strides_));

    memory_desc_t dst_md;
    const dims_t dst_strides = {out_ld_, 1};
    CHECK(memory_desc_init_by_strides(dst_md, 2, dims, out_dt_, dst_strides));

    ss << md2fmt_str("src", &src_md, format_kind::undef) << " ";
    ss << md2fmt_str("dst", &dst_md, format_kind::undef);
    ss << ",,," << md2dim_str(&src_md);

    verbose_info_ = ss.str();
    return status::success;
}

////////////////
// Public API //
////////////////

/////////////////////////
// Attribute arguments //
/////////////////////////

status_t dnnl_ukernel_attr_params_create(attr_params_t **attr_params) {
    *attr_params = new attr_params_t();
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_post_ops_args(
        attr_params_t *attr_params, const void **post_ops_args) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_post_ops_args(post_ops_args));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_A_scales(
        attr_params_t *attr_params, const void *a_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(a_scales, DNNL_ARG_SRC));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_B_scales(
        attr_params_t *attr_params, const void *b_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(b_scales, DNNL_ARG_WEIGHTS));
    return status::success;
}

status_t dnnl_ukernel_attr_params_set_D_scales(
        attr_params_t *attr_params, const void *d_scales) {
    if (attr_params == nullptr) return status::invalid_arguments;

    CHECK(attr_params->set_scales(d_scales, DNNL_ARG_DST));
    return status::success;
}

status_t dnnl_ukernel_attr_params_destroy(attr_params_t *attr_params) {
    delete attr_params;
    return status::success;
}

////////////
// BRGeMM //
////////////

status_t dnnl_brgemm_create(brgemm_t **brgemm, dim_t M, dim_t N, dim_t K,
        dim_t batch_size, dim_t lda, dim_t ldb, dim_t ldc, data_type_t a_dt,
        data_type_t b_dt, data_type_t c_dt) {
    if (batch_size <= 0) {
        VCHECK_BRGEMM_STATUS(
                status::invalid_arguments, false, "batch size is non-positive");
    }

    *brgemm = new brgemm_t(
            M, N, K, batch_size, lda, ldb, ldc, a_dt, b_dt, c_dt);
    return status::success;
}

status_t dnnl_brgemm_set_add_C(brgemm_t *brgemm, int add_C) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_add_C(add_C));
    return status::success;
}

status_t dnnl_brgemm_set_post_ops(brgemm_t *brgemm, dim_t ldd, data_type_t d_dt,
        const post_ops_t *post_ops) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_post_ops(ldd, d_dt, post_ops));
    return status::success;
}

status_t dnnl_brgemm_set_A_scales(brgemm_t *brgemm, int a_scale_mask) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_scales(a_scale_mask, DNNL_ARG_SRC));
    return status::success;
}

status_t dnnl_brgemm_set_B_scales(brgemm_t *brgemm, int b_scale_mask) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_scales(b_scale_mask, DNNL_ARG_WEIGHTS));
    return status::success;
}

status_t dnnl_brgemm_set_D_scales(brgemm_t *brgemm, int d_scale_mask) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_scales(d_scale_mask, DNNL_ARG_DST));
    return status::success;
}

status_t dnnl_brgemm_finalize(brgemm_t *brgemm) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->finalize());
    return status::success;
}

status_t dnnl_brgemm_get_B_pack_type(
        dnnl_pack_type_t *pack_type, data_type_t dt_a, data_type_t dt_b) {
    if (pack_type) { return brgemm_t::get_B_pack_type(pack_type, dt_a, dt_b); }
    return status::success;
}

status_t dnnl_brgemm_get_scratchpad_size(const brgemm_t *brgemm, size_t *size) {
    if (brgemm == nullptr) return invalid_arguments;

    if (size) *size = brgemm->get_scratchpad_size();
    return status::success;
}

status_t dnnl_brgemm_is_execute_postops_valid(
        const brgemm_t *brgemm, int *valid) {
    if (brgemm == nullptr) return invalid_arguments;

    if (valid) *valid = static_cast<int>(brgemm->is_execute_postops_valid());
    return status::success;
}

status_t dnnl_brgemm_set_hw_context(const brgemm_t *brgemm) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->set_hw_context());
    return status::success;
}

status_t dnnl_brgemm_release_hw_context() {
    if (mayiuse(avx512_core_amx)) {
        VCHECK_BRGEMM(amx_tile_release() == status::success,
                "amx_tile_release failed");
    }

    return status::success;
}

status_t dnnl_brgemm_generate(brgemm_t *brgemm) {
    if (brgemm == nullptr) return invalid_arguments;

    CHECK(brgemm->generate());
    return status::success;
}

status_t dnnl_brgemm_execute(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, void *C_ptr,
        void *scratchpad_ptr) {
    CHECK(brgemm->execute(A_ptr, B_ptr, A_B_offsets, C_ptr, scratchpad_ptr));
    return status::success;
}

status_t dnnl_brgemm_execute_postops(const brgemm_t *brgemm, const void *A_ptr,
        const void *B_ptr, const dim_t *A_B_offsets, const void *C_ptr,
        void *D_ptr, void *scratchpad_ptr, const attr_params_t *attr_params) {
    CHECK(brgemm->execute(A_ptr, B_ptr, A_B_offsets, C_ptr, D_ptr,
            scratchpad_ptr, attr_params));
    return status::success;
}

status_t dnnl_brgemm_destroy(brgemm_t *brgemm) {
    delete brgemm;
    return status::success;
}

///////////////
// Transform //
///////////////

status_t dnnl_transform_create(transform_t **transform, dim_t K, dim_t N,
        pack_type_t in_pack_type, dim_t in_ld, dim_t out_ld, data_type_t in_dt,
        data_type_t out_dt) {
    if (transform == nullptr) return status::invalid_arguments;
    VCHECK_BRGEMM(utils::one_of(out_ld, 16, 32, 48, 64),
            "Transform routine supports only \'out_ld\' of 16, 32, 48, or 64.");

    *transform
            = new transform_t(K, N, in_pack_type, in_ld, out_ld, in_dt, out_dt);
    return status::success;
}

status_t dnnl_transform_generate(transform_t *transform) {
    if (transform == nullptr) return status::invalid_arguments;

    CHECK(transform->generate());
    return status::success;
}

status_t dnnl_transform_execute(
        const transform_t *transform, const void *in_ptr, void *out_ptr) {
    if (utils::any_null(transform, in_ptr, out_ptr))
        return status::invalid_arguments;

    CHECK(transform->execute(in_ptr, out_ptr));
    return status::success;
}

status_t dnnl_transform_destroy(transform_t *transform) {
    delete transform;
    return status::success;
}

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
