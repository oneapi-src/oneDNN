/*******************************************************************************
* Copyright 2020-2025 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_X64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/primitive_attr.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// The type defines organization of batch of matrices
typedef enum {
    // Undefined brgemm batch kind
    brgemm_batch_kind_undef = 0,
    // A and B arrays of pointers
    brgemm_addr = 1,
    // Base address and array of offsets from base address.
    brgemm_offs = 2,
    // Base address and fixed stride between matrices.
    brgemm_strd = 3,
    // Base address and static array of fixed offsets.
    brgemm_static_offs = 4,
} brgemm_batch_kind_t;

// The type defines the storage format of matrix
typedef enum {
    brgemm_layout_undef = 0,
    brgemm_col_major = 1,
    brgemm_row_major = 2,
} brgemm_layout_t;

typedef enum {
    none = 0,
    per_tensor = 1,
    per_m = 2,
    per_n = 3,
    per_k = 4,
} brgemm_broadcast_t;

struct brgemm_strides_t {
    // Stride between A matrices
    dim_t stride_a;
    // Stride between B matrices
    dim_t stride_b;
};

typedef enum {
    brgemm_lo_default = 0,
    brgemm_lo_bl_1load,
    brgemm_lo_bl_1bcst,
} brgemm_kernel_loop_order_t;

typedef enum {
    brgemm_prf_default = 1,
    brgemm_prf0,
    brgemm_prf1,
    brgemm_prf2,
    brgemm_prfNTA,
} brgemm_kernel_prefetching_t;

typedef enum {
    brgemm_innermost_undef = 0,
    brgemm_bd_loop_innermost,
    brgemm_ld_loop_innermost,
} brgemm_kernel_innermost_loop_t;

typedef enum {
    brgemm_hint_nt_undef = 0,
    brgemm_hint_nt_false,
    brgemm_hint_nt_true,
} brgemm_kernel_hint_nt_t;

struct brgemm_prf_t {
    int dist0 {-1};
    int dist1 {-1};
    int dist2 {-1};
    int distNTA {-1};
};

struct brgemm_batch_element_t {
    brgemm_batch_element_t() {
        ptr.A = ptr.B = nullptr;
        vvpad.top = vvpad.bottom = 0;
        has_s8s8_comp_batch_pad = 0;
    }
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
    struct {
        dim_t top;
        dim_t bottom;
    } vvpad; // w.r.t. M dimension

    // Used to calculate compensation when batch padding is present.
    // Note: batch_pad represent the overlap between weights and the height
    // dimension w.r.t. convolution dimensions.
    dim_t has_s8s8_comp_batch_pad;
};

struct DNNL_API brgemm_attr_t {
    brgemm_attr_t();
    // if unrolled kernel is used (use_uker == true)
    // then "max_bs" is the the only batch size that can be used on kernel call
    // else "max_bs" is the maximum batch size that can be used
    int max_bs;
    int max_top_vpad, max_bottom_vpad;
    int max_top_bpad, max_bottom_bpad;
    dim_t hint_expected_A_size, hint_expected_B_size, hint_expected_C_size;
    brgemm_kernel_innermost_loop_t hint_innermost_loop
            = brgemm_ld_loop_innermost;
    brgemm_kernel_loop_order_t hint_loop_order;
    brgemm_kernel_prefetching_t hint_prefetching
            = brgemm_kernel_prefetching_t::brgemm_prf_default;
    brgemm_prf_t hint_prfA, hint_prfB, hint_prfC;

    // This parameter determines how we will read the tail by K dimension from
    // matrix A. For AMX if the parameter is true then the brgemm will first
    // copy the data to the intermediate buffer and only then use the tileload.
    // For non-AMX the A data are loaded byte by byte if flag is set
    bool wary_A_k_tail_read {false};
    // For AMX the K dimension given to the brgemm is required to be divisible
    // by vnni granularity. In addition blocking by K dimension may not be
    // optimal if K greater than tile size and divisible by it.
    // The parameter 'extendable_k' enables the brgemm to use the optimal K
    // block size assuming that the following requirements for the matrix B are
    // fulfilled:
    //  - It is ​​properly blocked (64 bytes block by K dimension).
    //  - The dimension K is padded by zeros.
    // For K tail handling in this case the brgemm behavior is determined by the
    // 'wary_A_k_tail_read' parameter.
    bool extendable_k {false};
    bool generate_skip_accumulation;
    // Value of bd_mask_level specifies how bd_mask is used in brgemm kernel
    // 0 – bd_mask is not used
    // 1 – bd_mask is used on storing stage only
    // 2 – bd_mask used both on reading and storing stages
    int bd_mask_level;
    // use_uker is a boolean value that determines whether to use the unrolled
    // kernel or not
    bool use_uker;
    // use_interleave_stores is a value that determines whether to use the
    // interleave stores or not
    bool use_interleave_stores;
    impl::fpmath_mode_t fpmath_mode = fpmath_mode::strict;
    bool b_is_vnni {false};
    // Second level leading dimension describing distance between 16-line
    // blocks in case of blocked layout. Used to calculate address of next
    // bd block. By default are equal to regular leading dimension parameters
    // specified on brgemm creation.
    // Supported by brgemm unrolled kernel for now.
    int LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    // If "true" then batchsize is allowed to change on each kernel call
    // and there is no unrolling by batchsize in kernel
    bool var_bs {false};
    bool postops_only {false};
    // Hint for bs_group value in brgemm_desc_t
    int hint_bs_group {0};

    int hint_bd_block {0};
    int hint_ld_block {0};
    int hint_bd_block2 {0};
    int hint_ld_block2 {0};
    bool hint_ununroll_bd_loop {false};

    brgemm_kernel_hint_nt_t hint_load_nt_A {brgemm_hint_nt_undef};
    brgemm_kernel_hint_nt_t hint_load_nt_B {brgemm_hint_nt_undef};
    // this variable is used in tile decomposition heuristics
    // to calculate "effective" K value which may be different from just
    // K * batch_size for non-1x1 convolutions due to data overlap
    float K_koef = 1.f;
    // this flag may be used to omit some actions on calling from blocking
    // in convolutions init_conf
    bool test_call = false;
    // bd_mask is char array in which each element is a boolean value that
    // determines whether to write this row to the result matrix or skip
    const char *bd_mask;
    // static_offsets is a static array of fixed offsets used for
    // brgemm_static_offs batch kind
    const brgemm_batch_element_t *static_offsets;
};

struct brgemm_desc_t {
    brgemm_desc_t() {}
    brgemm_desc_t(const brgemm_desc_t &other);
    DNNL_API ~brgemm_desc_t();

    // Note: new added parameters must be taken into account in the brgemm
    // comparison function
    int bcast_dim = 0; // M;
    int load_dim = 0; // N;
    int reduce_dim = 0; // K;
    int LDA = 0;
    int LDB = 0;
    int LDC = 0;
    int LDD = 0;
    // we use two isa_ variables
    // isa_user to store the user provided isa value
    // isa_impl to store actual implementation. This can change until the kernel
    // is created, as calls to set_attr can affect this variable. Ex: bf32
    impl::cpu::x64::cpu_isa_t isa_user = isa_undef;
    impl::cpu::x64::cpu_isa_t isa_impl = isa_undef;
    float alpha = 0.0f;
    float beta = 0.0f;

    impl::data_type_t dt_a = data_type::undef;
    impl::data_type_t dt_c = data_type::undef;
    impl::data_type_t dt_b = data_type::undef;
    impl::data_type_t dt_d = data_type::undef;
    impl::data_type_t dt_bias = data_type::undef;

    dim_t stride_a = 0; // Offset in bytes
    dim_t stride_b = 0;

    brgemm_layout_t layout = brgemm_layout_undef;
    brgemm_batch_kind_t type = brgemm_batch_kind_t::brgemm_addr;
    bool is_dgmm = false; // set to true in brdgmm_desc_init
    bool with_sum = false;
    bool req_cal_comp_pads = false;
    bool req_comp_pads_with_bcast = false;

    float sum_scale = 0.0f;
    int32_t sum_zp = 0;
    impl::data_type_t sum_dt = data_type::undef;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_scales = false;
    bool skip_zp_b_compensation = false;
    bool skip_scales = false;
    bool n_bcast_1_load = false;

    brgemm_broadcast_t zp_type_a = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_b = brgemm_broadcast_t::none;
    brgemm_broadcast_t zp_type_c = brgemm_broadcast_t::none;

    int is_oc_scale = 0;
    bool with_dst_scales = false;
    // Grouping in batch used by brdgmm kernel
    int bs_group {0};

    brgemm_attr_t brgattr;

    // Derived  parameters
    int LDA2 {0}, LDB2 {0}, LDC2_M {0}, LDC2_N {0};
    bool is_blocked = false;

    int bdb = 0, bd_block = 0, bdb_tail = 0;
    int bdb2 = 0, bd_block2 = 0, bdb2_tail = 0;
    int ldb = 0, ld_block = 0, ldb_tail = 0;
    int ldb2 = 0, ld_block2 = 0, ldb2_tail = 0;
    int rdb = 0, rd_block = 0, rdb_tail = 0;
    int rd_step = 0, ld_step = 0;

    int typesize_A = 0;
    int typesize_B = 0;
    int typesize_C = 0;
    int typesize_D = 0;
    int typesize_bias = 0;

    bool is_ymm = false;
    bool is_zmm = false;
    bool is_tmm = false;
    bool is_int8 = false, is_int8_tmm = false;
    bool is_bf16 = false, is_bf16_tmm = false, is_bf16_emu = false;
    bool is_fp8 = false, is_fp8_tmm = false;
    bool is_f16 = false, is_f16_tmm = false;
    bool is_f32 = false;
    bool is_bf32 = false;

    bool has_int8_vnni = false;

    bool load_nt_A = false;
    bool load_nt_B = false;

    bool embd_bcst = false;
    bool with_bias = false;
    bool req_s8s8_compensation = false;
    bool with_weights_scale_adjust = false;
    brgemm_kernel_innermost_loop_t innermost_loop = brgemm_ld_loop_innermost;
    int is_M_tail = false;
    bool interleave_tilestores_ = false;
    brgemm_prf_t prfA, prfB, prfC;
    bool is_runtime_lda = false;
    bool is_runtime_ldb = false;
    bool is_runtime_ldc = false;
    bool is_runtime_ldd = false;

    static constexpr int MAX_VPAD = 100;
    static constexpr int AMX_TILES_NUM = 8;
    static constexpr int tilesize = 1024;

    void set_attr(const primitive_attr_t *ppdattr);
    void set_dst_md(const memory_desc_t *pdst_md);
    const primitive_attr_t *attr() const { return attr_; };
    const memory_desc_t *dst_md() const { return dst_md_; };

    // return 'true' when FP8 MAC is not natively supported by the CPU ISA
    bool is_fp8_via_convert() const {
        return is_fp8 && utils::one_of(isa_impl, avx10_1_512_amx_fp16);
    }

    bool is_input_convert() const { return is_bf32 || is_fp8_via_convert(); }

    bool is_row_major() const {
        assert(layout != brgemm_layout_undef);
        return layout == brgemm_row_major;
    }

    // Tile register decomposition
    int get_bd_block2() const noexcept {
        auto res = (bdb <= bd_block2) ? bdb : (bd_block2 + (bdb_tail ? 1 : 0));
        return res;
    }

    int get_ld_block2() const noexcept {
        auto res = (ldb <= ld_block2) ? ldb : (ld_block2 + (ldb_tail ? 1 : 0));
        return res;
    }

    int get_num_C_tiles() const noexcept {
        return get_bd_block2() * get_ld_block2();
    }
    int get_C_tensor(int m, int n, bool m_tail = false,
            bool n_tail = false) const noexcept {
        auto M = m_tail ? get_bd_block2() - 1 : m;
        auto N = n_tail ? get_ld_block2() - 1 : n;
        return (M * get_ld_block2() + N);
    }

    int get_num_A_tiles() const noexcept {
        const auto req_tiles = (bdb_tail && bdb > 1) ? 2 : 1;
        const auto max_tiles = AMX_TILES_NUM - get_num_C_tiles() - 1;
        const auto n_tiles = nstl::min(get_bd_block2(), max_tiles);
        assert(n_tiles >= req_tiles);
        return nstl::max(req_tiles, n_tiles);
    }

    int get_A_tensor(int m, bool m_tail = false) const noexcept {
        const auto full_A_tiles = get_num_A_tiles() - (bdb_tail ? 1 : 0);
        auto M = (m_tail || full_A_tiles == 0) ? get_num_A_tiles() - 1
                                               : m % full_A_tiles;
        return (get_num_C_tiles() + M);
    }

    int get_num_B_tiles() const noexcept {
        const auto req_tiles = (ldb_tail && ldb > 1) ? 2 : 1;
        const auto max_tiles
                = AMX_TILES_NUM - get_num_C_tiles() - get_num_A_tiles();
        const auto n_tiles = nstl::min(get_ld_block2(), max_tiles);
        assert(n_tiles >= req_tiles);
        return nstl::max(req_tiles, n_tiles);
    }

    int get_B_tensor(int n, bool n_tail = false) const noexcept {
        const auto full_B_tiles = get_num_B_tiles() - (ldb_tail ? 1 : 0);
        auto N = (n_tail || full_B_tiles == 0) ? get_num_B_tiles() - 1
                                               : n % full_B_tiles;
        return (get_num_C_tiles() + get_num_A_tiles() + N);
    }

    int get_convert_wsp_buffer_size() const noexcept {
        if (!is_input_convert()) return 0;
        const int n_bdb = bd_block2;
        const int n_rdb = rdb + (rdb_tail != 0);
        const int n_ldb = ldb + (ldb_tail != 0);
        const int downcvt_tiles = brgattr.max_bs * n_rdb * (n_bdb + n_ldb);
        return downcvt_tiles * tilesize;
    }

    int get_wsp_buffer_size() const noexcept {
        int sz = 0;
        if (is_tmm) {
            sz = get_num_C_tiles() * tilesize; // postops buffer
            sz += get_convert_wsp_buffer_size();
            if (amx_wary_k_tail()) sz += tilesize;
        }
        return sz;
    }

    // A class version of the `static` version of the function.
    // Note: used in benchdnn only, not used inside the library.
    bool is_b_data_layout_vnni() const {
        return is_b_data_layout_vnni(dt_a, dt_b, brgattr.b_is_vnni, isa_impl);
    }

    // This function indicates when VNNI granularity packing is expected by the
    // kernel.
    //
    // Note: used as the `static` function in ukernel only, not anywhere else.
    //   `static`-ness is required to identify if the transform routine must be
    //   used for the ukernel to work properly. This information is critical
    //   because the transform routine accepts only 4 `ldb` values which affects
    //   ukernel creation. Otherwise, the user must create the ukernel object,
    //   query the packing info, and if it's required, likely re-create the
    //   object with a different `ldb` value, which may not work because
    //   creation stage for user's application may not provide all the info to
    //   create a ukernel object.
    // Note: for `bf32` (or brgattr.fpmath_mode_ == bf16) the function returns
    //   `true` because the data transformation to vnni layout is internal and
    //   transparent to the user.
    // Note: the library MUST NOT break the ability to provide this information
    //   without brgemm_desc_t object creation.
    static bool is_b_data_layout_vnni(data_type_t dt_a, data_type_t dt_b,
            bool attr_b_is_vnni, cpu_isa_t isa) {
        using namespace data_type;
        switch (dt_b) {
            case f32: return false;
            // Note: `dt_a == f32` means implicit up-conversion of B to f32.
            case f16:
                return dt_a != f32
                        && (is_f16_b_non_amx_vnni(dt_b, attr_b_is_vnni, isa)
                                || is_superset(isa, avx512_core_amx_fp16)
                                || is_superset(isa, avx2_vnni_2));
            // Note: `dt_a == f32` means implicit up-conversion of B to f32.
            case bf16: return dt_a != f32;
            default: return true;
        }
    }

    // This function indicates when the kernel would operate with the D pointer
    // (`true`) and when not (`false`). It's important to distinguish these two
    // cases due to the fact that kernel would ignore D pointer completely if
    // no post-accumulation work is identified.
    //
    // Correspondent decisions are done in `store_accumulators` function.
    // The function is used inside kernel generation and ukernel API.
    // TODO: extend usage to primitives (each of them utilize their own copy
    // of this definition).
    bool are_post_ops_applicable() const {
        const bool has_zero_points = !utils::everyone_is(
                brgemm_broadcast_t::none, zp_type_a, zp_type_b, zp_type_c);
        return dt_c != dt_d || with_eltwise || with_binary || with_scales
                || with_bias || with_sum || req_s8s8_compensation
                || has_zero_points || with_dst_scales;
    }

    bool is_xf16() const noexcept { return is_bf16 || is_f16; }

    bool is_f16_b_non_amx_vnni() const {
        return is_f16_b_non_amx_vnni(dt_b, brgattr.b_is_vnni, isa_impl);
    }

    // Note: `static` version appears because of `static is_b_data_layout_vnni`.
    static bool is_f16_b_non_amx_vnni(
            data_type_t dt_b, bool attr_b_is_vnni, cpu_isa_t isa) {
        // This function controls the code section which relies on
        // `avx512_core_fp16` instructions directly.
        return dt_b == data_type::f16 && attr_b_is_vnni
                && isa == avx512_core_fp16;
    }

    bool reduce_by_words() const {
        return is_bf16_tmm || is_f16_tmm || is_input_convert();
    }
    int max_rd_block() const { return reduce_by_words() ? 32 : 64; }
    int rd_block_step() const { return (reduce_by_words() && !is_fp8) ? 2 : 4; }

    bool amx_may_extend_k() const {
        return (is_superset(isa_impl, avx512_core_amx) && brgattr.extendable_k
                && (reduce_dim % data_type_vnni_granularity(dt_a)
                        || (reduce_dim > max_rd_block()
                                && reduce_dim % max_rd_block())));
    }
    bool amx_wary_k_tail() const {
        return amx_may_extend_k() && brgattr.wary_A_k_tail_read;
    }

    bool operator==(const brgemm_desc_t &rhs) const;
    bool operator<(const brgemm_desc_t &rhs) const;

private:
    primitive_attr_t *attr_ {nullptr};
    memory_desc_t *dst_md_ {nullptr};
    void set_attr_null() { attr_ = nullptr; };
    void set_dst_md_null() { dst_md_ = nullptr; };

    void cleanup_attr();
    void cleanup_dst_md();

    // The default assignment operator is intended to be used in custom copy
    // constructor only to avoid copying field-by-field
    brgemm_desc_t &operator=(const brgemm_desc_t &) = default;
};

struct brgemm_dynamic_values_t {
    dim_t dynamic_LDA = 0;
    dim_t dynamic_LDB = 0;
    dim_t dynamic_LDC = 0;
    dim_t dynamic_LDD = 0;
    brgemm_dynamic_values_t(dim_t LDA, dim_t LDB, dim_t LDC, dim_t LDD)
        : dynamic_LDA(LDA)
        , dynamic_LDB(LDB)
        , dynamic_LDC(LDC)
        , dynamic_LDD(LDD) {}
};

struct brgemm_kernel_params_t {
    const void *ptr_A;
    const void *ptr_B;
    const brgemm_batch_element_t *batch;
    void *ptr_C;

    const void *ptr_bias;
    void *ptr_D;

    /* kernel takes single pointer scales, but configuration relies on a
     * combination of arg scales. This helps to reuse attributes from
     * primitives, but requires them to pre-compute
     * scales = src_scale * wei_scale[:]
     */
    const void *ptr_scales;
    void *ptr_buf;

    size_t do_post_ops;
    size_t do_apply_comp;
    size_t BS;

    /*
     * ptr to table of void * elements that are pointers to post_op binary
     * src1 tensors
     */
    const void *post_ops_binary_rhs_arg_vec;
    size_t oc_logical_off;
    size_t first_mb_matrix_addr_off;
    size_t dst_row_logical_off;

    const char *data_C_ptr_;

    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *a_zp_values = nullptr;
    const void *c_zp_values = nullptr;
    size_t skip_accm = 0;
    int32_t zp_a_val = 1;
    const void *ptr_dst_scales = nullptr;
    dim_t dynamic_LDA = 0;
    dim_t dynamic_LDB = 0;
    dim_t dynamic_LDC = 0;
    dim_t dynamic_LDD = 0;
};

template <typename Vmm>
struct jit_brgemm_kernel_t;
struct jit_brgemm_amx_uker_base_t;
template <typename Vmm>
struct jit_brdgmm_kernel_base_t;
class jit_generator;

struct brgemm_kernel_t {
    brgemm_kernel_t() {};
    virtual ~brgemm_kernel_t() {};
    virtual status_t create_kernel() = 0;
    virtual void operator()(brgemm_kernel_params_t *) const = 0;
    virtual const jit_generator *get_jit_generator() const = 0;
    virtual const brgemm_desc_t &get_brg() const = 0;
};

struct jit_base_brgemm_kernel_t : public jit_generator {
    jit_base_brgemm_kernel_t(const char *impl_name, cpu_isa_t isa_impl)
        : jit_generator(impl_name, isa_impl) {}
    virtual const brgemm_desc_t &get_brg() const = 0;
};

template <typename Vmm>
struct brgemm_kernel_common_t : public brgemm_kernel_t {
    brgemm_kernel_common_t(const brgemm_desc_t &abrd);
    ~brgemm_kernel_common_t();

    status_t create_kernel() override;
    void operator()(brgemm_kernel_params_t *) const override;
    virtual const jit_generator *get_jit_generator() const override;
    virtual const brgemm_desc_t &get_brg() const override {
        return ((jit_base_brgemm_kernel_t *)brgemm_kernel_)->get_brg();
    }

private:
    jit_brgemm_kernel_t<Vmm> *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_kernel_common_t);
};

struct brgemm_amx_uker_t : public brgemm_kernel_t {
    brgemm_amx_uker_t(const brgemm_desc_t &abrd);
    ~brgemm_amx_uker_t();

    status_t create_kernel() override;
    void operator()(brgemm_kernel_params_t *) const override;
    virtual const jit_generator *get_jit_generator() const override;
    virtual const brgemm_desc_t &get_brg() const override {
        return ((jit_base_brgemm_kernel_t *)brgemm_kernel_)->get_brg();
    }

private:
    jit_brgemm_amx_uker_base_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_amx_uker_t);
};

template <typename Vmm>
struct brdgmm_kernel_t : public brgemm_kernel_t {
    brdgmm_kernel_t(const brgemm_desc_t &abrd);
    ~brdgmm_kernel_t();

    status_t create_kernel() override;
    void operator()(brgemm_kernel_params_t *) const override;
    virtual const jit_generator *get_jit_generator() const override;
    virtual const brgemm_desc_t &get_brg() const override {
        return ((jit_base_brgemm_kernel_t *)brgemm_kernel_)->get_brg();
    }

private:
    jit_brdgmm_kernel_base_t<Vmm> *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brdgmm_kernel_t);
};

/// @param bias Vector of bias (vector length is N)
/// @param scales - Vector of scale factor values which represents combination
///     scale factors for matrixes A and B. If brgemm_desc_t::is_oc_scale = true
///     vector length is N otherwise it must be broadcasted to vector of simd
///     width length
/// @param binary_post_ops_rhs - Ptr to table of pointers to tensors used as rhs
///     in binary post-operation { void* binary_op_tensor1, ...,
///      void* binary_op_tensor_n}
/// @param oc_logical_off - Used in binary postops in per_oc bcast strategy.
///     Offset to start oc processed by given thread in elements.
/// @param dst_row_logical_off - Used in binary postops in per_oc bcast
///     strategy. Offset to start oc processed by given thread in elements.
/// @param a_zp_compensations - Pre-computed compensations for A matrix zero
///     point values.
/// @param b_zp_compensations - Pre-computed compensations for B matrix zero
///     point values.
/// @param c_zp_values - C matrix zero point values.
/// @param skip_accumulation - specifies whether to skip accumulation when
///    computing post-ops. `Beta` value from descriptor affects final
///    accumulator values taken.
/// @param do_only_comp - specifies whether to perform accumulation only and skip
///    post-ops.
/// @param do_only_zp_a_val - specifies to apply pre-calculated compensation for
///     A zero point only and skip the rest post-ops.
/// @param zp_a_val - zero point value for A, required to adjust compensation
///     values if do_only_zp_a_val = true.
/// @param dst_scales - Vector of inverted scale factor values for matix C,
///     common scale vector type only is supported, it must be broadcasted to
///     vector of simd width length.
///
struct brgemm_post_ops_data_t {
    brgemm_post_ops_data_t() = default;
    brgemm_post_ops_data_t(const void *bias, const float *scales,
            const void *binary_post_ops_rhs, size_t oc_logical_off,
            const size_t dst_row_logical_off = 0,
            const char *data_C_ptr_ = nullptr,
            const size_t first_mb_matrix_addr_off = 0,
            const void *a_zp_compensations = nullptr,
            const void *b_zp_compensations = nullptr,
            const void *c_zp_values = nullptr, bool skip_accumulation = false,
            int32_t zp_a_val = 1, bool do_only_comp = false,
            bool do_only_zp_a_val = false, const float *dst_scales = nullptr,
            const void *a_zp_values = nullptr)
        : bias(bias)
        , scales(scales)
        , binary_post_ops_rhs(binary_post_ops_rhs)
        , oc_logical_off(oc_logical_off)
        , dst_row_logical_off(dst_row_logical_off)
        , data_C_ptr_(data_C_ptr_)
        , first_mb_matrix_addr_off(first_mb_matrix_addr_off)
        , a_zp_compensations(a_zp_compensations)
        , b_zp_compensations(b_zp_compensations)
        , c_zp_values(c_zp_values)
        , skip_accumulation(skip_accumulation)
        , zp_a_val {zp_a_val}
        , do_only_comp {do_only_comp}
        , do_only_zp_a_val {do_only_zp_a_val}
        , dst_scales(dst_scales)
        , a_zp_values(a_zp_values) {}

    const void *bias = nullptr;
    const float *scales = nullptr;
    const void *binary_post_ops_rhs = nullptr;
    size_t oc_logical_off = 0;
    size_t dst_row_logical_off = 0;
    const char *data_C_ptr_ = nullptr;
    size_t first_mb_matrix_addr_off = 0;
    const void *a_zp_compensations = nullptr;
    const void *b_zp_compensations = nullptr;
    const void *c_zp_values = nullptr;
    const bool skip_accumulation = false;
    int32_t zp_a_val = 1;
    const bool do_only_comp = false;
    const bool do_only_zp_a_val = false;
    const float *dst_scales = nullptr;
    const void *a_zp_values = nullptr;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
