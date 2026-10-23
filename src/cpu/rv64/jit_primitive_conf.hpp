/*******************************************************************************
* Copyright 2025 ZTE Corporation
* Copyright 2026 Institute of Software, Chinese Academy of Sciences
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

#ifndef CPU_RV64_JIT_PRIMITIVE_CONF_HPP
#define CPU_RV64_JIT_PRIMITIVE_CONF_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"

#include "cpu/rv64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

inline int calculate_end_padding(int start_padding, int dst_size, int src_size,
        int spatial_stride, int dilated_filter_size) {
    return (dst_size - 1) * spatial_stride + dilated_filter_size
            - (src_size + start_padding);
}

// Memory layout family handled by the pooling primitive.
//   blocked - nChw{c_block}c (c_block = VLEN/32), the x64/aarch64-style baked
//             kernel path (channels are the vector dim, c_block per register).
//   nspc    - channels-last plain; also the baked kernel (c_off = C).
//   ncsp    - plain nchw; the RETAINED rv64-native path (vectorize along OW/C,
//             no plain<->blocked transpose since rv64 has no JIT transpose).
enum class jit_pool_tag_kind_t { undef, nspc, ncsp, blocked };

// Fused-binary src1 broadcast category relative to dst, uniform across the whole
// post-op chain (enforced by the pd gate). Classified ONCE during init_conf and
// stored in jpp.binary_bcast; the kernels and driver read it back instead of
// re-deriving the category, so the injector rhs load form (scalar flw / per-oc
// contiguous vle / full-dst strided vlse) and the shared rhs offset stay in sync.
//   none    - eltwise-only chain or no post-ops (no binary rhs)
//   scalar  - per-tensor src1 (one value broadcast to every lane)
//   per_oc  - [1,C,1,..] dense, matching dst's channel dim
//   full_dst - dst-shaped src1 (per-element)
enum class pool_binary_bcast_t {
    none = 0,
    scalar = 1,
    per_oc = 2,
    full_dst = 3
};

// Centralized pooling configuration. For the blocked/nspc baked kernel this is
// the x64/aarch64 jit_pool_conf_t superset (c_block, ur, ur_bc, training,
// backward, workspace dtype, ...). The native path (nspc/ncsp forward inference,
// which is faster on RV64 than the baked port) additionally uses the fields
// grouped at the bottom (ur_w / fuse_* / with_relu); use_native selects it.
struct jit_pool_conf_t {
    int ndims;
    int mb, c, c_without_padding;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad;
    alg_kind_t alg;
    bool is_training;
    bool is_backward;
    bool simple_alg; // backward: kd<=stride_d (single accumulation pass)
    bool is_c_padded; // blocked tag with padded channels (c_without_padding<c)
    data_type_t ind_dt; // workspace index dtype (u8 or s32), max training/bwd
    data_type_t src_dt;
    data_type_t dst_dt;
    int dt_size;
    bool is_f16; // f16 data (f32 accumulation; requires zvfh)
    // bf16 data (f32 accumulation; requires the Zvfbfmin converts, tracked by
    // the zvfbfwma bit). Mutually exclusive with is_f16; both imply a 2-byte
    // element that is widened on load and narrowed on store.
    bool is_bf16;

    // Blocked/nspc vectorization: one m1 register == c_block f32 lanes.
    int c_block, c_tail, nb_c;
    int ur; // output-width unroll baked into the kernel
    int ur_bc, ur_bc_tail; // channel-block unroll (nspc) / 1 (blocked)

    jit_pool_tag_kind_t tag_kind;
    // True: the retained native kernel (nspc/ncsp forward inference). False: the
    // x64/aarch64-style baked kernel (blocked, and nspc/blocked training +
    // backward).
    bool use_native;
    cpu_isa_t isa;

    post_ops_t post_ops;
    bool with_postops;
    bool with_eltwise;
    bool with_binary;
    // Broadcast category of the fused binary chain (classified once by the pd
    // gate; none when there is no binary). See pool_binary_bcast_t.
    pool_binary_bcast_t binary_bcast;
    int nthr;

    // --- native ncsp path only (retained rv64 design) ---
    bool with_relu; // f32 single-ReLU flag; feeds empty_window_value()
    bool fuse_eltwise; // eltwise chain fused (ncsp native)
    bool fuse_binary; // binary fused (ncsp native, channel-vec)
    float relu_alpha;
    int ur_w; // ncsp native interior OW-unroll
};

// Per-call arguments for the x64/aarch64-style baked kernel (blocked/nspc).
// Mirrors aarch64 jit_uni_pooling_args_t: the pooling window is baked into the
// generated code and the driver passes per-(n,oh[,od]) padding counts and the
// channel-block slice. The last two fields carry the rv64 in-kernel binary
// post-op rhs (host-prepared pointer array + dst logical origin for full-dst).
struct jit_uni_pooling_args_t {
    const void *src;
    const void *dst;
    const void *indices; // max training (store) / backward (load) workspace
    const void *zero_ptr; // backward: diff_src zeroing base
    size_t zero_id; // backward: # input planes to zero
    size_t zero_ih; // backward: # input rows to zero
    size_t kd_padding; // valid kernel planes (kd - front/back overflow)
    size_t kh_padding; // valid kernel rows   (kh - top/bottom overflow)
    size_t kh_padding_shift; // index base offset from t/f overflow
    size_t kd_padding_shift; // index d-step adjustment
    float ker_area_h; // avg_exclude divisor base (valid kh*kd area)
    size_t ur_bc; // # channel blocks to process in this call
    size_t b_c; // # channel blocks already processed
    const void *post_ops_binary_rhs_arg_vec; // per-binary rhs pointer array
    const void *dst_orig; // dst logical origin (raw base + off_l(0)); the
            // full-dst binary offset is (dst - dst_orig)
};

// Per-call arguments for the RETAINED native ncsp kernel (renamed from the old
// jit_uni_pooling_args_t). Shape-agnostic: the window bounds and strides are
// passed here, so one routine serves ncsp, OW==1, and boundary columns.
struct jit_uni_pool_ncsp_args_t {
    const void *src;
    void *dst;
    dim_t channels;
    dim_t id_start, ih_start, iw_start;
    dim_t id_end, ih_end, iw_end;
    dim_t inW_stride; // IW * C (elements) for nspc, IW for ncsp
    dim_t inD_stride; // IH * IW * C (elements) for nspc, IH*IW for ncsp
    dim_t w_spatial_byte_stride; // C*dt_size for nspc, 1*dt_size for ncsp
    float init_val; // -FLT_MAX (max) / 0 (avg); the f16 kernel bakes its own
    float scale_val; // 1.0/count for avg, unused for max
    float relu_alpha;
    bool with_relu;
    dim_t src_vec_byte_stride; // byte stride between vector elements in src
    dim_t dst_vec_byte_stride; // byte stride between vector elements in dst
    // Binary post-op: the injector runs in indirect mode so one chain can carry
    // any number of binaries (all sharing one broadcast). post_op_rhs is the base
    // of the per-binary rhs (src1) ORIGIN pointer array (one f32 pointer per
    // binary). post_op_off0 is the shared byte offset of the first active lane:
    // 0 for scalar/per-oc, and the full-dst position's channel-0 element offset
    // (* sizeof(f32)) otherwise; the kernel advances it per channel chunk.
    const void *post_op_rhs = nullptr;
    dim_t post_op_off0 = 0;
    // Position batching: process n_pos output positions in one call (the
    // positions share the window shape, so only the base pointers advance).
    // Used by the f16 nspc generic path and the ncsp interior rows; ignored
    // (single position) when a binary post-op or max-training is fused.
    dim_t n_pos = 1;
    dim_t pos_src_byte_stride = 0;
    dim_t pos_dst_byte_stride = 0;
    // Max forward-training only (f32): the argmax workspace for this output
    // position (indices) and the running kernel-window index. pos_base = the
    // flattened index of the first (clamped) window element
    // (kd_base*KH*KW + kh_base*KW + kw_base); as the kernel sweeps the clamped
    // window it increments by 1 per iw, then adds pos_ih_step (= KW - kw_count)
    // after each row and pos_id_step (= KH*KW - kh_count*KW) after each plane to
    // skip the clamped-off positions. The per-channel argmax is stored here.
    void *indices = nullptr;
    dim_t pos_base = 0;
    dim_t pos_ih_step = 0;
    dim_t pos_id_step = 0;
    // Byte stride between channel elements in the argmax workspace: ind_dt_size
    // for the contiguous nspc workspace, dst_spatial * ind_dt_size for the
    // strided ncsp workspace. The per-channel store is unit when this equals
    // ind_dt_size, otherwise strided (vsse).
    dim_t ws_vec_byte_stride = 0;
};

// Per-call arguments for the shape-baked interior kernel (nspc f32 forward
// inference). The unroll structure (kw, stride_w, ur_w), algorithm, init value,
// avg_include scale, and the fused eltwise chain are baked into the generated
// code; element strides are passed so 64-bit strides stay correct.
struct jit_uni_pool_interior_args_t {
    const void *src; // channel 0 of (id_start, ih_start, iw_block_start)
    void *dst; // channel 0 of the first output column in the block
    dim_t channels; // remaining channels (VLA tail handled by vsetvli)
    dim_t kh_count; // valid kernel rows  = ih_end - ih_start
    dim_t kd_count; // valid kernel planes = id_end - id_start
    dim_t n_blocks; // number of ur_w output blocks to process in this call
    dim_t w_stride; // byte stride between W positions (= C * dt_size)
    dim_t inW_stride; // byte stride between H rows (= iw * C * dt_size)
    dim_t inD_stride; // byte stride between D planes (= ih * iw * C * dt_size)
    float scale_val; // 1 / window for avg_exclude (ignored otherwise)
};

// One covering-output contribution to a single input position in the native
// gather backward. Backward pooling is a gather: for each input position the
// driver enumerates the (few) output positions whose pooling window covers it
// and fills one contribution per output; the kernel accumulates their diff_dst
// channel rows into the input's diff_src channel row and stores it once (no
// scatter read-modify-write, so parallelising over inputs has no data race).
// index/scale carry the max/avg payload: max adds diff_dst only where the stored
// argmax equals index; avg adds diff_dst * scale (scale = 1 / num_summands, so
// include- and exclude-padding are uniform per contribution).
struct jit_uni_pool_bwd_contrib_t {
    const void *diff_dst; // channel 0 of the covering output
    const void *ws; // channel 0 of the covering output workspace (max)
    int32_t index; // full-kernel index of this input in the window (max)
    float scale; // 1 / num_summands for this output (avg)
};

// Per-call arguments for the native gather backward kernel (one input position).
struct jit_uni_pool_bwd_args_t {
    void *diff_src; // channel 0 of this input position (store target)
    const jit_uni_pool_bwd_contrib_t *contribs;
    dim_t count; // number of covering outputs (0 -> diff_src row is zeroed)
    dim_t channels;
    dim_t src_vec_byte_stride; // diff_src channel stride (nspc: dt, ncsp: ID*IH*IW*dt)
    dim_t dst_vec_byte_stride; // diff_dst channel stride (nspc: dt, ncsp: OD*OH*OW*dt)
    dim_t ws_vec_byte_stride; // ws channel stride (max; ncsp: OD*OH*OW*ind_sz)
};

// Resampling forward configuration, filled by init_conf, read by the driver
// and the kernel.
struct jit_resampling_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    alg_kind_t alg; // resampling_nearest or resampling_linear
    data_type_t data_type; // f32 or f16
    int dt_size;
    // Layout, in elements, taken from the blocking descriptors (not tags). The
    // vector runs along C; the kernel tests c_stride at run time to pick
    // unit-stride or strided access. block is the channel inner block, 1 when
    // plain: channels are contiguous inside a block (c_stride 1, cb_stride
    // steps between blocks), while a plain layout is one group (cb_stride 0).
    int block;
    dim_t src_c_stride, dst_c_stride;
    dim_t src_cb_stride, dst_cb_stride;
    dim_t src_mb_stride, dst_mb_stride;
    dim_t src_d_stride, src_h_stride, src_w_stride;
    dim_t dst_d_stride, dst_h_stride, dst_w_stride;
    cpu_isa_t isa;
    // Source points combined per output point: 1 for nearest; for linear
    // 1 << (spatial dims) = 2 (1D) / 4 (2D) / 8 (3D).
    int num_corners;
    // Plain ncsp (channel-strided, W contiguous) f32 without a binary
    // post-op: vectorize along OW instead of C. One call computes one output
    // row from w_rows source rows (nearest: 1; linear: 1 / 2 / 4 for 1D / 2D /
    // 3D), gathering columns through per-OW index and weight tables.
    bool w_vec;
    int w_rows;
    // Injector-fused post-ops: an eltwise chain (f32/f16, computed at f32) plus
    // at most one binary (f32 only).
    bool with_postops;
    bool fuse_eltwise;
    bool fuse_binary;
    // sum (dst = sum_scale * dst + result). The injector skips sum entries, so
    // the kernel applies it itself at sum_idx; -1 when there is no sum.
    bool fuse_sum;
    int sum_idx;
    float sum_scale;
    post_ops_t post_ops;
};

// Per-output-point kernel arguments. The driver resolves the corner pointers
// and weights on the host via resampling_utils (the same math as ref); the
// kernel combines their channel vectors.
struct jit_resampling_args_t {
    // Channel-0 of each source corner; conf.num_corners of them are live.
    const void *src[8];
    void *dst;
    // Product of the 1D linear weights per corner; unused for nearest.
    float weights[8];
    dim_t channels; // channels to combine (the vector work); OW when w_vec
    // w_vec only: per-OW tables shared by every row. w_idx[k] holds the byte
    // offset of the k-th source column (k = 0 left, 1 right; nearest uses
    // only 0), w_wei[k] its 1D linear weight. src[r] / weights[r] then are
    // the source rows and their D*H weight products.
    const void *w_idx[2];
    const void *w_wei[2];
    dim_t src_vec_byte_stride; // byte stride between consecutive channels in src
    dim_t dst_vec_byte_stride; // byte stride between consecutive channels in dst
    // Binary rhs for the injector's indirect mode, as in pooling: post_op_rhs
    // is the per-binary src1 origin pointer array, post_op_off0 the shared byte
    // offset of the first active lane (0 scalar, channel-group per-oc, this
    // point's channel-0 element offset full-dst). The kernel advances it.
    const void *post_op_rhs = nullptr;
    dim_t post_op_off0 = 0;
};

enum class binary_op_t : unsigned { none, c_blocked, n_spatial_c, n_c_spatial };

enum class binary_bcast_t : unsigned {
    none, // tensor operation
    scalar,
    per_batch,
    per_c,
    per_w
};

// Binary configuration, mirroring the x64/aarch64 jit_binary_conf_t. Fields that
// only make sense for x86 (i8/bf16/f16 selection, per_w scratchpad expansion,
// isa/ternary-vmm budgeting) are dropped; RVV keeps a single f32 compute domain
// and lets the driver's strategies compute offsets from the mds at run time.
// A compact RV64 physical run plan is retained for dense combinations outside
// the synchronized x64 strategy matrix.
struct jit_binary_conf_t {
    binary_op_t op_type = binary_op_t::none;
    binary_bcast_t bcast_type = binary_bcast_t::none;
    bool do_scale_src0 = false;
    bool do_scale_src1 = false;
    bool do_sum = false;
    bool with_eltwise = false;
    bool with_binary = false;
    bool with_postops = false;
    float sum_scale = 0.f;
    // src1 is a single value broadcast over the whole run: the kernel keeps it
    // in a scalar FP register and uses the .vf instruction forms (x64 broadcasts
    // it into a full vector register instead).
    bool broadcast_src1_value = false;
    // src1 advances 1:1 with the run (contiguous vle, or strided vlse when the
    // src0/src1 plain layouts differ). When neither this nor broadcast is set,
    // the driver slices the call so a fixed src1 vector maps to each run.
    bool use_stride_src1 = false;
    // x64 parity: a per_oc/per_oc_spatial post-op rhs exists. The dispatch
    // then routes to the channel-aligned per_c/per_w strategies and the
    // kernel addresses the rhs with a per-call channel offset (x64's model)
    // instead of the general per-lane gather; an op_t::none src0 cannot be
    // routed and keeps the gather-addressed flat path. x64's
    // use_stride_rhs_postops is not needed: the dst_l_off the kernel already
    // maintains advances with the run.
    bool postops_per_oc_broadcast_exists = false;
    bool is_ternary_op = false; // select: dst = src2 ? src0 : src1 (src2 is s8)
    bool is_i8 = false; // dst is s8/u8 (x64-parity gating; kernel saturates)
    bool is_src_different_layouts = false;
    dim_t outer_dims = 1;
    // src1 element stride along the run when layouts differ: 1 = contiguous
    // (vle), >1 = strided (vlse; e.g. nchw:nhwc — x64 gathers with an indices
    // vector instead). dim_t rather than x64's int: the byte stride is
    // materialized in a 64-bit GPR, no narrowing needed.
    dim_t src1_stride = 1;
    int not_bcasted_sp_dims = 0;

    // RV64's VLA kernel can also execute dense broadcast/layout combinations
    // outside the x64 strategy matrix. The PD coalesces the physical dst
    // dimensions into contiguous inner runs and records the src1 stride for
    // each outer dimension. This retains the pre-synchronization JIT coverage
    // while keeping the x64-style driver paths as the primary implementation.
    bool use_generic_strategy = false;
    bool generic_whole = false;
    bool generic_scalar_inner = false;
    bool generic_src1_same_layout = false;
    int generic_ndims = 0;
    dim_t generic_inner = 0;
    dim_t generic_n_outer = 0;
    dim_t generic_total = 0;
    dim_t generic_tail = 0;
    int generic_tail_axis = -1;
    dims_t generic_outer_dims = {};
    dims_t generic_src1_strides = {};

    data_type_t src0_type = data_type::undef;
    data_type_t src1_type = data_type::undef;
    data_type_t dst_type = data_type::undef;
};

// Per-call arguments. Pointer/size fields mirror x64/aarch64
// jit_uni_binary_args_t names; divergences (element-count work_amount, scale
// values instead of pointers) are noted in place.
struct jit_uni_binary_args_t {
    const void *src0;
    const void *src1;
    const void *src2; // ternary select condition (s8), else null
    const void *dst;
    // Per-binary post-op rhs base array, or null.
    const void *post_ops_binary_rhs_arg_vec;
    // Elements to process in this call. The VLA loop consumes elements via
    // vsetvli, unlike x64's byte-based spat_offt_count.
    dim_t work_amount;
    // Per-tensor scale and sum values, resolved by the driver (x64 passes
    // scale pointers and broadcasts in-kernel; sum_scale is baked into code).
    float scales_src0, scales_src1, sum_scale;
    // x64: the dst tensor origin; the binary post-op injector recovers each
    // chunk's element offset as (dst address - dst_orig) >> log2(dt size).
    const void *dst_orig;
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int iw, ih, id;
    int ow, oh, od;
    int os, is;
    int kw, kh, kd;
    int stride_w, stride_h, stride_d;
    int t_pad, l_pad, f_pad;

    int ic_block, oc_block;
    int load_block, reduce_block;
    int bcast_block;

    dim_t load_dim, bcast_dim, reduce_dim;

    int simd_w;
    int ur, ur_tail;
    int load_loop_blk;
    int reduce_loop_unroll;
    int nthr;
    int nb_bcast, nb_load, nb_reduce, load_grp_count;
    int nb_load_blocking, nb_load_blocking_max;
    int nb_bcast_blocking, nb_bcast_blocking_max;
    int nb_reduce_blocking;

    dim_t reduce_loop_bcast_step;
    int reduce_loop_load_step;
    int bcast_loop_bcast_step;
    int bcast_loop_output_step;
    int load_loop_load_step;
    int load_loop_iter_step;

    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_binary;
    bool with_dw_conv;

    // src dtype: f32, bf16 (Zvfbfwma), or f16 (Zvfh).
    data_type_t src_dt;
    // weights dtype. Equal to src_dt for f32/f32 and the symmetric bf16/f16
    // paths; for weight compression it is bf16/f16 while src_dt is f32 and the
    // weights are widened to f32 in-kernel (like x64 is_f32_bf16/is_f32_f16).
    data_type_t wei_dt;
    // bias dtype: f32, or bf16/f16 (== src_dt) widened to f32 in-kernel.
    data_type_t bia_dt;
    // destination dtype: f32, or f16 narrowed from the f32 accumulators in
    // kernel (Zvfh).
    data_type_t dst_dt;

    int typesize_in;
    int typesize_out;
    int typesize_wei;
    int typesize_bia;
    int typesize_acc;

    format_tag_t src_tag, wei_tag, dst_tag;
};

struct jit_1x1_conv_args_t {
    const void *bcast_data;
    const void *load_data;
    const void *output_data;
    const void *bias_data;

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t first_last_flag;
};

enum {
    FLAG_REDUCE_FIRST = 1 << 0,
    FLAG_REDUCE_LAST = 1 << 1,
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
