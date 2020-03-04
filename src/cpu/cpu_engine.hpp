/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_ENGINE_HPP
#define CPU_ENGINE_HPP

#include <assert.h>

#include "../common/engine.hpp"
#include "c_types_map.hpp"
#include "cpu_isa_traits.hpp" // also give common/cpu_target.h
#include "dnnl.h"

#if 0 // oops, I removed the verbose create wrapper.(can be separate PR).
/** This adds extra code to respect \c dnnl_set_verbose(int) or environment
 * DNNL_VERBOSE values 3 and 4. It's activated by `cmake -DDNNL_VERBOSE=EXTRA`,
 * and defaults to '1' for debug builds.
 */
#define VERBOSE_PRIMITIVE_SKIP DNNL_VERBOSE_EXTRA
#endif

namespace dnnl {
namespace impl {
namespace cpu {

#define DECLARE_IMPL_LIST(kind) \
    const engine_t::primitive_desc_create_f *get_##kind##_impl_list( \
            const kind##_desc_t *desc);

DECLARE_IMPL_LIST(batch_normalization);
DECLARE_IMPL_LIST(binary);
DECLARE_IMPL_LIST(convolution);
DECLARE_IMPL_LIST(deconvolution);
DECLARE_IMPL_LIST(eltwise);
DECLARE_IMPL_LIST(inner_product);
DECLARE_IMPL_LIST(layer_normalization);
DECLARE_IMPL_LIST(lrn);
DECLARE_IMPL_LIST(logsoftmax);
DECLARE_IMPL_LIST(matmul);
DECLARE_IMPL_LIST(pooling);
DECLARE_IMPL_LIST(resampling);
DECLARE_IMPL_LIST(rnn);
DECLARE_IMPL_LIST(shuffle);
DECLARE_IMPL_LIST(softmax);

#undef DECLARE_IMPL_LIST

class cpu_engine_t : public engine_t {
public:
    cpu_engine_t()
        : engine_t(engine_kind::cpu, get_default_runtime(engine_kind::cpu)) {}

    /* implementation part */

    virtual status_t create_memory_storage(memory_storage_t **storage,
            unsigned flags, size_t size, void *handle) override;

    virtual status_t create_stream(stream_t **stream, unsigned flags) override;

    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const override;
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const memory_desc_t *src_md,
            const memory_desc_t *dst_md) const override;
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const override;
    virtual const primitive_desc_create_f *get_implementation_list(
            const op_desc_t *desc) const override {
        static const primitive_desc_create_f empty_list[] = {nullptr};

#define CASE(kind) \
    case primitive_kind::kind: \
        return get_##kind##_impl_list((const kind##_desc_t *)desc);
        switch (desc->kind) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(logsoftmax);
            CASE(matmul);
            CASE(pooling);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            default: assert(!"unknown primitive kind"); return empty_list;
        }
#undef CASE
    }
};

class cpu_engine_factory_t : public engine_factory_t {
public:
    virtual size_t count() const override { return 1; }
    virtual status_t engine_create(
            engine_t **engine, size_t index) const override {
        assert(index == 0);
        *engine = new cpu_engine_t();
        return status::success;
    };
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

/** \def INSTANCE_CREATOR(...)
 * \b Each cpu_layer_list.cpp file should set this macro.
 * Usually
 * ```
 * \#define INSTANCE_CREATOR(...) DEFAULT_INSTANCE_CREATOR(__VA_ARGS__)
 * // or equivalently
 * // \#define INSTANCE_CREATOR(...) &primitive_desc_t::create<__VA_ARGS__::pd_t>
 * ```
 * is appropriate.
 *
 * Some files, like \ref cpu_sum.cpp and \ref cpu_concat.cpp may use
 * ```
 * \#define INSTANCE_CREATOR(...) __VA_ARGS__::pd_t::create
 * ```
 *
 * `cmake -DDNNL_VERBOSE=EXTRA` builds may wish to provide a generic wrapper to
 * track impls that declined the invitation (and maybe why they declined) with a
 * generic wrapper function like
 * ```
 * \#define INSTANCE_CREATOR(...) verbose_primitive_desc_create<__VA_ARGS__>
 * ```
 *
 * (XXX include an example verbose func later)
 */
#if 1
#define DEFAULT_INSTANCE_CREATOR(...) \
    &primitive_desc_t::create<__VA_ARGS__::pd_t>,
#else /* show a verbose mode one, for debugging, here */
#endif

/// @addtogroup impl_list_support Implementations included at different ISA levels.
/**
 * Two interesting cmake \e build choices will be:
 *
 * - -DDNNL_ISA=VANILLA (no x86 jit, also a start point for non-x86 cpu), or
 * - -DDNNL_ISA=ALL     [default, support jit production for all x86 cpus]
 *    - x86:     \e max compile-time jit level = AVX512, lower available too.
 *    - non-x86: some other "max feature set" of implementations.
 *
 * The implementation lists in libdnnl can then be tuned to include \e no
 * or \e all jit impls (or in between, for expert use, but runtime dispatch
 * (XXX doc link here) is the preferred way.
 *
 * \ref cmake/options.cmake
 * \detail
 * Reference or other "all-C/C++" implementations should use the
 * \c INSTANCE macro.  Jit impls \e requiring jit isa \e foo' \b must use an
 * \c INSTANCE_foo macro.  This keeps impl lists in a terse format, and
 * allows compile time pruning (particularly all the way to VANILLA).
 *
 * \note because INSTANCE macros can be empty, do not append a comma
 * \note only a few \em coarse-grained INSTANCE_foo macros are available,
 * (sse41, avx2, avx512, uni)
 *
 * Coarse-grained cmake \b DNNL_ISA_foo constants defined in \c dnnl_config.h
 * are \b unrelated to \c cpu_isa_t or \c src/cpu/cpu_isa_traits.hpp .
 *
 * For readability in reference impls, it is conveniet to introduce
 * \e DNNL_TARGET_bar 0/1 values, rather than testing various DNNL_ISA values
 * (subject to change).
 */
//@{

/** Ref impls [no jit, maximum portability to new cpus] should use this.
 * JIT impl macros will look like INSTANCE_avx2(jit_avx2_layer_fwd_t). */
#define INSTANCE(...) INSTANCE_CREATOR(__VA_ARGS__)

// different avx512 types not distinguished
#if DNNL_ISA >= DNNL_ISA_AVX512_MIC && DNNL_ISA <= DNNL_ISA_X86_FULL
//#warning "cpu_engine WITH _avx512"
#define INSTANCE_avx512(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_avx512(...)
#endif

#if DNNL_ISA >= DNNL_ISA_AVX2 && DNNL_ISA <= DNNL_ISA_X86_FULL
//#warning "cpu_engine WITH _avx2"
#define INSTANCE_avx2(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_avx2(...)
#endif

#if DNNL_ISA >= DNNL_ISA_AVX && DNNL_ISA <= DNNL_ISA_X86_FULL
//#warning "cpu_engine WITH _avx"
#define INSTANCE_avx(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_avx(...)
#endif

// no x86 jit specific to sse42

#if DNNL_ISA >= DNNL_ISA_SSE41 && DNNL_ISA <= DNNL_ISA_X86_FUL_FULL
//#warning "cpu_engine WITH _sse41"
#define INSTANCE_sse41(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_sse41(...)
#endif

/** _uni CAN run on any ISA, but at runtime will choose highest available.
 * This circumvents compile-time attempts to limit the ISA -- the runtime
 * limit via environment variable DNNL_MAX_CPU_ISA should still work.
 * Do we need INSTANCE_uni? or are all uni drivers now templated on ISA?
 *
 * XXX CHECKME this is not equiv to sse41 (does not strictly mandate) ??
 */
#if DNNL_ISA >= DNNL_ISA_X86 && DNNL_ISA <= DNNL_ISA_X86_FULL
//#warning "cpu_engine WITH _uni"
#define INSTANCE_uni(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_uni(...)
#endif

// Add implementations for non-x86 CPU here...
#if 0 /*placeholder*/
/** an implementation ONLY working for VE (Aurora) vector processor. */
#if DNNL_ISA >= DNNL_ISA_VE && DNNL_ISA <= DNNL_ISA_VE_FULL
#define INSTANCE_ve(...) INSTANCE_CREATOR(__VA_ARGS__)
#else
#define INSTANCE_ve(...)
#endif
#endif

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
