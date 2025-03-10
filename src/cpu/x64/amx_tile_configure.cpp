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

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_amx_tilecfg_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_tilecfg_t)

    // TODO: Need to check status
    jit_amx_tilecfg_t(bool lazy = false)
        : jit_generator(jit_name(), avx512_core_amx), is_lazy_(lazy) {
        create_kernel();
    }

    void tile_configure(const char *palette) const { (*this)(palette); }
    // TODO: merge into a single call. Keep both versions for now until there's
    // a clear path lazy initialization API used across the library.
    void tile_lazy_configure(
            const char *new_palette, const char *old_palette) const {
        (*this)(new_palette, old_palette);
    }

private:
    // Lazy initialization first checks if tile is configured on a core. If it
    // is, and the palette loaded is same as palette provided by user, then the
    // tileload instruction is skipped.
    // According to measurements, the impact on performance is marginal compared
    // to manual handling of when palette should be loaded.
    bool is_lazy_;

    void generate() override {
        if (is_lazy_) {
            Xbyak::Label skip_tilecfg;
            // Store currect tilecfg into `old_palette`.
            sttilecfg(ptr[abi_param2]);
            // Move tilecfg into Zmm for further comparison.
            vmovdqu64(Xbyak::Zmm(0), ptr[abi_param2]);
            // Sets `1` per word, 32 words total for Zmms, if values are equal.
            vpcmpeqw(Xbyak::Opmask(0), Xbyak::Zmm(0), ptr[abi_param1]);
            // `kortestd` will set CF=1 if all `1` in the mask. Double word
            // takes 32 bits to compare.
            kortestd(Xbyak::Opmask(0), Xbyak::Opmask(0));
            // Checks if CF=1. If it is, everything matched, skipping config...
            jc(skip_tilecfg, T_NEAR);
            // ... otherwise, configure tile with user palette.
            ldtilecfg(ptr[abi_param1]);
            L(skip_tilecfg);
            ret();
        } else {
            ldtilecfg(ptr[abi_param1]);
            ret();
        }
    }
};

struct jit_amx_tilerelease_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_tilerelease_t)

    // TODO: Need to check status
    jit_amx_tilerelease_t() : jit_generator(jit_name(), avx512_core_amx) {
        create_kernel();
    }

    void tile_release() const { (*this)(); }

private:
    void generate() override {
        tilerelease();
        ret();
    }
};

status_t amx_tile_configure(const char palette[AMX_PALETTE_SIZE]) {
    static const jit_amx_tilecfg_t tilecfg(/* is_lazy = */ false);
    tilecfg.tile_configure(palette);
    return status::success;
};

status_t amx_tile_lazy_configure(const char palette[AMX_PALETTE_SIZE]) {
    static const jit_amx_tilecfg_t tilecfg(/* is_lazy = */ true);
    // Must be a per-thread storage to avoid writing race condition if used as
    // a member of `jit_amx_tilecfg_t` class.
    char palette_storage[AMX_PALETTE_SIZE];
    tilecfg.tile_lazy_configure(palette, palette_storage);
    return status::success;
};

status_t amx_tile_release() {
    static const jit_amx_tilerelease_t tilerls;
    tilerls.tile_release();
    return status::success;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
