/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/jit/conv/block_helper.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper class to assign block sizes for problem dimensions according to BMNK
// block sizes.
class block_assigner_t {
public:
    block_assigner_t(const dim_info_t &bmnk_dim,
            const std::vector<dim_info_t *> &prb_dims)
        : bmnk_dim_(bmnk_dim), prb_dims_(prb_dims) {
        move_to_next_bmnk_level();
        move_to_next_prb_dim();
    }

    bool has_blocks() const {
        if (utils::one_of(level_, tile_level_t::unknown, tile_level_t::_last))
            return false;
        if (prb_dim_idx_ >= (int)prb_dims_.size()) return false;
        return true;
    }

    void assign_block() {
        ir_assert(has_blocks());
        ir_assert(rem_bmnk_dim_.is_unlimited() || rem_bmnk_dim_ > 1);
        ir_assert(rem_prb_dim_ > 1);

        // Try a shortcut path to assign all dimensions to the current tile
        // level at once.
        if (try_assign_multi_blocks()) return;

        dim_value_t target_dim
                = min(rem_bmnk_dim_, prb_dims_[prb_dim_idx_]->max_dim(level_));
        int dim = compute_next_block(level_, prb_level_dim(), target_dim,
                bmnk_dim_.base_iter_block(), rem_prb_dim_,
                prb_dims_[prb_dim_idx_]->base_iter_block(), is_last_prb_dim());
        if (level_ == tile_level_t::iter) {
            ir_assert(dim % prb_dims_[prb_dim_idx_]->base_iter_block() == 0);
        }

        // Assign the computed block size to the current problem dimension.
        prb_dims_[prb_dim_idx_]->set_dim(level_, dim);

        // Update the remaining dimensions.
        if (!rem_bmnk_dim_.is_unlimited()) rem_bmnk_dim_ = rem_bmnk_dim_ / dim;
        rem_prb_dim_ = utils::div_up(rem_prb_dim_, dim);

        ir_assert(rem_bmnk_dim_.is_unlimited() || rem_bmnk_dim_ >= 1);
        ir_assert(rem_prb_dim_ >= 1);

        // Move to the next BMNK tile or next problem dimension (depending on
        // split/fuse settings and remaining sizes).
        if (rem_bmnk_dim_ != 1 && rem_prb_dim_ != 1) {
            if (allow_fuse()) {
                rem_prb_dim_ = 1;
            } else if (prb_dims_[prb_dim_idx_]->allow_split()) {
                rem_bmnk_dim_ = 1;
            } else {
                rem_prb_dim_ = 1;
                rem_bmnk_dim_ = 1;
            }
        }

        if (rem_bmnk_dim_ == 1 && rem_prb_dim_ == 1) {
            move_to_next_bmnk_level();
            move_to_next_prb_dim();
            return;
        }

        if (rem_bmnk_dim_ == 1) {
            move_to_next_bmnk_level();
            if (!prb_dims_[prb_dim_idx_]->allow_split()) {
                move_to_next_prb_dim();
            }
            return;
        }
        if (rem_prb_dim_ == 1) {
            move_to_next_prb_dim();
            if (!allow_fuse()) move_to_next_bmnk_level();
            return;
        }
        ir_error_not_expected();
    }

private:
    bool allow_fuse() const {
        for (auto *d : prb_dims_)
            if (!d->allow_fuse()) return false;
        return true;
    }

    void move_to_next_bmnk_level() {
        bool found = false;
        int l_beg = (int)level_ + 1;
        int l_end = (int)tile_level_t::_last;
        for (int l = l_beg; l < l_end; l++) {
            tile_level_t level = (tile_level_t)l;
            if (bmnk_dim_.dim(level) != 1) {
                found = true;
                level_ = level;
                rem_bmnk_dim_ = bmnk_dim_.dim(level);
                break;
            }
        }
        if (!found) level_ = tile_level_t::_last;
    }

    void move_to_next_prb_dim() {
        bool found = false;
        for (int i = prb_dim_idx_ + 1; i < (int)prb_dims_.size(); i++) {
            if (prb_dims_[i]->size() != 1) {
                found = true;
                prb_dim_idx_ = i;
                rem_prb_dim_ = prb_dims_[i]->size();
                break;
            }
        }
        if (!found) prb_dim_idx_ = (int)prb_dims_.size();
    }

    bool is_last_prb_dim() const {
        if (!allow_fuse()) return true;
        for (int i = prb_dim_idx_ + 1; i < (int)prb_dims_.size(); i++) {
            auto *d = prb_dims_[i];
            int max_dim = min(d->size(), d->max_dim(level_));
            if (max_dim != 1) return false;
        }
        return true;
    }

    int prb_level_dim() const {
        int ret = 1;
        for (auto *d : prb_dims_)
            ret *= d->dim(level_);
        return ret;
    }

    int compute_next_block(tile_level_t level, int level_dim,
            dim_value_t target_dim, int target_base_blk, int dim,
            int base_iter_block, bool is_last_dim,
            double target_eff = 0.75) const {
        if (target_dim.is_unlimited()) return dim;

        bool require_pow_2 = false;
        if (level == tile_level_t::tg) require_pow_2 = true;
        if (level == tile_level_t::iter && utils::one_of(bmnk_dim_.bmnk(), 'N'))
            require_pow_2 = true;

        int step = 1;
        int rem_target_base_blk = 1;
        if (level == tile_level_t::iter) {
            rem_target_base_blk
                    = target_base_blk / math::gcd(level_dim, target_base_blk);
            step = base_iter_block;
            ir_assert(rem_target_base_blk % base_iter_block == 0);
            if (is_last_dim) step = rem_target_base_blk;
        }

        int ret = utils::rnd_dn(target_dim, step);
        while (ret >= step) {
            bool ok = true;
            if (require_pow_2 && !math::is_pow2(ret)) ok = false;
            if (!is_last_dim) {
                if (ret % rem_target_base_blk != 0
                        && rem_target_base_blk % ret != 0)
                    ok = false;
            }
            if (ok) {
                int dim_padded = utils::rnd_up(dim, ret);
                double eff = (double)dim / dim_padded;
                if (eff >= target_eff) break;
            }
            ret -= step;
        }
        if (ret == 0) ret = step;
        if (require_pow_2) ir_assert(math::is_pow2(ret));
        if (level == tile_level_t::iter) ir_assert(ret % base_iter_block == 0);
        return ret;
    }

    bool is_thr_unlimited() const {
        if (level_ != tile_level_t::thr) return false;
        if (rem_bmnk_dim_.is_unlimited()) return false;
        if (bmnk_dim_.tg_dim() != 1) return false;
        return true;
    }

    bool is_iter_full_match() const {
        if (level_ != tile_level_t::iter) return false;

        int prb_total = 1;
        for (auto *d : prb_dims_) {
            if (d->iter_dim() != 1) return false;
            int max_iter_dim = min(d->size(), d->max_dim(tile_level_t::iter));
            prb_total *= max_iter_dim;
        }

        if (rem_bmnk_dim_ != prb_total) return false;
        return true;
    }

    bool try_assign_multi_blocks() {
        // Check restrictions to apply the heuristics.
        if (!allow_fuse()) return false;
        if (!is_thr_unlimited() && !is_iter_full_match()) return false;

        int nprb_dims = (int)prb_dims_.size();
        std::vector<int> rem_dims(nprb_dims, 1);
        std::vector<int> dims(nprb_dims, 1);

        int max_total_dim = 1;
        for (int i = prb_dim_idx_; i < nprb_dims; i++) {
            int dim = prb_dims_[i]->size();
            int rem_dim = (i == prb_dim_idx_) ? (int)rem_prb_dim_ : dim;
            rem_dim = min(rem_dim, prb_dims_[i]->max_dim(level_));
            rem_dims[i] = rem_dim;
            max_total_dim *= rem_dim;
        }

        bool found = false;
        std::function<void(int, int, double, double)> step;
        step = [&](int idx, int total_dim, double eff, double target_eff) {
            if (total_dim > rem_bmnk_dim_) return;
            if (eff < target_eff) return;
            if (idx == nprb_dims) {
                double min_dim_ratio = 0.5;
                double dim_ratio = total_dim / (double)rem_bmnk_dim_;
                // If all available dimensions are assigned, skip any checks.
                if (total_dim != max_total_dim) {
                    // Skip if the full dimension is too small relative to the
                    // target size.
                    if (dim_ratio < min_dim_ratio) return;
                    // Skip if the padding due to blocking is too large.
                    if (eff < target_eff) return;
                }
                // Found good blocking, set the flag.
                found = true;
                return;
            }
            int dim = prb_dims_[idx]->size();
            int rem_dim = rem_dims[idx];
            for (int blk = rem_dim; blk >= 1; blk--) {
                int dim_padded = utils::rnd_up(dim, blk);
                double dim_eff = (double)dim / dim_padded;
                dims[idx] = blk;
                step(idx + 1, total_dim * blk, eff * dim_eff, target_eff);
                if (found) break;
            }
        };

        if (level_ == tile_level_t::iter) {
            // is_iter_full_match() returned true so all dimensions can be
            // assigned as is.
            ir_assert(rem_bmnk_dim_ == max_total_dim);
            for (int i = prb_dim_idx_; i < nprb_dims; i++) {
                dims[i] = rem_dims[i];
            }
            found = true;
        } else {
            // Try to target different efficiencies until a good blocking is
            // found.
            for (double eff = 1.0; eff >= 0.5; eff -= 0.05) {
                step(prb_dim_idx_, 1, 1.0, eff);
                if (found) break;
            }
        }

        ir_assert(found) << "Can't assign blocks.";
        for (int i = prb_dim_idx_; i < nprb_dims; i++) {
            prb_dims_[i]->set_dim(level_, dims[i]);
        }

        prb_dim_idx_ = nprb_dims;
        return true;
    }

    tile_level_t level_ = tile_level_t::unknown;

    dim_info_t bmnk_dim_;
    dim_value_t rem_bmnk_dim_;

    std::vector<dim_info_t *> prb_dims_;
    int prb_dim_idx_ = -1;
    dim_value_t rem_prb_dim_ = 0;
};

void block_helper_t::compute() {
    is_frozen_ = true;

    ir_assert(vectorize_by_b() || vectorize_by_n());

    init_bmnk_dims();
    init_bmnk_blocks();
    init_prb_blocks();

#ifdef GEN_CONV_DEBUG
    for (auto &kv : dims_) {
        auto &d = kv.second;
        const char *tags[] = {"iter", "thr", "tg"};
        for (int i = min_tile_level_idx; i <= max_tile_level_idx; i++) {
            auto level = (tile_level_t)i;
            std::string env_name
                    = d.name() + "_" + tags[i - min_tile_level_idx] + "_dim";
            int env_dim = getenv_int(env_name.c_str(), -1);
            if (env_dim != -1) d.set_dim(level, env_dim);
        }
    }
#endif

    // Verify blocks.
    for (auto &kv : dims_) {
        auto &d = kv.second;
        ir_assert(d.iter_dim() % d.base_iter_block() == 0);
        for (int i = min_tile_level_idx; i <= max_tile_level_idx; i++) {
            auto level = (tile_level_t)i;
            auto max_dim = d.max_dim(level);
            ir_assert(max_dim.is_unlimited() || d.dim(level) <= max_dim);
        }
    }

    for (char bmnk : {'B', 'M', 'N', 'K'}) {
        int iter_blk = 1;
        for (auto &kv : dims_) {
            auto &d = kv.second;
            if (d.bmnk() != bmnk) continue;
            iter_blk *= d.iter_dim();
        }
        ir_assert(iter_blk % bmnk_dim(bmnk).base_iter_block() == 0);
    }

    // Initialize padded dim and iter block sizes.
    for (auto &kv : dims_) {
        auto &d = kv.second;
        padded_dim_sizes_.emplace(d.name(), d.padded_size());
        iter_blocks_.emplace(d.name(), d.iter_dim());
    }
}

void block_helper_t::init_bmnk_blocks() {
    int m_blk = 0;
    int k_blk = 0;
    int bn_blk = 0;
    int m_inst_blk = 0;
    int k_inst_blk = 0;
    int bn_inst_blk = 0;
    bool is_ge_hpc = (hw_cfg_.hw() >= ngen::HW::XeHPC);
    bool reduce_m_block = false;
    if (reduce_m_block_hint_set_) {
        reduce_m_block = reduce_m_block_hint_;
    } else {
        if (m_dim().base_iter_block() == 1) reduce_m_block = true;
        if (k_dim().base_iter_block() == 1) reduce_m_block = true;
    }
    if (is_tf32() && fma_kind_ != fma_kind_t::mad) reduce_m_block = true;
    int eu_thr_mul = (!is_ge_hpc && reduce_m_block) ? 2 : 4;
#ifdef GEN_CONV_DEBUG
    eu_thr_mul = getenv_int("eu_thr_mul", eu_thr_mul);
#endif
    auto &bn_dim = (vectorize_by_b() ? b_dim() : n_dim());
    switch (fma_kind_) {
        case fma_kind_t::mad: {
            int max_m_iter_dim = prb_max_dim('M', tile_level_t::iter);
            m_inst_blk = std::min(8, utils::rnd_down_pow2(max_m_iter_dim));
            bn_inst_blk = hw_cfg_.vec_size();
            k_inst_blk = 1;
            bool use_small_m_block = hw_cfg_.hw() <= ngen::HW::XeHP
                    && m_dim().base_iter_block() == 1;
            m_blk = (is_x8x8s32() || use_small_m_block ? 8 : 16);
            bool small_m_tg = m_dim().base_iter_block() == 1
                    && hw_cfg_.hw() == ngen::HW::XeHPG
                    && !m_dim().pref_tg_block();
            if (!m_dim().pref_tg_block())
                m_dim().set_max_dim(tile_level_t::tg, small_m_tg ? 1 : 4);
            bn_blk = hw_cfg_.vec_size();
            k_blk = compute_mad_k_block();
            if (!allow_k_grid_slicing_ && !allow_k_tg_slicing_) {
                do {
                    int est_bmn_threads = 1;
                    est_bmn_threads *= utils::div_up(m_dim().size(), m_blk);
                    est_bmn_threads *= utils::div_up(bn_dim.size(), bn_blk);
                    if (est_bmn_threads >= eu_thr_mul * hw_cfg_.eu_count())
                        break;
                    m_blk /= 2;
                    m_inst_blk = std::min(m_inst_blk, m_blk);
                } while (m_blk > 1);
            }
            break;
        }
        case fma_kind_t::dp4a:
        case fma_kind_t::dpas:
        case fma_kind_t::dpasw: {
            ir_assert(vectorize_by_n())
                    << "dpas can support N vectorization only.";
            int max_iter_dim = prb_max_dim('M', tile_level_t::iter);
            int target_m_blk = reduce_m_block ? 16 : 32;
            if (max_iter_dim % target_m_blk != 0 && max_iter_dim > 32) {
                float max_utilization_rate = 0.;
                for (int i
                        = min(32, utils::rnd_dn((int)(1.5 * target_m_blk), 4));
                        i > target_m_blk; i -= 4) {
                    float utilization_rate = (float)max_iter_dim
                            / utils::rnd_up(max_iter_dim, i);
                    // Heuristic constant preferring larger blocks, experimentally determined.
                    const float threshhold = 1.05;
                    if (utilization_rate > threshhold * max_utilization_rate) {
                        max_utilization_rate = utilization_rate;
                        target_m_blk = i;
                    }
                }
            }
            m_blk = target_m_blk;
            m_inst_blk = m_blk % 8 == 0 ? 8 : m_blk;
            bn_inst_blk = 8;
            k_inst_blk = is_x8x8s32() ? 32 : 16;
            bn_blk = is_ge_hpc ? 64 : 32;
            k_blk = k_inst_blk;
            break;
        }
        default: ir_error_not_expected();
    }

    m_blk = math::lcm(m_blk, m_dim().base_iter_block());
    k_blk = math::lcm(k_blk, k_dim().base_iter_block());
    bn_blk = math::lcm(bn_blk, bn_dim.base_iter_block());

    // Shrink block sizes to leverage is_iter_full_match() when applicable.
    const char bmnks[] = {'M', 'K', vectorize_by_b() ? 'B' : 'N'};
    int *blocks[] = {&m_blk, &k_blk, &bn_blk};
    int inst_blocks[] = {m_inst_blk, k_inst_blk, bn_inst_blk};
    for (int i = 0; i < 3; i++) {
        int max_iter_dim = prb_max_dim(bmnks[i], tile_level_t::iter);
        int base_blk = bmnk_dim(bmnks[i]).base_iter_block();
        int &blk = *blocks[i];
        int inst_blk = inst_blocks[i];
        if (max_iter_dim % base_blk == 0 && max_iter_dim % inst_blk == 0) {
            blk = std::min(blk, max_iter_dim);
        }
        ir_assert(blk % inst_blk == 0);
    }

    // Pad base iteration blocks according to instruction blocks.
    for (char bmnk : {'B', 'M', 'N', 'K'}) {
        bool is_bn = utils::one_of(bmnk, 'B', 'N');
        auto &d = bmnk_dim(bmnk);
        if (is_bn && !vectorize_by_bmnk(bmnk)) continue;

        int blk = d.base_iter_block();
        int inst_blk
                = is_bn ? bn_inst_blk : (bmnk == 'M') ? m_inst_blk : k_inst_blk;
        d.set_base_iter_block(math::lcm(blk, inst_blk));
    }

    m_blk = compute_block(m_dim().size(), m_blk, m_dim().base_iter_block());
    k_blk = compute_block(k_dim().size(), k_blk, k_dim().base_iter_block());
    bn_blk = compute_block(bn_dim.size(), bn_blk, bn_dim.base_iter_block());

#ifdef GEN_CONV_DEBUG
    m_blk = getenv_int("m_iter_blk", m_blk);
    k_blk = getenv_int("k_iter_blk", k_blk);
    bn_blk = getenv_int("bn_iter_blk", bn_blk);
    if (vectorize_by_b()) {
        bn_blk = getenv_int("b_iter_blk", bn_blk);
    } else {
        bn_blk = getenv_int("n_iter_blk", bn_blk);
    }
#endif
    m_dim().set_iter_dim(m_blk);
    bn_dim.set_iter_dim(bn_blk);
    k_dim().set_iter_dim(k_blk);

    init_k_blocking();

    for (char bmnk : {'B', 'M', 'N', 'K'}) {
        auto &d = bmnk_dim(bmnk);
        ir_assert(d.iter_dim() % d.base_iter_block() == 0
                || d.base_iter_block() % d.iter_dim() == 0);
    }

    // Set thread group blocks.
    bool with_k_tg_slicing = (k_dim().tg_dim() > 1);
    if (!with_k_tg_slicing) {
        int target_tg_size = hw_cfg_.max_tg_size();
        int est_threads = 1;
        for (char bmnk : {'B', 'M', 'N'})
            est_threads *= bmnk_dim(bmnk).grid_dim();
        if (est_threads < 2 * hw_cfg_.eu_count()
                || hw_cfg_.hw() >= ngen::HW::XeHPC) {
            target_tg_size = std::min(target_tg_size, 16);
        }

        // Compute max thread group blocks, independently for each dimension.
        std::vector<char> tg_bmnks = {vectorize_by_b() ? 'B' : 'N', 'M'};
        std::vector<int> tg_dims(tg_bmnks.size(), 1);
        bool any_pref_dim = false;
        int *split_dim_idx;
        for (size_t i = 0; i < tg_bmnks.size(); i++) {
            auto &d = bmnk_dim(tg_bmnks[i]);
            int i_max_tg_dim = min(target_tg_size, d.max_dim(tile_level_t::tg));
            int target_blk = i_max_tg_dim * d.iter_dim();
            int tg_dim = compute_block(d.size(), target_blk, d.iter_dim())
                    / d.iter_dim();
            //restrict maximum single tg dim as max_tg size is reduced
            tg_dim = std::min(utils::rnd_down_pow2(tg_dim),
                    hw_cfg_.max_tg_overriden() ? target_tg_size
                                    / (hw_cfg_.hw() >= ngen::HW::XeHPC ? 2 : 4)
                                               : hw_cfg_.simd_size());
            tg_dims[i] = tg_dim;
            if (d.pref_tg_block()) {
                //only one preferred dim allowed
                assert(!any_pref_dim);
                any_pref_dim = true;
            } else {
                split_dim_idx = &tg_dims.at(i);
            }
        }

        auto total_tg_dim = [&]() {
            return std::accumulate(
                    tg_dims.begin(), tg_dims.end(), 1, std::multiplies<int>());
        };
        auto max_tg_dim = [&]() -> int & {
            if (any_pref_dim && *split_dim_idx > 1) return *split_dim_idx;
            return *std::max_element(tg_dims.begin(), tg_dims.end());
        };

        // Reduce thread group size until it fits the target size.
        while (total_tg_dim() > target_tg_size) {
            max_tg_dim() /= 2;
        }

        for (size_t i = 0; i < tg_bmnks.size(); i++) {
            auto &d = bmnk_dim(tg_bmnks[i]);
            d.set_tg_dim(tg_dims[i]);
        }
    }
}

void block_helper_t::init_k_blocking() {
    // Thread and thread group dims must not be set yet.
    for (char bmnk : {'B', 'M', 'N', 'K'}) {
        auto &d = bmnk_dim(bmnk);
        ir_assert(d.thr_dim() == 1);
        ir_assert(d.tg_dim() == 1);
    }

    if (allow_k_grid_slicing_) {
        int est_threads = 1;
        for (char bmnk : {'B', 'M', 'N', 'K'})
            est_threads *= bmnk_dim(bmnk).grid_dim();
        int def_k_thr_dim = utils::div_up(est_threads, 2 * hw_cfg_.eu_count());
        def_k_thr_dim = std::min(100, def_k_thr_dim);
        def_k_thr_dim = std::max(1, def_k_thr_dim);
        int k_thr_dim = def_k_thr_dim;
#ifdef GEN_CONV_DEBUG
        k_thr_dim = getenv_int("k_thr_dim", k_thr_dim);
#endif
        k_dim().set_thr_dim(k_thr_dim);
        return;
    }

    if (!enable_k_tg_slicing()) {
        k_dim().set_thr_dim(dim_value_t::unlimited());
        return;
    }

    int k_nblks = utils::div_up(
            prb_blocked_dim('K').size(), k_dim().base_iter_block());
    int tg_dim0 = min(hw_cfg_.max_tg_size(), k_dim().max_dim(tile_level_t::tg));
    for (int tg_dim = tg_dim0; tg_dim >= 1; tg_dim /= 2) {
        if (k_nblks % tg_dim == 0) {
            k_dim().set_thr_dim(k_nblks / tg_dim);
            k_dim().set_tg_dim(tg_dim);
            return;
        }
    }

    // Couldn't enable TG slicing.
    k_dim().set_thr_dim(dim_value_t::unlimited());
}

bool block_helper_t::enable_k_tg_slicing() const {
#ifdef GEN_CONV_DEBUG
    int env_value = getenv_int("enable_k_tg_slicing", -1);
    if (env_value != -1) return (bool)env_value;
#endif
    if (!allow_k_tg_slicing_) return false;

    if (m_dim().iter_dim() > 16) return false;

    // TG slicing is supported only when there is only one k dimension.
    if (prb_blocked_ndims('K') > 1) return false;

    // Do not enable TG slicing if there are enough non-K threads.
    int non_k_threads = 1;
    for (char bmnk : {'B', 'M', 'N'}) {
        auto &d = bmnk_dim(bmnk);
        non_k_threads *= d.grid_dim() * d.tg_dim();
    }
    if (non_k_threads >= hw_cfg_.eu_count()) return false;

    // Do not enable TG slicing if reduction is small.
    int k_nblks = utils::div_up(k_dim().size(), k_dim().base_iter_block());
    if (k_nblks < 16) return false;

    return true;
}

void block_helper_t::init_prb_blocks() {
    // Pad sizes to base block multiples.
    for (auto &kv : dims_) {
        auto &d = kv.second;
        d.set_size(utils::rnd_up(d.size(), d.base_iter_block()));
    }

    // Filter blocked dimensions and sort them according to their keys.
    std::vector<dim_info_t *> sorted_dims;
    for (auto &kv : dims_) {
        auto &d = kv.second;
        if (!d.is_blocked()) continue;
        sorted_dims.push_back(&d);
    }
    std::sort(sorted_dims.begin(), sorted_dims.end(),
            [](const dim_info_t *a, const dim_info_t *b) {
                if (a->order_key() == b->order_key()) {
                    return a->name().compare(b->name()) < 0;
                }
                return a->order_key() < b->order_key();
            });

    for (char bmnk : {'B', 'N', 'M', 'K'}) {
        std::vector<dim_info_t *> cur_dims;
        for (auto *d : sorted_dims) {
            if (d->bmnk() != bmnk) continue;
            cur_dims.push_back(d);
        }

        ir_assert(!cur_dims.empty());

        // Pad dimensions according to BMNK base block requirements.
        int max_iter_dim = prb_max_dim(bmnk, tile_level_t::iter);
        int base_blk = bmnk_dim(bmnk).base_iter_block();
        if (max_iter_dim == 1 && base_blk > 1) {
            ir_assert(cur_dims[0]->base_iter_block() == 1);
            cur_dims[0]->set_size(base_blk);
        }

        block_assigner_t assigner(bmnk_dim(bmnk), cur_dims);
        while (assigner.has_blocks()) {
            assigner.assign_block();
        }
    }
}

int block_helper_t::compute_mad_k_block() const {
    int k_base_blk = k_dim().base_iter_block();
    if (k_base_blk >= 16) return k_base_blk;

    bool is_fused = true;
    int k_blocked_size = 1;
    for (auto &kv : dims_) {
        auto &d = kv.second;
        if (!d.is_blocked()) continue;
        if (d.bmnk() != 'K') continue;
        k_blocked_size *= d.size();
        if (!d.allow_fuse()) is_fused = false;
    }

    if (!is_fused) return 16;

    int max_k_blk = 32;
    if ((k_blocked_size <= max_k_blk)
            && (k_blocked_size % k_dim().base_iter_block() == 0))
        return k_blocked_size;

    return 16;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
