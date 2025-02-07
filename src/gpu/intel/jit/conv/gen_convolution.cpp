/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#include "gpu/intel/jit/conv/gen_convolution.hpp"

#include <iostream>
#include <utility>

#include "common/primitive_desc_iface.hpp"
#include "oneapi/dnnl/dnnl.hpp"

#include "common/impl_registration.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "gpu/gpu_zero_points_conv.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "gpu/intel/jit/reorder/reorder_kernel.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/ocl/utils.hpp"

#include "gpu/intel/jit/conv/config.hpp"
#include "gpu/intel/jit/conv/conv_kernel.hpp"
#include "gpu/intel/jit/conv/tiler.hpp"
#include "gpu/intel/jit/conv/zero_out.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct conv_pd_data_t {
    conv_config_t pd_cfg;
    tensor_config_t tensor_cfg;
    std::vector<kernel_info_t> kernel_infos;
    std::shared_ptr<primitive_desc_t> zp_pd;
};

class gen_convolution_t {
public:
    static const int max_kernels = 16;

    template <typename T>
    static status_t init_pd(T *pd, impl::engine_t *engine) {
        try {
            using compute::compute_engine_t;
            auto *compute_engine = utils::downcast<compute_engine_t *>(engine);

            VDISPATCH_CONV_IC(compute_engine->mayiuse_ngen_kernels(),
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_CONV_IC(
                    pd->set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);

            conv_problem_t prb;
            CHECK(prb.init(engine, pd));

            // The IR generator hard-codes s32 as the type for problem parameters.
            VDISPATCH_CONV_IC(
                    INT_MAX > std::max({prb.mb, prb.ic, prb.id, prb.ih, prb.iw,
                            prb.oc, prb.od, prb.oh, prb.ow, prb.kd, prb.kh,
                            prb.kw, prb.sd, prb.sh, prb.sw, prb.pd, prb.ph,
                            prb.pw, prb.dd, prb.dh, prb.dw}),
                    VERBOSE_SHAPE_RESTRICTION);

            pd->data = std::make_shared<conv_pd_data_t>();
            CHECK(init_pd_time_cfg(
                    prb, pd->data->pd_cfg, engine, pd, &pd->attr_));

            if (pd->data->pd_cfg.zp_cfg().needs_src_reorder_precalc
                    || pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                primitive_attr_t attr;
                if (pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                    int mask = 0;
                    CHECK(pd->attr_.zero_points_.get(DNNL_ARG_SRC, &mask));
                    attr.zero_points_.set(DNNL_ARG_SRC, mask);
                    attr.post_ops_.append_eltwise(
                            1.f, alg_kind::eltwise_linear, -1.f, 0.f);
                }
                dim_t I[3], O[3], P[3], D[3];
                prepare_zp_precompute_conv(prb, I, O, P, D);
                CHECK(create_zp_precompute_conv_pd(pd->data->zp_pd, engine,
                        attr, pd->weights_md(), I, O, P, D, data_type::f32,
                        pd->get_prop_kind(),
                        !pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc));
                if (pd->data->pd_cfg.zp_cfg().needs_src_conv_precalc) {
                    auto scratchpad = pd->scratchpad_registry().registrar();
                    scratchpad.book(memory_tracking::names::key_nested_multiple,
                            pd->data->zp_pd->scratchpad_registry());
                }
            }

            pd->data->tensor_cfg = get_tensor_config(
                    pd->data->pd_cfg, zp_conv_md_in(*pd->data));
            pd->data->kernel_infos.reserve(max_kernels);
            CHECK(init_kernel_infos(pd));

            return status::success;
        } catch (std::runtime_error &err) {
            // If verbose is enabled, print the primitive case and rethrow the
            // exception.
            VERROR(primitive, gpu, "%s,%s", pd->info(engine), err.what());
            return status::runtime_error;
        }
    }

    gen_convolution_t() = default;

    template <typename T>
    status_t init(T *primitive, impl::engine_t *engine) {
        auto &data = *primitive->pd()->data;
        auto &tensor_cfg = data.tensor_cfg;
        auto tiler = std::make_shared<conv_tiler_t>(data.pd_cfg);

        if (primitive->cache_blob()) {
            int32_t version;
            CHECK(primitive->cache_blob().get_value(
                    (uint8_t *)&version, sizeof(version)));
            primitive->set_version(version);
        }

        bool ok = false;
        int max_tries = 100;
        conv_config_t cfg;
        layout_t zp_dst;
        if (data.zp_pd) zp_dst = layout_t(zp_conv_md_out(data), false);

        if (primitive->cache_blob()) {
            tiler->set_cur_version(primitive->version());
        }

        for (int try_iter = 0; try_iter < max_tries; try_iter++) {
            if (try_iter != 0 && !tiler->is_tuning_mode())
                tiler->move_next(cfg);
            try {
                cfg = data.pd_cfg;
                cfg.set_pd(
                        static_cast<const convolution_pd_t *>(primitive->pd()));
                cfg.set_tiler(tiler);
                CHECK(init_cfg(cfg, primitive));

                if (!tiler->is_grf_limit_ok(cfg)) continue;

                gpu_info() << "Configuration:";
                gpu_info() << cfg;

                init_nd_ranges(primitive, cfg);
                auto &kernel_infos = data.kernel_infos;

                // This absolutely HAS to be executed first if present,
                // since it adds its own version mark to the cache blob
                for (int i = 0; i < int(kernel_infos.size()); i++)
                    if (kernel_infos[i].id() == kernel_id_t::zp_precalc) {
                        gpu_assert(data.zp_pd);
                        CHECK(primitive->create_nested_primitive(
                                zp_prim_, data.zp_pd, engine));
                    }

                std::vector<compute::kernel_t> tmp_kernels;
                for (int i = 0; i < int(kernel_infos.size()); i++) {
                    auto &info = kernel_infos[i];
                    switch (info.id()) {
                        case kernel_id_t::convolution: {
                            tmp_kernels.push_back(make_kernel<conv_kernel_t>(
                                    primitive, /*register_kernel=*/false,
                                    engine, cfg, info,
                                    nd_ranges_[i].local_range(), zp_dst));
                            break;
                        }
                        case kernel_id_t::pre_reorder: {
                            reorder_config_t reorder_cfg(cfg.exec_cfg(),
                                    tensor_cfg.user_layout(info.arg_name(1)),
                                    tensor_cfg.compute_layout(
                                            info.arg_name(1)));
                            tmp_kernels.push_back(
                                    make_kernel<reorder_kernel_t>(primitive,
                                            /*register_kernel=*/false, engine,
                                            reorder_cfg, "conv_reorder", info,
                                            cfg.is_dpas_or_dpasw_fma()));
                            break;
                        }
                        case kernel_id_t::post_reorder: {
                            reorder_config_t reorder_cfg(cfg.exec_cfg(),
                                    tensor_cfg.compute_layout(info.arg_name(0)),
                                    tensor_cfg.user_layout(info.arg_name(0)));
                            tmp_kernels.push_back(
                                    make_kernel<reorder_kernel_t>(primitive,
                                            /*register_kernel=*/false, engine,
                                            reorder_cfg, "conv_reorder", info,
                                            cfg.is_dpas_or_dpasw_fma()));
                            break;
                        }
                        case kernel_id_t::zero_out:
                            if (can_skip_zero_out(info, cfg)) {
                                tmp_kernels.emplace_back();
                                continue;
                            }
                            tmp_kernels.push_back(
                                    make_kernel<zero_out_kernel_t>(primitive,
                                            /*register_kernel=*/false, engine,
                                            cfg.exec_cfg(), info,
                                            cfg.is_dpas_or_dpasw_fma(),
                                            engine));
                            break;

                        case kernel_id_t::zp_precalc:
                            tmp_kernels.emplace_back();
                            continue;

                        default: gpu_error_not_expected();
                    }
                    if (!tmp_kernels[i]) return status::runtime_error;
                }
                ok = true;
                primitive->set_version(tiler->cur_version());
                kernels_ = std::move(tmp_kernels);
                break;
            } catch (ngen::out_of_registers_exception &err) {
                if (handle_exception(
                            err, primitive, engine, try_iter, max_tries))
                    return status::runtime_error;
                tiler->notify_out_of_registers(cfg);
                continue;
            } catch (std::runtime_error &err) {
                if (handle_exception(
                            err, primitive, engine, try_iter, max_tries))
                    return status::runtime_error;
                continue;
            }
        }
        if (!ok) return status::runtime_error;
        gpu_assert(kernels_.size() == data.kernel_infos.size());
        CHECK(primitive->register_kernels(kernels_));

        conv_tiler_t::after_create_hook(cfg, primitive);
        return status::success;
    }

    template <typename T>
    status_t execute(const T *primitive, const exec_ctx_t &ctx) const {
        auto &data = *primitive->pd()->data;
        auto &kernel_infos = data.kernel_infos;

        conv_tiler_t::before_exec_hook(primitive, ctx.stream());

        int max_stage = 100;
        int nsubmitted = 0;
        int nkernels = int(kernel_infos.size());
        for (int stage = 0; stage < max_stage; stage++) {
            for (int i = 0; i < nkernels; i++) {
                auto &info = kernel_infos[i];
                if (info.stage_id() != stage) continue;

                if (kernels_[i]) {
                    std::vector<memory_storage_wrapper_t> storage_list;
                    info.init_memory_storage_list(storage_list, ctx, primitive);

                    compute::kernel_arg_list_t arg_list;
                    info.set_args(arg_list, storage_list);

                    CHECK(primitive->parallel_for(
                            ctx, nd_ranges_[i], kernels_[i], arg_list));
                } else if (info.id() == kernel_id_t::zp_precalc) {
                    auto scratchpad_arg = [&](std::unique_ptr<memory_t,
                                                      memory_deleter_t> &retn,
                                                  const std::string &name,
                                                  const memory_desc_t *md) {
                        auto s = ctx.get_scratchpad_grantor()
                                         .get_memory_storage(info.key(name));
                        return safe_ptr_assign(retn,
                                new memory_t(ctx.stream()->engine(), md,
                                        std::move(s)));
                    };
                    gpu_assert(zp_prim_);
                    std::unique_ptr<memory_t, memory_deleter_t> zp_src, zp_dst;
                    CHECK(scratchpad_arg(
                            zp_src, "src_zero_points", zp_conv_md_in(data)));
                    CHECK(scratchpad_arg(zp_dst, "dst", zp_conv_md_out(data)));

                    exec_args_t e_args;
                    auto src_zp_idx = DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC;
                    e_args[src_zp_idx] = ctx.args().at(src_zp_idx);
                    e_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_WEIGHTS);
                    e_args[DNNL_ARG_SRC] = memory_arg_t {zp_src.get(), true};
                    e_args[DNNL_ARG_DST] = memory_arg_t {zp_dst.get(), false};
                    exec_ctx_t e_ctx(ctx, std::move(e_args));
                    const auto nm = memory_tracking::names::key_nested_multiple;
                    nested_scratchpad_t ns(ctx, nm, zp_prim_);
                    e_ctx.set_scratchpad_grantor(ns.grantor());
                    CHECK(zp_prim_->execute(e_ctx));
                }
                nsubmitted++;
                if (nsubmitted == nkernels) break;
            }
        }

        return status::success;
    }

private:
    static const memory_desc_t *zp_conv_md_in(const conv_pd_data_t &data) {
        if (!data.zp_pd) return nullptr;
        const bool is_bwd_d
                = (data.zp_pd->get_prop_kind() == prop_kind::backward_data);
        return (is_bwd_d) ? data.zp_pd->diff_dst_md() : data.zp_pd->src_md();
    }

    static const memory_desc_t *zp_conv_md_out(const conv_pd_data_t &data) {
        if (!data.zp_pd) return nullptr;
        const bool is_bwd_d
                = (data.zp_pd->get_prop_kind() == prop_kind::backward_data);
        return (is_bwd_d) ? data.zp_pd->diff_src_md() : data.zp_pd->dst_md();
    }

    template <typename T>
    static kernel_info_t &create_kernel_info(T *pd, kernel_id_t kernel_id) {
        auto &infos = pd->data->kernel_infos;
        gpu_assert((int)infos.size() + 1 <= max_kernels);
        infos.emplace_back();
        auto &ret = infos.back();
        ret.set_id(kernel_id);
        return ret;
    }

    template <typename T>
    static status_t init_kernel_infos(T *pd) {
        auto &data = *pd->data;
        auto &cfg = data.pd_cfg;
        auto &conv_info = create_kernel_info(pd, kernel_id_t::convolution);
        auto &zp_precalc_info = (cfg.zp_cfg().needs_src_conv_precalc)
                ? create_kernel_info(pd, kernel_id_t::zp_precalc)
                : conv_info;

        static_assert(DNNL_ARG_UNDEF == memory_tracking::names::key_none,
                "Undefined argument and empty scratchpad key are out of sync!");

        // Initialize kernel arguments.
        int scratchpad_key = memory_tracking::names::key_none;
        for (auto &t : data.tensor_cfg.tensors()) {
            const bool wei_reorder_precalc = (t.name == "wei")
                    && cfg.zp_cfg().needs_src_reorder_precalc;
            const bool src_conv_precalc = (t.name == "src_zero_points")
                    && cfg.zp_cfg().needs_src_conv_precalc;

            const auto compute_buf = make_buffer(t.name);
            size_t compute_size = t.compute_layout.size();
            int compute_arg_key = t.arg_key;

            if (compute_arg_key == DNNL_ARG_UNDEF) {
                gpu_assert(!t.needs_reorder);
                gpu_assert(!t.needs_zero_out);
                gpu_error_not_expected();
                continue;
            }

            auto add_compute_arg = [&](kernel_info_t &ki, const expr_t &buf,
                                           bool is_input) {
                if (t.needs_reorder || src_conv_precalc)
                    ki.register_scratchpad_arg(
                            buf, compute_arg_key, is_input, compute_size);
                else
                    ki.register_user_arg(buf, compute_arg_key, is_input);
            };
            auto scratchpad_book = [&](int key) {
                pd->scratchpad_registry().registrar().book(into<uint32_t>(key),
                        compute_size, 1, ocl::OCL_BUFFER_ALIGNMENT);
            };
            auto create_zero_out_info = [&]() -> kernel_info_t & {
                auto &zero_out_info
                        = create_kernel_info(pd, kernel_id_t::zero_out);
                auto size_var = var_t::make(type_t::u32(), "size");
                zero_out_info.register_internal_arg(
                        size_var, into<uint32_t>(compute_size));
                zero_out_info.set_nd_range(zero_out_kernel_desc_t::nd_range(
                        cfg.simd(), compute_size));
                return zero_out_info;
            };

            if (t.needs_reorder || src_conv_precalc) {
                int user_arg_key = compute_arg_key;
                auto user_buf = make_buffer(t.name + "_user");
                compute_arg_key = ++scratchpad_key;

                if (!src_conv_precalc && t.is_input) {
                    auto &reorder_info
                            = create_kernel_info(pd, kernel_id_t::pre_reorder);
                    reorder_info.register_user_arg(user_buf, user_arg_key,
                            /*is_input=*/true);
                    add_compute_arg(reorder_info, compute_buf, false);
                    reorder_info.set_nd_range(reorder_kernel_t<>::nd_range(
                            cfg.exec_cfg(), t.user_layout, t.compute_layout));
                }
                if (!src_conv_precalc && t.is_output) {
                    auto &reorder_info
                            = create_kernel_info(pd, kernel_id_t::post_reorder);
                    add_compute_arg(reorder_info, compute_buf, true);
                    reorder_info.register_user_arg(user_buf, user_arg_key,
                            /*is_input=*/false);
                    reorder_info.set_nd_range(reorder_kernel_t<>::nd_range(
                            cfg.exec_cfg(), t.compute_layout, t.user_layout));
                }
                if (src_conv_precalc) {
                    scratchpad_book(++scratchpad_key);
                    create_zero_out_info().register_scratchpad_arg(compute_buf,
                            scratchpad_key, /*is_input=*/false, compute_size);

                    zp_precalc_info.register_scratchpad_arg(compute_buf,
                            scratchpad_key, /*is_input=*/true, compute_size);
                    const auto &dim = ir_utils::max_unique_pad_states;
                    const auto &prb = cfg.prb();
                    const dim_t KDD = (prb.kd - 1) * (prb.dd + 1) + 1;
                    const dim_t KDH = (prb.kh - 1) * (prb.dh + 1) + 1;
                    const dim_t KDW = (prb.kw - 1) * (prb.dw + 1) + 1;
                    compute_size = int64_t(compute_size) * sizeof(int32_t)
                            * dim(prb.od, prb.id, KDD, prb.pd, prb.sd, true)
                            * dim(prb.oh, prb.ih, KDH, prb.ph, prb.sh, true)
                            * dim(prb.ow, prb.iw, KDW, prb.pw, prb.sw, true)
                            * utils::rnd_up(prb.g * prb.oc, cfg.simd())
                            / std::min(KDD, prb.id) / std::min(KDH, prb.ih)
                            / std::min(KDW, prb.iw) / (prb.g * prb.ic);
                    add_compute_arg(zp_precalc_info, make_buffer("dst"), false);
                }
                scratchpad_book(compute_arg_key);
                if (wei_reorder_precalc) {
                    // user-supplied weights contain precomputed ZP values, so
                    // the buffer is to be passed to the conv alongside weights
                    conv_info.register_user_arg(
                            user_buf, user_arg_key, t.is_input && !t.is_output);
                }
            }
            if (t.needs_zero_out) {
                add_compute_arg(create_zero_out_info(), compute_buf, false);
            }
            add_compute_arg(conv_info, compute_buf, t.is_input && !t.is_output);
        }

        return status::success;
    }

    template <typename T>
    void init_nd_ranges(T *primitive, const conv_config_t &cfg) {
        auto *pd = primitive->pd();
        auto &data = *pd->data;
        int nkernels = int(data.kernel_infos.size());
        nd_ranges_.resize(nkernels);
        for (int i = 0; i < nkernels; i++) {
            auto &info = data.kernel_infos[i];
            switch (info.id()) {
                case kernel_id_t::convolution:
                    // Convolution kernel info is initialized at PD creation
                    // time when ND range/grid information are not known yet so
                    // we need to directly query config here.
                    nd_ranges_[i] = cfg.nd_range();
                    break;
                case kernel_id_t::pre_reorder:
                case kernel_id_t::post_reorder:
                case kernel_id_t::zero_out:
                    nd_ranges_[i] = info.nd_range();
                    break;
                case kernel_id_t::zp_precalc: break;
                default: gpu_error_not_expected();
            }
        }
    }

    static bool can_skip_zero_out(
            const kernel_info_t &info, const conv_config_t &cfg) {
        gpu_assert(info.id() == kernel_id_t::zero_out);
        auto &buf_name = info.arg_var(1).as<var_t>().name;
        if (buf_name == "wei") return cfg.can_skip_wei_zero_out();
        if (buf_name == "bia") return cfg.can_skip_bia_zero_out();
        return false;
    }

    template <typename ExceptionT, typename T>
    static bool handle_exception(const ExceptionT &err, T *primitive,
            impl::engine_t *engine, int iter, int max_iters) {
        if (iter + 1 < max_iters) return false;
        VERROR(primitive, gpu, "%s,%s", primitive->pd()->info(engine),
                err.what());
        return true;
    }

    std::vector<compute::kernel_t> kernels_;
    std::vector<compute::nd_range_t> nd_ranges_;
    std::shared_ptr<impl::primitive_t> zp_prim_;
};

status_t gen_convolution_fwd_t::pd_t::init(impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_fwd(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_fwd_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_data_t::pd_t::init(impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_bwd_d(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_weights_t::pd_t::init(impl::engine_t *engine) {
    VDISPATCH_CONV_IC(is_bwd_w(), VERBOSE_BAD_PROPKIND);
    CHECK(gen_convolution_t::init_pd(this, engine));
    return status::success;
}

status_t gen_convolution_bwd_data_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

status_t gen_convolution_bwd_weights_t::init(impl::engine_t *engine) {
    impl_.reset(new gen_convolution_t());
    return impl_->init(this, engine);
}

status_t gen_convolution_bwd_weights_t::execute(const exec_ctx_t &ctx) const {
    return impl_->execute(this, ctx);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
