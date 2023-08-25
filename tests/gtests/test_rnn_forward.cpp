/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#include <numeric>
#include <utility>
#include <type_traits>

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {

struct test_rnn_sizes_t {
    memory::dim l, d, t, mb;
    memory::dim slc, sic, dhc, dic;
};

struct test_rnn_formats_t {
    dnnl::memory::format_tag src_layer_fmt;
    dnnl::memory::format_tag src_iter_fmt;
    dnnl::memory::format_tag weights_layer_fmt;
    dnnl::memory::format_tag weights_iter_fmt;
    dnnl::memory::format_tag weights_peephole_fmt;
    dnnl::memory::format_tag weights_projection_fmt;
    dnnl::memory::format_tag bias_fmt;
    dnnl::memory::format_tag dst_layer_fmt;
    dnnl::memory::format_tag dst_iter_fmt;
};

struct test_rnn_extra_t {
    dnnl::algorithm activation;
    float alpha;
};

struct test_rnn_params_t {
    test_rnn_extra_t extra;
    prop_kind aprop;
    dnnl::rnn_direction direction;
    test_rnn_formats_t fmts;
    test_rnn_sizes_t sizes;
    bool expect_to_fail;
    dnnl_status_t expected_status;
};

// We assume uniform data type accross tensors for now
template <typename T, typename data_t>
class rnn_forward_test_t : public ::testing::TestWithParam<test_rnn_params_t> {

private:
    memory::dim getNGates();

    typename T::primitive_desc get_pd(prop_kind aprop, algorithm activation,
            rnn_direction direction, const memory::desc &src_layer_md,
            const memory::desc &src_iter_md, const memory::desc &src_iter_c_md,
            const memory::desc &attention_md,
            const memory::desc &weights_layer_md,
            const memory::desc &weights_iter_md,
            const memory::desc &weights_peephole_md,
            const memory::desc &weights_projection_md,
            const memory::desc &bias_md, const memory::desc &dst_layer_md,
            const memory::desc &dst_iter_md, const memory::desc &dst_iter_c_md,
            float alpha = 0.0f);

    bool skipTest(bool src_layer_match, bool augru_attention_match,
            bool src_iter_match, bool src_iter_c_match,
            bool weights_layer_match, bool weights_iter_match, bool bias_match,
            bool dst_layer_match, bool dst_iter_match, bool dst_iter_c_match) {
        // By default, we ignore src_iter_c and dst_iter_c as they are
        // only supported for lstm. For LSTM tests, this function
        // should be specialized to handle them.
        return src_layer_match && src_iter_match && weights_layer_match
                && weights_iter_match && bias_match && dst_layer_match
                && dst_iter_match;
    }

    memory::desc querySrcIterC(const typename T::primitive_desc &rpd) {
        return memory::desc();
    }

    memory::desc queryWeightsPeephole(const typename T::primitive_desc &rpd) {
        return memory::desc();
    }

    memory::desc queryWeightsProjection(const typename T::primitive_desc &rpd) {
        return memory::desc();
    }

    memory::desc queryDstIterC(const typename T::primitive_desc &rpd) {
        return memory::desc();
    }

    void testExecArgQueries(typename T::primitive_desc pd) {
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_LAYER)
                == pd.weights_layer_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_ITER)
                == pd.weights_iter_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PEEPHOLE)
                == pd.weights_peephole_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS_PROJECTION)
                == pd.weights_projection_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_BIAS)
                == pd.bias_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_LAYER)
                == pd.src_layer_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_AUGRU_ATTENTION)
                == pd.augru_attention_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER)
                == pd.src_iter_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_SRC_ITER_C)
                == querySrcIterC(pd));
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST_LAYER)
                == pd.dst_layer_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST_ITER)
                == pd.dst_iter_desc());
        ASSERT_TRUE(pd.query_md(query::exec_arg_md, DNNL_ARG_DST_ITER_C)
                == queryDstIterC(pd));
    };

    void test_primitive_param_queries(typename T::primitive_desc pd) {
        auto p = ::testing::TestWithParam<test_rnn_params_t>::GetParam();

        dnnl::algorithm expected_cell_kind = algorithm::undef;
        if (is_vanilla_rnn) expected_cell_kind = algorithm::vanilla_rnn;
        if (is_lstm) expected_cell_kind = algorithm::vanilla_lstm;
        if (is_gru) expected_cell_kind = algorithm::vanilla_gru;
        if (is_lbr_gru) expected_cell_kind = algorithm::lbr_gru;
        if (is_augru) expected_cell_kind = algorithm::vanilla_augru;
        if (is_lbr_augru) expected_cell_kind = algorithm::lbr_augru;

        ASSERT_NE(expected_cell_kind, algorithm::undef);
        ASSERT_EQ(pd.get_cell_kind(), expected_cell_kind);
        ASSERT_EQ(pd.get_prop_kind(), p.aprop);
        ASSERT_EQ(pd.get_direction(), p.direction);

        if (is_vanilla_rnn) {
            ASSERT_EQ(pd.get_alpha(), p.extra.alpha);
            ASSERT_EQ(pd.get_activation_kind(), p.extra.activation);
        } else {
            ASSERT_EQ(pd.get_alpha(), 0.0f);
            ASSERT_EQ(pd.get_beta(), 0.0f);
            ASSERT_EQ(pd.get_activation_kind(), algorithm::undef);
        }
    }

protected:
    static constexpr bool is_lstm = std::is_same<T, lstm_forward>::value;
    static constexpr bool is_vanilla_rnn
            = std::is_same<T, vanilla_rnn_forward>::value;
    static constexpr bool is_gru = std::is_same<T, gru_forward>::value;
    static constexpr bool is_lbr_gru = std::is_same<T, lbr_gru_forward>::value;
    static constexpr bool is_augru = std::is_same<T, augru_forward>::value;
    static constexpr bool is_lbr_augru
            = std::is_same<T, lbr_augru_forward>::value;

    void SetUp() override {
        auto p = ::testing::TestWithParam<test_rnn_params_t>::GetParam();
        catch_expected_failures(
                [&]() { Test(); }, p.expect_to_fail, p.expected_status, false);
    }

    void Test() {
        auto p = ::testing::TestWithParam<test_rnn_params_t>::GetParam();
        const bool is_lstm_peephole
                = p.fmts.weights_peephole_fmt != memory::format_tag::undef;
        const bool is_lstm_projection
                = p.fmts.weights_projection_fmt != memory::format_tag::undef;
        auto eng = get_test_engine();
        auto strm = make_stream(eng);
        //@todo check algorithm is one of the supported by RNN
        //ASSERT_EQ(p.aalgorithm, algorithm::vanilla_lstm);

        // Initialize the data
        memory::data_type prec = data_traits<data_t>::data_type;
        auto dims = p.sizes;
        auto t = dims.t, mb = dims.mb, l = dims.l, d = dims.d;
        auto slc = dims.slc, sic = dims.sic, dhc = dims.dhc, dic = dims.dic;
        auto dlc = (p.direction == rnn_direction::bidirectional_concat ? 2 : 1)
                * dic;
        memory::dim g = getNGates();
        memory::dim bias_extra_gate = std::is_same<T, lbr_gru_forward>::value
                        || std::is_same<T, lbr_augru_forward>::value
                ? 1
                : 0;

        auto attention_dims = {t, mb, (memory::dim)1};
        auto weights_layer_dims = {l, d, slc, g, dhc};
        auto weights_iter_dims = {l, d, sic, g, dhc};
        auto weights_peephole_dims = {l, d, (memory::dim)3, dhc};
        auto weights_projection_dims = {l, d, dhc, dic};
        auto bias_dims = {l, d, g + bias_extra_gate, dhc};
        auto src_layer_dims = {t, mb, slc};
        auto src_iter_dims = {l, d, mb, sic};
        auto src_iter_c_dims = {l, d, mb, dhc};
        auto dst_layer_dims = {t, mb, dlc};
        auto dst_iter_dims = {l, d, mb, dic};
        auto dst_iter_c_dims = {l, d, mb, dhc};

        auto attention_md_any
                = memory::desc({attention_dims}, prec, memory::format_tag::any);
        auto weights_layer_md_any = memory::desc(
                {weights_layer_dims}, prec, memory::format_tag::any);
        auto weights_iter_md_any = memory::desc(
                {weights_iter_dims}, prec, memory::format_tag::any);
        auto weights_peephole_md_any = memory::desc(
                {weights_peephole_dims}, prec, memory::format_tag::any);
        auto weights_projection_md_any = memory::desc(
                {weights_projection_dims}, prec, memory::format_tag::any);
        auto bias_md_any
                = memory::desc({bias_dims}, prec, memory::format_tag::any);
        auto src_layer_md_any
                = memory::desc({src_layer_dims}, prec, memory::format_tag::any);
        auto src_iter_md_any
                = memory::desc({src_iter_dims}, prec, memory::format_tag::any);
        auto src_iter_c_md_any = memory::desc(
                {src_iter_c_dims}, prec, memory::format_tag::any);
        auto dst_layer_md_any
                = memory::desc({dst_layer_dims}, prec, memory::format_tag::any);
        auto dst_iter_md_any
                = memory::desc({dst_iter_dims}, prec, memory::format_tag::any);
        auto dst_iter_c_md_any = memory::desc(
                {dst_iter_c_dims}, prec, memory::format_tag::any);

        auto attention_md_tgt = (is_augru || is_lbr_augru)
                ? memory::desc({attention_dims}, prec, memory::format_tag::tnc)
                : memory::desc();
        auto weights_layer_md_tgt = memory::desc(
                {weights_layer_dims}, prec, p.fmts.weights_layer_fmt);
        auto weights_iter_md_tgt = memory::desc(
                {weights_iter_dims}, prec, p.fmts.weights_iter_fmt);
        auto weights_peephole_md_tgt = is_lstm_peephole
                ? memory::desc({weights_peephole_dims}, prec,
                        p.fmts.weights_peephole_fmt)
                : memory::desc();
        auto weights_projection_md_tgt = is_lstm_projection
                ? memory::desc({weights_projection_dims}, prec,
                        p.fmts.weights_projection_fmt)
                : memory::desc();
        auto bias_md_tgt = memory::desc({bias_dims}, prec, p.fmts.bias_fmt);
        auto src_layer_md_tgt
                = memory::desc({src_layer_dims}, prec, p.fmts.src_layer_fmt);
        auto augru_attention_md_tgt
                = memory::desc({attention_dims}, prec, memory::format_tag::tnc);
        auto src_iter_md_tgt
                = (p.fmts.src_iter_fmt != memory::format_tag::undef)
                ? memory::desc({src_iter_dims}, prec, p.fmts.src_iter_fmt)
                : memory::desc();
        auto src_iter_c_md_tgt
                = (p.fmts.src_iter_fmt != memory::format_tag::undef)
                ? memory::desc({src_iter_c_dims}, prec, p.fmts.src_iter_fmt)
                : memory::desc();
        auto dst_layer_md_tgt
                = memory::desc({dst_layer_dims}, prec, p.fmts.dst_layer_fmt);
        auto dst_iter_md_tgt
                = (p.fmts.dst_iter_fmt != memory::format_tag::undef)
                ? memory::desc({dst_iter_dims}, prec, p.fmts.dst_iter_fmt)
                : memory::desc();
        auto dst_iter_c_md_tgt
                = (p.fmts.dst_iter_fmt != memory::format_tag::undef)
                ? memory::desc({dst_iter_c_dims}, prec, p.fmts.dst_iter_fmt)
                : memory::desc();

        auto weights_projection_md_ldio = memory::desc(
                {weights_projection_dims}, prec, memory::format_tag::ldio);

        // Create the reference primitive descriptor
        auto ref_pd = get_pd(p.aprop, p.extra.activation, p.direction,
                src_layer_md_any, src_iter_md_any, src_iter_c_md_any,
                attention_md_any, weights_layer_md_any, weights_iter_md_any,
                weights_peephole_md_any, weights_projection_md_any, bias_md_any,
                dst_layer_md_any, dst_iter_md_any, dst_iter_c_md_any,
                p.extra.alpha);
        // test construction from a C pd
        ref_pd = typename T::primitive_desc(ref_pd.get());
        testExecArgQueries(ref_pd);
        test_primitive_param_queries(ref_pd);

        // Query the descriptor for memory descriptors
        auto weights_layer_md_ref = ref_pd.weights_layer_desc();
        auto weights_iter_md_ref = ref_pd.weights_iter_desc();
        auto weights_peephole_md_ref = queryWeightsPeephole(ref_pd);
        auto weights_projection_md_ref = queryWeightsProjection(ref_pd);
        auto bias_md_ref = ref_pd.bias_desc();
        auto src_layer_md_ref = ref_pd.src_layer_desc();
        auto augru_attention_md_ref = ref_pd.augru_attention_desc();
        auto src_iter_md_ref = ref_pd.src_iter_desc();
        auto src_iter_c_md_ref = querySrcIterC(ref_pd);
        auto dst_layer_md_ref = ref_pd.dst_layer_desc();
        auto dst_iter_md_ref = ref_pd.dst_iter_desc();
        auto dst_iter_c_md_ref = queryDstIterC(ref_pd);

        if (skipTest(src_layer_md_ref == src_layer_md_tgt,
                    augru_attention_md_ref == augru_attention_md_tgt,
                    src_iter_md_ref == src_iter_md_tgt,
                    src_iter_c_md_ref == src_iter_c_md_tgt,
                    weights_layer_md_ref == weights_layer_md_tgt,
                    weights_iter_md_ref == weights_iter_md_tgt,
                    bias_md_ref == bias_md_tgt,
                    dst_layer_md_ref == dst_layer_md_tgt,
                    dst_iter_md_ref == dst_iter_md_tgt,
                    dst_iter_c_md_ref == dst_iter_c_md_tgt))
            return;

        /* initialize data */
        auto weights_layer_ref = test::make_memory(weights_layer_md_ref, eng);
        auto weights_iter_ref = test::make_memory(weights_iter_md_ref, eng);
        auto weights_peephole_ref
                = test::make_memory(weights_peephole_md_ref, eng);
        auto weights_projection_ref
                = test::make_memory(weights_projection_md_ref, eng);
        auto bias_ref = test::make_memory(bias_md_ref, eng);
        auto src_layer_ref = test::make_memory(src_layer_md_ref, eng);
        auto augru_attention_ref
                = test::make_memory(augru_attention_md_ref, eng);
        auto src_iter_ref = test::make_memory(src_iter_md_ref, eng);
        auto src_iter_c_ref = test::make_memory(src_iter_c_md_ref, eng);
        auto dst_layer_ref = test::make_memory(dst_layer_md_ref, eng);
        auto dst_iter_ref = test::make_memory(dst_iter_md_ref, eng);
        auto dst_iter_c_ref = test::make_memory(dst_iter_c_md_ref, eng);

        auto weights_layer_tgt = test::make_memory(weights_layer_md_tgt, eng);
        auto weights_iter_tgt = test::make_memory(weights_iter_md_tgt, eng);
        auto weights_peephole_tgt
                = test::make_memory(weights_peephole_md_tgt, eng);
        auto weights_projection_tgt
                = test::make_memory(weights_projection_md_tgt, eng);
        auto bias_tgt = test::make_memory(bias_md_tgt, eng);
        auto src_layer_tgt = test::make_memory(src_layer_md_tgt, eng);
        auto augru_attention_tgt
                = test::make_memory(augru_attention_md_tgt, eng);
        auto src_iter_tgt = test::make_memory(src_iter_md_tgt, eng);
        auto src_iter_c_tgt = test::make_memory(src_iter_c_md_tgt, eng);
        auto dst_layer_tgt = test::make_memory(dst_layer_md_tgt, eng);
        auto dst_iter_tgt = test::make_memory(dst_iter_md_tgt, eng);
        auto dst_iter_c_tgt = test::make_memory(dst_iter_c_md_tgt, eng);

        auto weights_projection_ldio = memory(weights_projection_md_ldio, eng);

        // Assumption: b is a plain layout
        auto init_tensor = [&](memory a, memory b, int scale = 1) {
            auto desc = a.get_desc();
            auto b_dims = desc.get_dims();
            auto n_elems = std::accumulate(b_dims.begin(), b_dims.end(),
                    size_t(1), std::multiplies<dnnl_dim_t>());
            const dnnl::impl::memory_desc_wrapper mdw(desc.get());
            {
                auto b_ptr = map_memory<float>(b);
                for (size_t i = 0; i < n_elems; i++)
                    b_ptr[i] = scale * i;
            }
            reorder(b, a).execute(strm, b, a);
            strm.wait();
        };
        auto init_zero_tensor = [&](const memory &a, memory::format_tag fmt) {
            auto desc = a.get_desc();
            memory::desc tmp_md(desc.get_dims(), desc.get_data_type(), fmt);
            auto tmp = test::make_memory(tmp_md, eng);
            // Zero fill the tmp tensor
            init_tensor(a, tmp, 0);
        };
        auto init_id_wights_projection = [&](memory &w_plain, memory &w_rnn) {
            auto w_plain_ptr = map_memory<float>(w_plain);
            for_(memory::dim l = 0; l < dims.l; ++l)
            for_(memory::dim d = 0; d < dims.d; ++d)
            for_(memory::dim i = 0; i < dims.dhc; ++i)
            for (memory::dim o = 0; o < dims.dic; ++o) {
                auto off = (((l * dims.d) + d) * dims.dhc + i) * dims.dic + o;
                w_plain_ptr[off] = (i == o) ? 1.f : 0.f;
            }

            reorder(w_plain, w_rnn).execute(strm, w_plain, w_rnn);
            strm.wait();
        };

        init_tensor(weights_layer_ref, weights_layer_tgt);
        init_tensor(weights_iter_ref, weights_iter_tgt);
        if (is_lstm_peephole)
            init_tensor(weights_peephole_ref, weights_peephole_tgt);
        else if (std::is_same<T, lstm_forward>::value)
            init_zero_tensor(weights_peephole_ref, memory::format_tag::ldgo);
        if (is_lstm_projection)
            init_tensor(weights_projection_ref, weights_projection_tgt);
        else if (std::is_same<T, lstm_forward>::value)
            init_id_wights_projection(
                    weights_projection_ldio, weights_projection_ref);
        init_tensor(bias_ref, bias_tgt);
        init_tensor(src_layer_ref, src_layer_tgt);
        if (is_augru || is_lbr_augru)
            init_tensor(augru_attention_ref, augru_attention_tgt);
        if (p.fmts.src_iter_fmt != memory::format_tag::undef) {
            init_tensor(src_iter_ref, src_iter_tgt);
            if (std::is_same<T, lstm_forward>::value)
                init_tensor(src_iter_c_ref, src_iter_c_tgt);
        } else {
            init_zero_tensor(src_iter_ref, memory::format_tag::ldnc);
            if (std::is_same<T, lstm_forward>::value)
                init_zero_tensor(src_iter_c_ref, memory::format_tag::ldnc);
        }

        EXPECT_ANY_THROW(T(ref_pd, {}));
        // run the non packed version
        T(ref_pd).execute(strm,
                {{DNNL_ARG_SRC_LAYER, src_layer_ref},
                        {DNNL_ARG_AUGRU_ATTENTION, augru_attention_ref},
                        {DNNL_ARG_SRC_ITER, src_iter_ref},
                        {DNNL_ARG_SRC_ITER_C, src_iter_c_ref},
                        {DNNL_ARG_WEIGHTS_LAYER, weights_layer_ref},
                        {DNNL_ARG_WEIGHTS_ITER, weights_iter_ref},
                        {DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_ref},
                        {DNNL_ARG_WEIGHTS_PROJECTION, weights_projection_ref},
                        {DNNL_ARG_BIAS, bias_ref},
                        {DNNL_ARG_DST_LAYER, dst_layer_ref},
                        {DNNL_ARG_DST_ITER, dst_iter_ref},
                        {DNNL_ARG_DST_ITER_C, dst_iter_c_ref}});
        strm.wait();

        // run the packed version
        auto tgt_pd = get_pd(p.aprop, p.extra.activation, p.direction,
                src_layer_md_tgt, src_iter_md_tgt, src_iter_c_md_tgt,
                attention_md_tgt, weights_layer_md_tgt, weights_iter_md_tgt,
                weights_peephole_md_tgt, weights_projection_md_tgt, bias_md_tgt,
                dst_layer_md_tgt, dst_iter_md_tgt, dst_iter_c_md_tgt,
                p.extra.alpha);
        testExecArgQueries(tgt_pd);
        test_primitive_param_queries(tgt_pd);

        EXPECT_ANY_THROW(T(tgt_pd, {}));
        T(tgt_pd).execute(strm,
                {{DNNL_ARG_SRC_LAYER, src_layer_tgt},
                        {DNNL_ARG_AUGRU_ATTENTION, augru_attention_tgt},
                        {DNNL_ARG_SRC_ITER, src_iter_tgt},
                        {DNNL_ARG_SRC_ITER_C, src_iter_c_tgt},
                        {DNNL_ARG_WEIGHTS_LAYER, weights_layer_tgt},
                        {DNNL_ARG_WEIGHTS_ITER, weights_iter_tgt},
                        {DNNL_ARG_WEIGHTS_PEEPHOLE, weights_peephole_tgt},
                        {DNNL_ARG_WEIGHTS_PROJECTION, weights_projection_tgt},
                        {DNNL_ARG_BIAS, bias_tgt},
                        {DNNL_ARG_DST_LAYER, dst_layer_tgt},
                        {DNNL_ARG_DST_ITER, dst_iter_tgt},
                        {DNNL_ARG_DST_ITER_C, dst_iter_c_tgt}});
        strm.wait();

        // compare dst_layer and dst_iter
        static constexpr data_t threshold = static_cast<data_t>(1e-5);
        compare_data<data_t>(dst_layer_ref, dst_layer_tgt, threshold);
        if (p.fmts.dst_iter_fmt != memory::format_tag::undef) {
            compare_data<data_t>(dst_iter_ref, dst_iter_tgt, threshold);
            if (std::is_same<T, lstm_forward>::value)
                compare_data<data_t>(dst_iter_c_ref, dst_iter_c_tgt, threshold);
        }
    }
};

/* RNN specializations */
template <>
memory::dim rnn_forward_test_t<vanilla_rnn_forward, float>::getNGates() {
    return 1;
}

template <>
vanilla_rnn_forward::primitive_desc
rnn_forward_test_t<vanilla_rnn_forward, float>::get_pd(prop_kind aprop,
        algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md, const memory::desc &,
        const memory::desc &, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return vanilla_rnn_forward::primitive_desc(get_test_engine(), aprop,
            activation, direction, src_layer_md, src_iter_md, weights_layer_md,
            weights_iter_md, bias_md, dst_layer_md, dst_iter_md, alpha);
}

/* LSTM specializations */
template <>
memory::dim rnn_forward_test_t<lstm_forward, float>::getNGates() {
    return 4;
}

template <>
lstm_forward::primitive_desc rnn_forward_test_t<lstm_forward, float>::get_pd(
        prop_kind aprop, algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md,
        const memory::desc &weights_peephole_md,
        const memory::desc &weights_projection_md, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return lstm_forward::primitive_desc(get_test_engine(), aprop, direction,
            src_layer_md, src_iter_md, src_iter_c_md, weights_layer_md,
            weights_iter_md, weights_peephole_md, weights_projection_md,
            bias_md, dst_layer_md, dst_iter_md, dst_iter_c_md);
}

template <>
bool rnn_forward_test_t<lstm_forward, float>::skipTest(bool src_layer_match,
        bool augru_attention_match, bool src_iter_match, bool src_iter_c_match,
        bool weights_layer_match, bool weights_iter_match, bool bias_match,
        bool dst_layer_match, bool dst_iter_match, bool dst_iter_c_match) {
    return src_layer_match && src_iter_match && src_iter_c_match
            && weights_layer_match && weights_iter_match && bias_match
            && dst_layer_match && dst_iter_match && dst_iter_c_match;
}

template <>
bool rnn_forward_test_t<augru_forward, float>::skipTest(bool src_layer_match,
        bool augru_attention_match, bool src_iter_match, bool src_iter_c_match,
        bool weights_layer_match, bool weights_iter_match, bool bias_match,
        bool dst_layer_match, bool dst_iter_match, bool dst_iter_c_match) {
    return src_layer_match && augru_attention_match && src_iter_match
            && src_iter_c_match && weights_layer_match && weights_iter_match
            && bias_match && dst_layer_match && dst_iter_match
            && dst_iter_c_match;
}

template <>
bool rnn_forward_test_t<lbr_augru_forward, float>::skipTest(
        bool src_layer_match, bool augru_attention_match, bool src_iter_match,
        bool src_iter_c_match, bool weights_layer_match,
        bool weights_iter_match, bool bias_match, bool dst_layer_match,
        bool dst_iter_match, bool dst_iter_c_match) {
    return src_layer_match && augru_attention_match && src_iter_match
            && src_iter_c_match && weights_layer_match && weights_iter_match
            && bias_match && dst_layer_match && dst_iter_match
            && dst_iter_c_match;
}

template <>
memory::desc rnn_forward_test_t<lstm_forward, float>::querySrcIterC(
        const lstm_forward::primitive_desc &rpd) {
    return rpd.src_iter_c_desc();
}

template <>
memory::desc rnn_forward_test_t<lstm_forward, float>::queryWeightsPeephole(
        const lstm_forward::primitive_desc &rpd) {
    return rpd.weights_peephole_desc();
}

template <>
memory::desc rnn_forward_test_t<lstm_forward, float>::queryWeightsProjection(
        const lstm_forward::primitive_desc &rpd) {
    return rpd.weights_projection_desc();
}

template <>
memory::desc rnn_forward_test_t<lstm_forward, float>::queryDstIterC(
        const lstm_forward::primitive_desc &rpd) {
    return rpd.dst_iter_c_desc();
}

/* GRU specializations */
template <>
memory::dim rnn_forward_test_t<gru_forward, float>::getNGates() {
    return 3;
}

template <>
gru_forward::primitive_desc rnn_forward_test_t<gru_forward, float>::get_pd(
        prop_kind aprop, algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md, const memory::desc &,
        const memory::desc &, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return gru_forward::primitive_desc(get_test_engine(), aprop, direction,
            src_layer_md, src_iter_md, weights_layer_md, weights_iter_md,
            bias_md, dst_layer_md, dst_iter_md);
}

/* LBR GRU specializations */
template <>
memory::dim rnn_forward_test_t<lbr_gru_forward, float>::getNGates() {
    return 3;
}

template <>
lbr_gru_forward::primitive_desc
rnn_forward_test_t<lbr_gru_forward, float>::get_pd(prop_kind aprop,
        algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md, const memory::desc &,
        const memory::desc &, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return lbr_gru_forward::primitive_desc(get_test_engine(), aprop, direction,
            src_layer_md, src_iter_md, weights_layer_md, weights_iter_md,
            bias_md, dst_layer_md, dst_iter_md);
}

/* AUGRU specializations */
template <>
memory::dim rnn_forward_test_t<augru_forward, float>::getNGates() {
    return 3;
}

template <>
augru_forward::primitive_desc rnn_forward_test_t<augru_forward, float>::get_pd(
        prop_kind aprop, algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md, const memory::desc &,
        const memory::desc &, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return augru_forward::primitive_desc(get_test_engine(), aprop, direction,
            src_layer_md, src_iter_md, attention_md, weights_layer_md,
            weights_iter_md, bias_md, dst_layer_md, dst_iter_md);
}

/* LBR AUGRU specializations */
template <>
memory::dim rnn_forward_test_t<lbr_augru_forward, float>::getNGates() {
    return 3;
}

template <>
lbr_augru_forward::primitive_desc
rnn_forward_test_t<lbr_augru_forward, float>::get_pd(prop_kind aprop,
        algorithm activation, rnn_direction direction,
        const memory::desc &src_layer_md, const memory::desc &src_iter_md,
        const memory::desc &src_iter_c_md, const memory::desc &attention_md,
        const memory::desc &weights_layer_md,
        const memory::desc &weights_iter_md, const memory::desc &,
        const memory::desc &, const memory::desc &bias_md,
        const memory::desc &dst_layer_md, const memory::desc &dst_iter_md,
        const memory::desc &dst_iter_c_md, float alpha) {
    return lbr_augru_forward::primitive_desc(get_test_engine(), aprop,
            direction, src_layer_md, src_iter_md, attention_md,
            weights_layer_md, weights_iter_md, bias_md, dst_layer_md,
            dst_iter_md);
}

using eng = engine::kind;
using fmt = memory::format_tag;
using alg = algorithm;
using dir = rnn_direction;
using rnn_forward_test_f32 = rnn_forward_test_t<vanilla_rnn_forward, float>;
using lstm_forward_test_f32 = rnn_forward_test_t<lstm_forward, float>;
using gru_forward_test_f32 = rnn_forward_test_t<gru_forward, float>;
using lbr_gru_forward_test_f32 = rnn_forward_test_t<lbr_gru_forward, float>;
using augru_forward_test_f32 = rnn_forward_test_t<augru_forward, float>;
using lbr_augru_forward_test_f32 = rnn_forward_test_t<lbr_augru_forward, float>;

using cfg_f32 = test_rnn_params_t;

#define PLAIN_RNN(a) \
    { a, 0.0f }
#define NOT_RNN \
    { alg::undef, 0.0f }

TEST_P(rnn_forward_test_f32, TestsRnn) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestRnn, rnn_forward_test_f32,
        ::testing::Values(
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 10, 16, 100, 100, 100, 100}},
                /* Check for invalid parameters: unsupported unrolling */
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {2, 1, 10, 16, 200, 100, 100, 100},
                        true, dnnl_invalid_arguments},
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {2, 1, 10, 16, 100, 200, 100, 100},
                        true, dnnl_invalid_arguments},
                /* Check for invalid parameters: inconsistent dimensions */
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {2, 1, 10, 16, 100, 100, 50, 100},
                        true, dnnl_invalid_arguments},
                /* Check if passing {src,dst}_iter impacts results */
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),

                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::undef, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}},
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}},
                cfg_f32 {PLAIN_RNN(alg::eltwise_tanh),
                        prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::undef, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}}));

TEST_P(lstm_forward_test_f32, TestsLSTM) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestLSTM, lstm_forward_test_f32,
        ::testing::Values(
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 10, 16, 100, 100, 100, 100}},
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo, fmt::ldgo,
                                fmt::undef, fmt::ldgo, fmt::tnc, fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 10, 16, 100, 100, 100, 100}},
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo, fmt::ldgo,
                                fmt::ldio, fmt::ldgo, fmt::tnc, fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 10, 16, 100, 100, 100, 100}},
                /* Non uniform sizes tests */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo, fmt::ldgo,
                                fmt::undef, fmt::ldgo, fmt::tnc, fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo, fmt::ldgo,
                                fmt::ldio, fmt::ldgo, fmt::tnc, fmt::ldnc},
                        test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 15}},
                /* Check if not passing dst_iter impacts results */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}}));

TEST_P(gru_forward_test_f32, TestsGRU) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestGRU, gru_forward_test_f32,
        ::testing::Values(cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                                  dir::unidirectional_left2right,
                                  {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                          fmt::undef, fmt::undef, fmt::ldgo,
                                          fmt::tnc, fmt::ldnc},
                                  test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                /* Check if not passing dst_iter impacts results */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}}));

TEST_P(lbr_gru_forward_test_f32, TestsGRUlbr) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestGRUlbr, lbr_gru_forward_test_f32,
        ::testing::Values(cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                                  dir::unidirectional_left2right,
                                  {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                          fmt::undef, fmt::undef, fmt::ldgo,
                                          fmt::tnc, fmt::ldnc},
                                  test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                /* Check if not passing dst_iter impacts results */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {3, 1, 5, 1, 4, 4, 4, 4}}));

TEST_P(augru_forward_test_f32, TestsAUGRU) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestAUGRU, augru_forward_test_f32,
        ::testing::Values(cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                                  dir::unidirectional_left2right,
                                  {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                          fmt::undef, fmt::undef, fmt::ldgo,
                                          fmt::tnc, fmt::ldnc},
                                  test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                /* Check if not passing dst_iter impacts results */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {1, 1, 5, 1, 4, 4, 4, 4}}));

TEST_P(lbr_augru_forward_test_f32, TestsAUGRUlbr) {}
CPU_INSTANTIATE_TEST_SUITE_P(TestAUGRUlbr, lbr_augru_forward_test_f32,
        ::testing::Values(cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                                  dir::unidirectional_left2right,
                                  {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                          fmt::undef, fmt::undef, fmt::ldgo,
                                          fmt::tnc, fmt::ldnc},
                                  test_rnn_sizes_t {1, 1, 1, 1, 10, 5, 5, 5}},
                /* Check if not passing dst_iter impacts results */
                cfg_f32 {NOT_RNN, prop_kind::forward_inference,
                        dir::unidirectional_left2right,
                        {fmt::tnc, fmt::ldnc, fmt::ldigo, fmt::ldigo,
                                fmt::undef, fmt::undef, fmt::ldgo, fmt::tnc,
                                fmt::undef},
                        test_rnn_sizes_t {1, 1, 5, 1, 4, 4, 4, 4}}));

} // namespace dnnl
