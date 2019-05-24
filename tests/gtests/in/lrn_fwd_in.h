#if defined(FP16) || defined(FP32)
#ifdef FP32
#define INSTANTIATE_TEST INSTANTIATE_TEST_SUITE_P
#endif

#ifdef FP16
#define INSTANTIATE_TEST GPU_INSTANTIATE_TEST_SUITE_P
#endif

TEST_P(lrn_forward_test_type, TestsLRN)
{
}

INSTANTIATE_TEST(TestLRNForwardZeroDim, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 0, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS }}
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 0, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS }}
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nChw16c, { 2, 16, 0, 4, 5, 1.0e-4f, 0.75f, 3.0f, ACROSS }}
            ));

INSTANTIATE_TEST(TestLRNForwardEF, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { -1, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS },
            true, mkldnn_invalid_arguments }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, -10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS },
            true, mkldnn_invalid_arguments }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 10, -4, 4, 5, 1.0e-4f, 0.75f, 3.0f, ACROSS },
            true, mkldnn_invalid_arguments }
            ));

INSTANTIATE_TEST(TestLRNForward_nChw16c_padded, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 17, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 19, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            ));

INSTANTIATE_TEST(TestLRNForward_nChw8c_padded, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 7, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 9, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 26, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 12, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            ));

INSTANTIATE_TEST(TestLRNForward, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 3.0f, ACROSS } }
            ));

INSTANTIATE_TEST(TestLRNForwardNHWC, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nhwc,
            memory::format_tag::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nhwc,
            memory::format_tag::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nhwc,
            memory::format_tag::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.85f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nhwc,
            memory::format_tag::nhwc, { 2, 10, 4, 4, 5, 1.0e-4f, 0.75f, 4.85f, ACROSS } }
            ));

INSTANTIATE_TEST(TestLRNForward_nChw8c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            ));

INSTANTIATE_TEST(TestLRNForward_nChw16c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 16, 4, 4, 5, 1.0e-4f, 0.75f, 5.7f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNAlexnetForwardNCHW, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNAlexnetForwardNHWC, lrn_forward_test_type,
        ::testing::Values(
                lrn_fwd_test_params{ prop_kind::forward_training,
                algorithm::lrn_across_channels, memory::format_tag::nhwc,
                memory::format_tag::nhwc, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
                lrn_fwd_test_params{ prop_kind::forward_scoring,
                algorithm::lrn_across_channels, memory::format_tag::nhwc,
                memory::format_tag::nhwc, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
                lrn_fwd_test_params{ prop_kind::forward_training,
                algorithm::lrn_across_channels, memory::format_tag::nhwc,
                memory::format_tag::nhwc, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
                lrn_fwd_test_params{ prop_kind::forward_scoring,
                algorithm::lrn_across_channels, memory::format_tag::nhwc,
                memory::format_tag::nhwc, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNAlexnetForward_nChw8c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNAlexnetForward_nChw16c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNGoogleNetV1ForwardNCHW, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNGoogleNetV1Forward_nChw8c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNGoogleNetV1Forward_nChw16c, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } },
            lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_across_channels, memory::format_tag::nChw16c,
            memory::format_tag::nChw16c, { 2, 192, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));

INSTANTIATE_TEST(
        TestLRNRCNNForwardBlocked, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 3, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 3, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 3, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 3, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 96, 55, 55, 5, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            , lrn_fwd_test_params{ prop_kind::forward_scoring,
            algorithm::lrn_within_channel, memory::format_tag::nChw8c,
            memory::format_tag::nChw8c, { 2, 256, 27, 27, 5, 1.0e-4f, 0.75f, 1.0f, WITHIN } }
            ));

// This tests compatibility with MKL-DNN 0.14
INSTANTIATE_TEST(
        TestLRNRegressionWeightFormat, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::oihw,
            memory::format_tag::oihw, { 2, 64, 56, 56, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
        ));

INSTANTIATE_TEST(
        TestLRNForwardNCHWTail, lrn_forward_test_type,
        ::testing::Values(
            lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 1, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 2, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 3, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 4, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 5, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 9, 6, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 7, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            , lrn_fwd_test_params{ prop_kind::forward_training,
            algorithm::lrn_across_channels, memory::format_tag::nchw,
            memory::format_tag::nchw, { 1, 64, 8, 9, 5, 1.0e-4f, 0.75f, 1.0f, ACROSS } }
            ));
#endif
