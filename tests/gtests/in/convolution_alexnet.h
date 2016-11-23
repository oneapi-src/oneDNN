INSTANTIATE_TEST_CASE_P(AlexnetNCHW, convolution_test, ::testing::Values(
    PARAMS(nchw, oihw, FMT_BIAS, nchw,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1),
    PARAMS(nchw, oihw, FMT_BIAS, nchw,
        2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(nchw, goihw, FMT_BIAS, nchw,
        2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1)
));

INSTANTIATE_TEST_CASE_P(AlexnetBlocked, convolution_test, ::testing::Values(
    PARAMS(nchw, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(nhwc, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 227, 227, 96, 55, 55, 11, 11, 0, 0, 4, 4),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 96, 27, 27, 256, 27, 27, 5, 5, 2, 2, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 256, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 384, 13, 13, 384, 13, 13, 3, 3, 1, 1, 1, 1),
    PARAMS(FMT_DATA_BLOCKED, FMT_WEIGHTS_BLOCKED_G, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 2, 384, 13, 13, 256, 13, 13, 3, 3, 1, 1, 1, 1)
));
