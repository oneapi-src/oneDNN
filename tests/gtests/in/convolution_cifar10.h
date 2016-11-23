INSTANTIATE_TEST_CASE_P(Cifar10Blocked, convolution_test, ::testing::Values(
    PARAMS(nchw, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 32, 32, 32, 32, 32, 5, 5, 2, 2, 1, 1)
));
