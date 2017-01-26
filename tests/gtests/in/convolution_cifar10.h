INST_TEST_CASE(Cifar10_Blocked,
    PARAMS(nchw, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED,
        2, 1, 3, 32, 32, 32, 32, 32, 5, 5, 2, 2, 1, 1)
);

INST_TEST_CASE(Cifar10_Blocked16,
    PARAMS(nchw, Ohwi8o, FMT_BIAS, FMT_DATA_BLOCKED16,
        2, 1, 3, 32, 32, 32, 32, 32, 5, 5, 2, 2, 1, 1)
);
