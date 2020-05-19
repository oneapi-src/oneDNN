enum policy {
  // best optimization possible. Minimum control to framework
  max_perf_policy = 0,
  // must-to-have fusion, give framework executor more control
  fusion_policy,
  // no optimization. Maximum control to framework
  debug_policy
};
