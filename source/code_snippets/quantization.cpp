// Quantization number representation
enum llga_q_rep {
  LLGA_Q_REP_FIXED_POINT_SIGNED,
  LLGA_Q_REP_FIXED_POINT_UNSIGNED,
  LLGA_Q_REP_AFFINE_SIGNED,
  LLGA_Q_REP_AFFINE_UNSIGNED,
  LLGA_Q_REP_FLOATING_POINT,
  LLGA_Q_REP_BFLOAT,
};

// Quantization scheme
enum llga_q_scheme {
  LLGA_Q_SCHEME_STATIC,
  LLGA_Q_SCHEME_DYN,
};

// Quantization capability
struct llga_q_cap {
  // bit mask for q10n number presentation capabilities
  unsigned long q_rep_bits;
  // bit mask for q10n scheme capabilities
  unsigned long q_scheme_bits;
  // number of precision bits
  int precision;
  // q10n granularity, only applicable for integer math, i.e.
  // fixed point and affine number representations.
  // -1: only per-tensor q10n supported
  // otherwise, the channel axis for
  // per-channel q10n besides per-tensor q10n
  int axis;
};
typedef std::vector<llga_q_cap> llga_q_caps;

struct llga_q_config_for_op {
  // name of the op, support wildcards with “*” and “?”
  std::string op_expr;
  llga_q_caps caps;
  // q10n caps for input and output tensors.
  // Only valid when only one op is specified.
  // Wildcard index with -1 for all tensors.
  // Concrete index overrides wildcard index.
  // Output tensors are ordered behind input tensors.
  std::map<int, llga_q_caps> tensor_caps;
};

struct llga_q_config_for_fusion {
  // TODO: specify fusion pattern description
  std::string fusion_pattern;
  llga_q_caps caps;
  // q10n caps for input/output and intermediate tensors.
  // TODO: further specify with pattern description defined.
  std::map<int, llga_q_caps> tensor_caps;
};

// API for getting the globally default q10n capabilities.
llga_q_caps get_default_q_caps(Engine* engine);

// API for getting the q10n capabilities for particular ops.
std::vector<llga_q_config_for_op > get_q_caps_for_ops(Engine* engine);

// API for getting the q10n capabilities for fusion patterns.
std::vector<llga_q_config_for_fusion > get_q_caps_for_patterns(Engine* engine);

