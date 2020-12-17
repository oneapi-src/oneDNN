# Naming Conventions {#dev_guide_conventions}

oneDNN Graph specification relies on a set of standard naming conventions for
variables. This section describes these conventions.

## Variable (Tensor) Names

| Name                    | Meaning
| :-                      | :-
| `input`                 | Input tensor
| `output`                | Output tensor
| `filter`                | Weight tensor used in convolution
| `bias`                  | Bias tensor used in convolution
| `gamma`                 | Scale tensor used in normalization
| `beta`                  | Shift tensor used in normalization
| `input_delta`           | Gradient tensor with respect to the input
| `output_delta`          | Gradient tensor with respect to the output
| `filter_delta`          | Gradient tensor with respect to the weight
| `bias_delta`            | Gradient tensor with respect to the bias
| `gamma_delta`           | Gradient tensor with respect to the gamma
| `beta_delta`            | Gradient tensor with respect to the beta
| `input_forward`         | Original input for backward op
| `result_forward`        | Original output for backward op
